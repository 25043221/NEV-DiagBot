from typing import List, Any, Dict
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

#第一版的直接读取-原型设计
# def read_data() -> str:
#     """
#     读取数据文件内容
#     :return: 文件内容字符串
#     """
#     with open("/home/yu/zjh/my_app/data/秦plusDMi用户手册.pdf", "r", encoding="utf-8") as file:
#         return file.read()
#
# def get_chunks() -> List[str]:
#     """
#     将文本分割成指定大小的块
#     :param text: 输入文本
#     :param chunk_size: 每个块的大小
#     :return: 文本块列表
#     """
#     content = read_data()
#     chunks = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50).split_text(content)
#     return chunks
#
# if __name__ == "__main__":
#     data = read_data()
#     chunks:  List[str] = get_chunks()
#     for data in chunks:
#         print(data)
#         print("----------------")

#第二版-采用“布局感知”进行多模态切分，适配Nomic Embed Multimodal嵌入模型
from typing import List, Dict, Any, Union
import pdfplumber
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io


def extract_elements_with_bbox(pdf_path: str) -> List[Dict[str, Any]]:
    """
    从PDF中提取带有包围盒（bounding box）的文本块和图像元素。
    """
    elements = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 提取文本块，使用 extract_text_blocks() 获取更精细的文本块信息（包括bbox）
            for text_block in page.extract_text_lines():
                if text_block['text'].strip():  # 确保文本内容非空
                    elements.append({
                        "type": "text",
                        "content": text_block['text'],
                        "bbox": (text_block["x0"], text_block["top"], text_block["x1"], text_block["bottom"]),  # (x0, y0, x1, y1)
                        "page_num": i + 1
                    })

            # 提取页面图像
            for img_info in page.images:
                try:
                    # 裁剪图像区域并转换为 PIL.Image.Image 对象
                    x0, y0, x1, y1 = img_info['x0'], img_info['y0'], img_info['x1'], img_info['y1']
                    # 确保包围盒在页面边界内且有效
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(page.width, x1), min(page.height, y1)

                    if x1 > x0 and y1 > y0:  # 确保尺寸有效
                        img_object = page.crop((x0, y0, x1, y1)).to_image().original
                        elements.append({
                            "type": "image",
                            "content": img_object,  # PIL Image 对象
                            "bbox": (x0, y0, x1, y1),
                            "page_num": i + 1
                        })
                except Exception as e:
                    print(f"Warning: Could not extract image from page {i + 1} due to {e}")

    # 按照页面号和垂直位置（y0坐标）对所有元素进行排序，以确保阅读顺序
    elements.sort(key=lambda x: (x["page_num"], x["bbox"][1]))
    return elements


def chunk_multimodal_content(
        pdf_path: str,
        max_chunk_text_length: int = 700,  # 每个多模态块中纯文本内容的最大字符数软限制
        text_chunk_overlap: int = 100,  # 对纯文本块进行递归切分时的重叠量
        layout_proximity_threshold: int = 60  # 像素值，用于判断元素垂直距离是否足够近以进行分组
) -> List[List[Union[str, Image.Image]]]:
    """
    根据布局和语义将多模态PDF内容切分为适合 Nomic Embed Multimodal 的块。
    每个返回的块是一个 List[Union[str, PIL.Image.Image]]，代表一个要嵌入的单元。
    """
    all_elements = extract_elements_with_bbox(pdf_path)

    final_nomic_chunks: List[List[Union[str, Image.Image]]] = []
    current_interleaved_chunk: List[Union[str, Image.Image]] = []
    current_chunk_text_length = 0

    # 用于处理纯文本部分的递归切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_text_length,
        chunk_overlap=text_chunk_overlap
    )

    prev_element_y1 = -1  # 上一个元素的底部y坐标
    prev_element_page_num = -1  # 上一个元素的页码

    for i, element in enumerate(all_elements):
        element_content = element["content"]
        element_type = element["type"]
        element_y0 = element["bbox"][1]  # 当前元素的顶部y坐标
        element_y1 = element["bbox"][3]  # 当前元素的底部y坐标
        current_page_num = element["page_num"]

        start_new_chunk = False  # 标记是否应该开始新的切片

        # 判断是否需要开始新的切片：
        # 1. 遇到页码变化 (换页)
        if current_page_num != prev_element_page_num:
            start_new_chunk = True
        # 2. 当前元素与上一个元素之间存在显著的垂直间距 (基于布局的切分)
        elif prev_element_y1 != -1 and (element_y0 - prev_element_y1 > layout_proximity_threshold):
            start_new_chunk = True
        # 3. 如果当前块是纯文本，并且加入新文本后会超过最大文本长度限制
        elif element_type == "text" and (current_chunk_text_length + len(element_content) > max_chunk_text_length) and \
                not any(isinstance(x, Image.Image) for x in current_interleaved_chunk):
            start_new_chunk = True

        # 如果需要开始新切片，并且当前切片不为空，则将当前切片添加到最终列表中
        if start_new_chunk and current_interleaved_chunk:
            # 检查当前切片是否是纯文本且过长，如果是则进行递归切分
            combined_text_in_current_chunk = "".join([str(s) for s in current_interleaved_chunk if isinstance(s, str)])
            has_images_in_current_chunk = any(isinstance(x, Image.Image) for x in current_interleaved_chunk)

            if not has_images_in_current_chunk and len(combined_text_in_current_chunk) > max_chunk_text_length:
                # 纯文本且过长，使用 RecursiveCharacterTextSplitter 切分
                sub_chunks_text = text_splitter.split_text(combined_text_in_current_chunk.strip())
                for sub_text in sub_chunks_text:
                    final_nomic_chunks.append([sub_text])  # 每个子文本块作为独立的切片
            else:
                # 混合内容或未超长的纯文本，直接添加为切片
                final_nomic_chunks.append(current_interleaved_chunk)

            # 重置当前切片和文本长度计数
            current_interleaved_chunk = []
            current_chunk_text_length = 0

        # 将当前元素添加到当前切片中
        current_interleaved_chunk.append(element_content)
        if element_type == "text":
            current_chunk_text_length += len(element_content)

        # 更新前一个元素的坐标和页码
        prev_element_y1 = element_y1
        prev_element_page_num = current_page_num

    # 处理循环结束后可能存在的最后一个切片
    if current_interleaved_chunk:
        combined_text_in_last_chunk = "".join([str(s) for s in current_interleaved_chunk if isinstance(s, str)])
        has_images_in_last_chunk = any(isinstance(x, Image.Image) for x in current_interleaved_chunk)

        if not has_images_in_last_chunk and len(combined_text_in_last_chunk) > max_chunk_text_length:
            for sub_text in text_splitter.split_text(combined_text_in_last_chunk.strip()):
                final_nomic_chunks.append([sub_text])
        else:
            final_nomic_chunks.append(current_interleaved_chunk)

    return final_nomic_chunks

# --- 使用示例 ---
if __name__ == "__main__":
    # 请将此路径替换为您的实际用户手册PDF文件路径
    pdf_file_path = "/home/yu/zjh/my_app/data/秦plusDMi用户手册.pdf"

    print("开始进行多模态内容切分...")
    # 您可以根据手册的实际布局调整这些参数
    chunks_for_nomic = chunk_multimodal_content(
        pdf_file_path,
        max_chunk_text_length=700,  # 单个多模态块中纯文本内容的软限制
        text_chunk_overlap=100,  # 对超长纯文本块进行切分时的重叠量
        layout_proximity_threshold=60  # 布局感知阈值，以像素为单位，用于判断元素是否属于同一逻辑块
    )

    print(f"总共生成了 {len(chunks_for_nomic)} 个多模态切片。")

    # 打印前5个切片的结构示例，以便您理解输出格式
    print("\n--- 前5个切片示例 ---")
    for i, chunk in enumerate(chunks_for_nomic[:5]):
        print(f"\n--- 切片 {i + 1} ---")
        current_chunk_text_chars = 0
        current_chunk_images_count = 0
        for item in chunk:
            if isinstance(item, str):
                current_chunk_text_chars += len(item)
            elif isinstance(item, Image.Image):
                current_chunk_images_count += 1

        print(f"  切片包含 {current_chunk_images_count} 张图片和 {current_chunk_text_chars} 个字符的文本内容。")

        # 打印切片中第一个元素的部分内容，以便直观感受
        if chunk:
            first_item = chunk[0]
            if isinstance(first_item, str):
                print(f"  第一个元素（文本）: \"{first_item[:100]}...\"")  # 打印前100个字符
            elif isinstance(first_item, Image.Image):
                print(f"  第一个元素（图片）: 尺寸={first_item.size}, 模式={first_item.mode}")
