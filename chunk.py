from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_data() -> str:
    """
    读取数据文件内容
    :return: 文件内容字符串
    """
    with open("fin.txt", "r", encoding="utf-8") as file:
        return file.read()

# def get_chunks(text: str, chunk_size: int = 256, chunk_overlap: int = 50) -> List[str]:
#     """
#     将文本分割成指定大小的块
#     :param text: 输入文本
#     :param chunk_size: 每个块的大小
#     :param chunk_overlap: 重叠部分的大小
#     :return: 文本块列表
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return text_splitter.split_text(text)
def get_chunks() -> List[str]:
    """
    将文本分割成指定大小的块
    :param text: 输入文本
    :param chunk_size: 每个块的大小
    :return: 文本块列表
    """
    content = read_data()
    chunks = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50).split_text(content)
    return chunks

#
# if __name__ == "__main__":
#     data = read_data()
#     chunks:  List[str] = get_chunks()
#     for data in chunks:
#         print(data)
#         print("----------------")