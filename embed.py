# from tkinter import Image
# from typing import List, Union
#
# import chromadb
# import langchain_ollama
# import ollama
# from langchain_ollama import OllamaEmbeddings
# from ollama import embed
#
# import chunk


# embedding = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
#
# def embed_text(text: str) -> list:
#     """
#     使用Ollama模型嵌入文本。
#
#     Args:
#         text (str): 要嵌入的文本。
#
#     Returns:
#         list: 嵌入后的文本向量。
#     """
#     return embedding.embed_query(text)
#
#
# chromadb_client = chromadb.PersistentClient("./chroma_db")
# chromadb_connection = chromadb_client.get_or_create_collection("my_collection")
# def create_db() -> None:
#     """
#     创建一个新的ChromaDB集合，并将嵌入的文本存储在其中。
#     """
#     if chromadb_connection.count() > 0:
#         print("数据库已存在，跳过创建。")
#         return
#     for idx, text in enumerate(chunk.get_chunks()):
#         vector = embed_text(text)
#         chromadb_connection.add(
#             ids=str(idx),  # 使用文本本身作为ID
#             documents=[text],  #原文
#             embeddings=[vector],  #原文的embedding
#         )
#     print("数据库创建成功，已存储嵌入向量。")
#
# def query_db(query: str, n_results: int = 3) -> dict:
#     """
#     查询ChromaDB，并返回文档及其元数据。
#     """
#     query_embed = embed_text(query)
#     results = chromadb_connection.query(
#         query_embeddings=[query_embed],
#         n_results=n_results
#     )
#     return results # 直接返回整个 results 字典
#
# if __name__ == "__main__":
# #     print(chromadb_connection.get())  # 获取集合中的所有数据和元数据
#     # 测试嵌入函数
#     # chunk = chunk.get_chunks_from_pdf()
#     # print(chunk)
#     # create_db()
#     query = "比亚迪"
#     # embedded_vector = embed_text(query)
#     # print(f"嵌入后的向量: {embedded_vector}")
#     chunks = query_db(query)
#     prompt = f"请根据Context回答用户的问题：\n"
#     prompt += f"query:{query}\n"
#     prompt += f"Context:\n"
#     for chunk in chunks:
#         prompt += f"{chunk}\n"
#         prompt += "----------------\n"
#     print(prompt)

#第二版-采用DC1LEX/nomic-embed-text-v1.5-multimodal:latest模型
import chromadb
from langchain_ollama import OllamaEmbeddings
from PIL import Image
from typing import List, Any, Dict, Union
import chunk
import base64
import io
import os

# --- Ollama-hosted Nomic Embeddings Initialization ---
embedding = OllamaEmbeddings(
    model="DC1LEX/nomic-embed-text-v1.5-multimodal:latest",
    base_url="http://localhost:11434"
)

# --- 新增辅助函数：PIL Image 转 Base64 字符串 ---
def _pil_image_to_base64(image: Image.Image, format="jpeg") -> str:
    """
    将 PIL Image 对象转换为 Base64 编码的字符串。
    默认使用 JPEG 格式，可以根据需要更改。
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return f"data:image/{format};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


# --- ChromaDB Client Initialization (remains the same) ---
chromadb_client = chromadb.PersistentClient("./chroma_db")
chromadb_collection = chromadb_client.get_or_create_collection("my_collection")


# --- Helper function: Convert multimodal chunk to string (remains the same) ---
def convert_multimodal_chunk_to_string(multimodal_chunk: List[Union[str, Image.Image]]) -> str:
    """
    将多模态切片（包含文本和PIL图像）转换为一个可供ChromaDB存储的字符串表示。
    """
    parts = []
    for item in multimodal_chunk:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Image.Image):
            parts.append("[图片]")
    return "\n".join(parts).strip()


# --- Embedding Functions ---

def embed_multimodal_chunk(multimodal_chunk_data: List[Union[str, Image.Image]]) -> List[float]:
    """
    使用 Ollama-hosted Nomic Embed Multimodal 模型嵌入多模态切片。
    将多模态切片转换为一个包含文本和Base64编码图像的单一字符串，然后进行嵌入。

    Args:
        multimodal_chunk_data (List[Union[str, Image.Image]]): 要嵌入的多模态切片。

    Returns:
        List[float]: 嵌入后的向量。
    """
    try:
        combined_content_for_embedding = []
        for item in multimodal_chunk_data:
            if isinstance(item, str):
                combined_content_for_embedding.append(item)
            elif isinstance(item, Image.Image):
                base64_image_str = _pil_image_to_base64(item)
                combined_content_for_embedding.append(base64_image_str)

        final_document_string_for_embedding = "\n".join(combined_content_for_embedding).strip()

        embeddings = embedding.embed_documents(
            [final_document_string_for_embedding]
        )
        return embeddings[0]
    except Exception as e:
        print(f"Warning: 嵌入多模态切片时发生错误: {e}")
        return []


def embed_query_text(query: str) -> List[float]:
    """
    使用 Ollama-hosted Nomic Embed Multimodal 模型嵌入查询文本。

    Args:
        query (str): 要嵌入的查询文本。

    Returns:
        List[float]: 嵌入后的查询向量。
    """
    try:
        return embedding.embed_query(query)
    except Exception as e:
        print(f"Warning: 嵌入查询文本时发生错误: {e}")
        return []


# --- Create Database Function (remains mostly the same, uses new embed functions) ---
def create_db(pdf_path: str) -> None:
    """
    创建一个新的ChromaDB集合，并将嵌入的多模态内容存储在其中。
    """
    if chromadb_collection.count() > 0:
        print("数据库已存在，跳过创建。")
        return

    print("开始切分PDF内容为多模态块...")
    multimodal_chunks = chunk.chunk_multimodal_content(
        pdf_path, # 传递pdf_path
        max_chunk_text_length=700,
        text_chunk_overlap=100,
        layout_proximity_threshold=60
    )
    print(f"PDF切分完成，共生成 {len(multimodal_chunks)} 个多模态切片。")

    print("开始将多模态切片嵌入并存储到ChromaDB...")
    for idx, multi_chunk in enumerate(multimodal_chunks):
        if not multi_chunk:
            continue

        document_string = convert_multimodal_chunk_to_string(multi_chunk)

        vector = embed_multimodal_chunk(multi_chunk)

        if not vector:
            print(f"Warning: 无法为切片 {idx} 生成嵌入向量，跳过。")
            continue

        # 尝试从chunk中获取页码信息并存储到metadata中
        # 假设chunk.py的extract_elements_with_bbox会为每个元素添加'page_num'
        # 我们可以尝试从multi_chunk的第一个元素中获取页码
        page_num = -1
        if multi_chunk and hasattr(multi_chunk[0], 'page_num'): # 假设这里可以访问到原始的带有page_num的元素
             # 注意：由于multi_chunk现在是Union[str, Image.Image]的列表，
             # 直接从其中的元素获取page_num可能需要额外的处理
             # 更合理的做法是在chunk.py生成multimodal_chunks时就将page_num作为metadata的一部分
             # 或者在convert_multimodal_chunk_to_string中将page_num编码进document_string
             # 这里先简化处理，如果能从metadata中获取，就在metadata中添加
            pass # 在这里暂时不处理，因为chunk.py的输出结构需要调整才能方便获取页码

        chromadb_collection.add(
            ids=str(idx),
            documents=[document_string],
            embeddings=[vector],
            metadatas=[{'page_num': multi_chunk[0].get('page_num', 'N/A') if isinstance(multi_chunk[0], dict) else 'N/A'}] # 假设multi_chunk的元素是字典
        )
        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(multimodal_chunks)} 个切片。")

    print("数据库创建成功，已存储多模态嵌入向量。")


# --- Query Database Function (remains the same, uses new embed functions) ---
def query_db(query: str, n_results: int = 3) -> dict:
    """
    查询ChromaDB，并返回文档及其元数据。
    """
    query_embed = embed_query_text(query)
    if not query_embed:
        print("查询嵌入失败，无法进行查询。")
        return {"ids": [], "distances": [], "documents": [], "metadatas": []}

    results = chromadb_collection.query(
        query_embeddings=[query_embed],
        n_results=n_results,
        include=['documents', 'distances', 'metadatas']
    )
    return results


# --- Main execution example ---
if __name__ == "__main__":
    pdf_file_path = "/home/yu/zjh/my_app/data/秦plusDMi用户手册.pdf"

    create_db(pdf_file_path)

    print("\n--- 执行查询 ---")
    user_query = "充电说明"

    results = query_db(user_query, n_results=5)

    print(f"\n查询: \"{user_query}\" 的结果:")
    if results and results.get('documents') and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_content = results['documents'][0][i]
            dist_score = results['distances'][0][i]
            meta = results['metadatas'][0][i] # 获取metadata
            print(f"  结果 {i + 1}:")
            print(f"    距离: {dist_score:.4f}")
            print(f"    文档内容: {doc_content[:300]}...")
            print(f"    元数据: {meta}")
    else:
        print("未找到相关结果。请检查数据库是否已成功创建或查询内容。")



