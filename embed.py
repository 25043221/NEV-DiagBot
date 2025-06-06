import chromadb
import langchain_ollama
from langchain_ollama import OllamaEmbeddings
import chunk



embedding = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")



def embed_text(text: str) -> list:
    """
    使用Ollama模型嵌入文本。

    Args:
        text (str): 要嵌入的文本。

    Returns:
        list: 嵌入后的文本向量。
    """
    return embedding.embed_query(text)


chromadb_client = chromadb.PersistentClient("./chroma_db")
chromadb_connection = chromadb_client.get_or_create_collection("my_collection")
def create_db() -> None:
    """
    创建一个新的ChromaDB集合，并将嵌入的文本存储在其中。
    """
    for idx, text in enumerate(chunk.get_chunks()):
        vector = embed_text(text)
        chromadb_connection.add(
            ids=str(idx),  # 使用文本本身作为ID
            documents=[text],  #原文
            embeddings=[vector],  #原文的embedding
        )
    print("数据库创建成功，已存储嵌入向量。")

def query_db(query: str, n_results: int = 3) -> dict:
    """
    查询ChromaDB，并返回文档及其元数据。
    """
    query_embed = embed_text(query)
    results = chromadb_connection.query(
        query_embeddings=[query_embed],
        n_results=n_results
    )
    return results # 直接返回整个 results 字典

# if __name__ == "__main__":
#     print(chromadb_connection.get())  # 获取集合中的所有数据和元数据
    # 首先运行创建数据库
    # create_db()
    # 测试嵌入函数
    # chunk = chunk.get_chunks()
    # print(chunk)
    # embedded_vector = embed_text(chunk)
    # print(f"嵌入后的向量: {embedded_vector}")
#     query = "理财"
#     create_db()
#     chunks = query_db(query)
#     prompt = f"请根据Context回答用户的问题：\n"
#     prompt += f"query:{query}\n"
#     prompt += f"Context:\n"
#     for chunk in chunks:
#         prompt += f"{chunk}\n"
#         prompt += "----------------\n"
#     # print(prompt)
#
#     llm = llm = langchain_ollama.ChatOllama(
#     model="qwen3:4B",
#     base_url="http://localhost:11434",
#     temperature=0.05,
# )
#     response = llm.invoke(prompt)
#     print(f"回答：{response}")





