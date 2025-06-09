
import chromadb
from langchain_ollama import OllamaEmbeddings

from chunk import get_text_chunks

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
    if chromadb_connection.count() > 0:
        print("数据库已存在，跳过创建。")
        return
    for idx, text in enumerate(get_text_chunks()):
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

if __name__ == "__main__":
    # 测试嵌入数据库创建
    print("正在创建嵌入数据库...")
    create_db()

    # 测试查询
    test_query = "如何更换电动汽车电池？"
    print(f"\n测试查询: {test_query}")
    result = query_db(test_query)

    # 展示查询结果
    print("\n查询结果：")
    for i, doc in enumerate(result.get("documents", [[]])[0]):
        print(f"结果 {i + 1}:\n{doc}\n{'-' * 30}")
