from typing import List, Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from embed import query_db


class ChatAgent:
    def __init__(self, model_name: str = "qwen3:4B", base_url: str = "http://localhost:11434"):
        """
        初始化聊天智能体

        Args:
            model_name: 使用的Ollama模型名称
            base_url: Ollama服务的基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.chain = self._build_chain()

    def _build_chain(self) -> Any:
        """
        构建聊天处理链

        Returns:
            配置好的LangChain处理链
        """
        # 1. 定义提示模板
        template = """
        你是一个专业的问答助手，请基于以下上下文信息回答问题。
        如果上下文信息不足以回答问题，请说明你不知道。
        保持回答简洁、专业且准确。

        上下文:
        {context}

        问题: 
        {question}

        回答:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 2. 初始化Ollama聊天模型
        ollama_llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.05  # 控制创造性，值越低越保守
        )

        # 3. 构建处理链
        return (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | ollama_llm
                | StrOutputParser()
        )

    def chat(self, question: str, context: List[str]) -> str:
        """
        基于上下文回答问题

        Args:
            question: 用户问题
            context: 相关上下文信息列表

        Returns:
            模型的回答
        """
        # 将上下文合并为单个字符串
        context_str = "\n\n".join(context)
        return self.chain.invoke({"context": context_str, "question": question})

    def rag_chat(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        完整的RAG聊天流程，并提供详细的来源信息。
        """
        # 1. 从向量库检索相关上下文
        retrieved_results = query_db(question, n_results)
        context_docs = retrieved_results["documents"][0]
        context_metadatas = retrieved_results["metadatas"][0]

        # 2. 使用智能体生成回答
        answer = self.chat(question, context_docs)

        # 3. 构建更丰富的来源信息
        sources = [
            f"来源: {meta.get('source', 'N/A')}, 块: {meta.get('chunk_index', 'N/A')}"
            for meta in context_metadatas
        ]

        return {
            "question": question,
            "answer": answer,
            "context": context_docs,
            "sources": sources  # 返回详细来源
        }


def main():
    # 初始化聊天智能体
    agent = ChatAgent(model_name="qwen3:4B")

    print("=" * 50)
    print("RAG 智能体已启动 (输入 'exit' 退出)")
    print("=" * 50)

    while True:
        # 获取用户输入
        user_input = input("\n你的问题: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("对话结束。")
            break

        if not user_input:
            continue

        # 处理查询
        response = agent.rag_chat(user_input)

        # 显示结果
        print("\n" + "=" * 50)
        print(f"问题: {response['question']}")
        print("-" * 50)
        print(f"回答: {response['answer']}")
        print("-" * 50)
        print("参考上下文:")
        for i, doc in enumerate(response["context"]):
            print(f"[文档 {i + 1}]: {doc[:100]}...")
        print("=" * 50)


if __name__ == "__main__":
    main()