import os
from typing import List, Any, Dict, Union

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.tools import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from embed import query_db
from load_key import load_key

from sentence_transformers import CrossEncoder

os.environ["TAVILY_API_KEY"] = load_key("TAVILY_API_KEY")


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
        self.store = {}
        print("初始化Re-ranking中...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("Re-ranking初始化完成.")
        self.chain = self._build_chain()


    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _build_chain(self) -> Any:
        """
        构建聊天处理链，支持工具调用和多轮对话。
        """
        # 1. 定义工具
        # Tavily搜索引擎工具
        tavily_tool = TavilySearchResults(max_results=3)
        tools = [tavily_tool]

        # 2. 初始化Ollama聊天模型
        llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.3
        )

        # 3. 定义Agent的提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """您是一位专业且富有同理心的新能源汽车诊断与知识助手。
                您的目标是帮助车主理解和解决他们的问题。

                请您综合利用您拥有的“上下文信息”和可用的“工具”来回答用户的问题。
                重要：如果“上下文信息”为空或没有直接相关的答案，请您优先尝试使用“工具”进行网络搜索以获取最新或更广泛的信息。如果工具也未能提供相关信息，则请您根据自己的通用知识进行回答。**
                如果您使用了工具，请在回答中清晰地说明您是如何获取到这些信息的。

                如果涉及故障诊断，请提供易懂的解释、可能的故障原因、初步的排查建议，以及后续的行动指南（如是否需要立即检修、是否可以继续行驶等）。
                请确保您的回答：
                1. 人性化和有同理心：开头可以使用亲切的语气，理解用户的困扰。
                2. 易懂且具体：避免过于专业的术语，用生活化语言解释复杂概念。
                3. 可执行：给出明确的行动步骤或建议，让用户知道“下一步该做什么”。
                4. 有信任感：提供必要的警示（如安全风险）、推荐专业服务，并在信息不足时坦诚告知。
                5. 故障码处理（重要）：如果用户提及故障码，请务必提供：
                    - 故障码的白话文解释。
                    - 可能的根本原因。
                    - 您基于现有知识的初步判断（例如，是否常见、是否紧急）。
                    - 初步的排查或缓解措施（如果存在）。
                    - 明确的维修建议和潜在费用范围（如果信息充足，可以预估一个大致范围）。
                6. 保养/功能咨询：提供详细的步骤、注意事项和最佳实践。
                7. 避免编造信息：如果您不确定，请坦诚告知用户，并建议他们咨询专业人士。

                上下文信息:
                {context}
                回答时，如果你的答案基于“上下文信息”，请在相关句子末尾使用 [来源: 页码 X] 的格式进行引用。
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 4. 创建可调用工具的Agent
        # create_tool_calling_agent 自动处理工具调用逻辑
        agent = create_tool_calling_agent(llm, tools, prompt)

        # 5. 创建Agent执行器
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # verbose=True 可以看到Agent的思考过程

        # 6. 使用RunnableWithMessageHistory包装AgentExecutor以支持多轮对话
        chain_with_history = RunnableWithMessageHistory(
            agent_executor,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",  # 确保与prompt中的variable_name匹配
        )
        return chain_with_history

    def rag_chat(self, question: str, session_id: str, n_results: int = 3) -> Dict[str, Any]:
        """
        完整的RAG聊天流程，集成了重排机制以提高上下文精度。
        """
        # --- 重排流程开始 ---
        initial_retrieval_count = 10
        print(f"向量检索中，获取 {initial_retrieval_count} 个候选文档...")
        retrieved_results = query_db(question, n_results=initial_retrieval_count)

        initial_docs = retrieved_results["documents"][0]
        initial_metadatas = retrieved_results.get("metadatas", [[]])[0]

        # 如果没有检索到任何文档，则直接跳过重排。
        if not initial_docs:
            print("未能从知识库检索到任何相关文档。")
            final_context_docs = []
            formatted_context = "无"
        else:
            # 重排器需要成对的 `[query, document]` 列表。
            print("准备重排数据...")
            rerank_pairs = []
            for doc in initial_docs:
                rerank_pairs.append([question, doc])

            # 使用重排模型计算相关性得分。
            print("计算相关性得分...")
            scores = self.reranker.predict(rerank_pairs)

            # 将文档、元数据和分数合并，并按分数降序排序。
            # 我们将元数据与文档绑定，以防排序后信息错乱。
            print("按相关性得分排序...")
            docs_with_scores_and_metadata = list(zip(initial_docs, initial_metadatas, scores))
            docs_with_scores_and_metadata.sort(key=lambda x: x[2], reverse=True)

            # 筛选出重排后得分最高的n_results个文档作为最终上下文。
            print(f"筛选出得分最高的 {n_results} 个文档。")
            final_docs_with_metadata = docs_with_scores_and_metadata[:n_results]

            # 从排序后的结果中分离出最终的文档和元数据
            final_context_docs = [item[0] for item in final_docs_with_metadata]
            final_metadatas = [item[1] for item in final_docs_with_metadata]

            # 准备最终给大模型的、带有来源信息的上下文文本。
            formatted_context_list = []
            for i, doc in enumerate(final_context_docs):
                metadata = final_metadatas[i] or {}  # 确保metadata不是None
                source = metadata.get('source', '未知文档')
                page = metadata.get('page', 'N/A')
                formatted_context_list.append(
                    f"内容片段 {i + 1} (来源: {os.path.basename(source)}, 页码 {page}):\n{doc}")

            formatted_context = "\n\n".join(formatted_context_list)
        # --- 重排流程结束 ---
        # 将重排并筛选后的、最相关的上下文信息加入到Agent的输入中。
        inputs = {
            "question": question,
            "context": formatted_context
        }

        # 调用支持多轮对话和工具的Agent
        response = self.chain.invoke(inputs, config={"configurable": {"session_id": session_id}})

        # 构建更丰富的来源信息
        sources = ["来源: 本地知识库"] if final_context_docs else []
        unique_sources = list(set(sources))
        return {
            "question": question,
            "answer": response.get("output", "无法获取回答。"),
            "context": final_context_docs,  # 返回经过重排的上下文，便于调试和展示
            "sources": unique_sources
        }


def main():
    agent = ChatAgent(model_name="qwen3:4B")
    session_id = "test_session"  # 固定的会话ID用于命令行测试

    print("=" * 50)
    print("RAG 智能体已启动 (输入 'exit' 退出)")
    print("=" * 50)

    while True:
        user_input = input("\n你的问题: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("对话结束。")
            break

        if not user_input:
            continue

        # 调用rag_chat并传入session_id
        response = agent.rag_chat(user_input, session_id=session_id)

        print("\n" + "=" * 50)
        print(f"问题: {response['question']}")
        print("-" * 50)
        print(f"回答: {response['answer']}")
        print("-" * 50)
        print("参考上下文:")
        if response['context']:
            for i, doc in enumerate(response["context"]):
                print(f"[文档 {i + 1}]: {doc[:200]}...")
        else:
            print("无相关上下文信息。")
        print("-" * 50)
        print("来源信息:")
        if response['sources']:
            for source in response['sources']:
                print(f"- {source}")
        else:
            print("无明确来源信息。")
        print("=" * 50)


if __name__ == "__main__":
    main()