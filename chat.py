import base64
import os
from typing import List, Any, Dict, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from embed import query_db
from load_key import load_key

from sentence_transformers import CrossEncoder

from multimodal_model import MultimodalModel

ONE_API_BASE_URL = load_key("ONE_API_BASE_URL") # 例如: http://localhost:3000/v1
ONE_API_KEY = load_key("ONE_API_KEY") # One API 的访问令牌

os.environ["TAVILY_API_KEY"] = load_key("TAVILY_API_KEY")


class ChatAgent:
    def __init__(self,
                 local_model_name_via_oneapi: str = "qwen3:4B",  # 在 One API 中为 Ollama 渠道配置的模型名称
                 intent_model_name: str = "qwen3:0.6b",  # 用于意图识别的本地 Ollama 模型
                 online_model_name_via_oneapi: str = "deepseek-chat",  # 在 One API 中为在线模型渠道配置的模型名称
                 ollama_base_url: str = "http://localhost:11434"):  # 本地 Ollama 服务的基础 URL):
        """
        初始化聊天智能体

        Args:
            model_name: 使用的Ollama模型名称
            base_url: Ollama服务的基础URL
        """
        self.store = {}
        print("初始化Re-ranking中...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("Re-ranking初始化完成.")

        #one api分法的多模态模型
        self.multimodal_model = MultimodalModel(model_name="qwen2.5vl:3b", base_url=ollama_base_url)

        # --- 意图识别模型 (直接调用本地 Ollama) ---
        self.intent_llm = ChatOllama(model=intent_model_name, base_url=ollama_base_url, temperature=0.1)

        # --- 通过 One API 调用本地大模型 (用于车辆问题) ---
        # 注意：这里使用 ChatOpenAI，但它的 base_url 指向 One API
        # model_name 对应 One API 中配置的 Ollama 渠道的模型名称
        self.local_model_for_vehicle_via_oneapi = ChatOpenAI(
            openai_api_base=ONE_API_BASE_URL,
            openai_api_key=ONE_API_KEY,
            model_name=local_model_name_via_oneapi,
            temperature=0.3,
            max_tokens=2000  # 可以根据需要调整
        )
        self.local_llm_chain = self._build_local_chain()

        # --- 通过 One API 调用在线大模型 (用于非车辆问题) ---
        # model_name 对应 One API 中在线模型渠道的模型名称
        self.online_llm_via_oneapi = ChatOpenAI(
            openai_api_base=ONE_API_BASE_URL,
            openai_api_key=ONE_API_KEY,
            model_name=online_model_name_via_oneapi,
            temperature=0.7,
            max_tokens=2000  # 可以根据需要调整
        )

        # 用于意图识别的Prompt
        self.intent_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图识别助手。根据用户的提问，判断其意图是关于“新能源汽车问题”还是“通用问题”。
                    请直接回答“车辆问题”或“通用问题”，不要添加任何其他解释或标点符号。

                    例如：
                    用户: 我的车仪表盘亮黄灯是什么意思？
                    回答: 车辆问题

                    用户: 明天天气怎么样？
                    回答: 通用问题

                    用户: 如何更换电动汽车电池？
                    回答: 车辆问题

                    用户: 什么是量子计算？
                    回答: 通用问题
                    """),
            ("human", "{question}")
        ])
        print("ChatAgent 初始化完成.")

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _determine_intent(self, question: str) -> str:
        """
        判断用户问题的意图。
        """
        print(f"正在判断用户意图：{question}")
        try:
            intent_chain = self.intent_prompt_template | self.intent_llm
            response = intent_chain.invoke({"question": question})
            intent = response.content.strip().lower()
            print(intent)
            if "车辆问题" in intent:
                print("意图识别为: 车辆问题")
                return "vehicle"
            else:
                print("意图识别为: 通用问题")
                return "general"
        except Exception as e:
            print(f"意图识别失败，默认为通用问题: {e}")
            return "general"  # 失败时默认使用通用模型


    def _build_local_chain(self) -> Any:
        """
        构建用于处理本地大模型的链（即RAG Agent），该模型通过 One API 调用。
        """
        tavily_tool = TavilySearchResults(max_results=3)
        tools = [tavily_tool]
        # 使用通过 One API 调用的本地大模型
        agent_llm = self.local_model_for_vehicle_via_oneapi

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
        agent = create_tool_calling_agent(agent_llm, tools, prompt)

        # 5. 创建Agent执行器
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # verbose=True 可以看到Agent的思考过程

        # 6. 使用RunnableWithMessageHistory包装AgentExecutor以支持多轮对话
        chain_with_history = RunnableWithMessageHistory(
            agent_executor,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        return chain_with_history

    def rag_chat(self, question: str, session_id: str, n_results: int = 3, image_bytes: bytes = None) -> Dict[str, Any]:
        """
        完整的RAG聊天流程，集成了重排机制以提高上下文精度。
        支持多模态的输入，并根据意图分发到不同的大模型。
        """
        original_question = question
        image_description = None
        if image_bytes:
            # 多模态处理，直接调用本地 Ollama
            image_description = self.multimodal_model.describe_image(image_bytes)
            # 根据原始问题是否为空，拼接问题
            if not question.strip():  # 如果用户只上传图片没有文字问题
                question = f"用户上传了一张图片，描述为：'{image_description}'。"
            else:  # 如果用户上传了图片也有文字问题
                question = f"用户上传了一张图片，描述为：'{image_description}'。\n用户的问题是：{question}"
            print(f"结合图片描述后的问题: {question}")

        # --- 意图识别 ---
        # 使用结合图片描述后的问题来判断意图
        intent = self._determine_intent(question)
        final_context_docs = []
        sources = []

        if intent == "vehicle":
            print("意图为车辆问题，使用本地大模型(通过One API)进行RAG...")
            # --- RAG 流程开始 ---
            initial_retrieval_count = 10
            print(f"向量检索中，获取 {initial_retrieval_count} 个候选文档...")
            retrieved_results = query_db(question, n_results=initial_retrieval_count)

            initial_docs = retrieved_results["documents"][0]
            initial_metadatas = retrieved_results.get("metadatas", [[]])[0]

            if not initial_docs:
                print("未能从知识库检索到任何相关文档。")
                formatted_context = "无"
            else:
                print("准备重排数据...")
                rerank_pairs = []
                for doc in initial_docs:
                    rerank_pairs.append([question, doc])

                print("计算相关性得分...")
                scores = self.reranker.predict(rerank_pairs)

                print("按相关性得分排序...")
                docs_with_scores_and_metadata = list(zip(initial_docs, initial_metadatas, scores))
                docs_with_scores_and_metadata.sort(key=lambda x: x[2], reverse=True)

                print(f"筛选出得分最高的 {n_results} 个文档。")
                final_docs_with_metadata = docs_with_scores_and_metadata[:n_results]
                final_context_docs = [item[0] for item in final_docs_with_metadata]
                final_metadatas = [item[1] for item in final_docs_with_metadata]

                formatted_context_list = []
                for i, doc in enumerate(final_context_docs):
                    metadata = final_metadatas[i] or {}
                    source = metadata.get('source', '未知文档')
                    page = metadata.get('page', 'N/A')
                    formatted_context_list.append(
                        f"内容片段 {i + 1} (来源: {os.path.basename(source)}, 页码 {page}):\n{doc}")
                formatted_context = "\n\n".join(formatted_context_list)
            # --- RAG 流程结束 ---

            human_message_content = [{"type": "text", "text": question}]
            new_human_message = HumanMessage(content=human_message_content)

            inputs = {
                "question": new_human_message,
                "context": formatted_context
            }
            # 调用本地大模型链 (通过One API)
            response = self.local_llm_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})
            full_response = response.get("output", "无法获取回答。")
            sources = ["来源: 本地知识库"] if final_context_docs else []

        else:  # intent == "general"
            print("意图为通用问题，使用在线大模型(通过One API)进行回答...")
            try:
                # 直接调用在线大模型 (通过One API)
                response_online = self.online_llm_via_oneapi.invoke(question)
                full_response = response_online.content
                sources = ["来源: 在线知识"]
            except Exception as e:
                print(f"调用在线大模型失败: {e}")
                full_response = "抱歉，在线服务暂时无法响应您的通用问题。"

        return {
            "question": original_question,  # 返回原始问题，未结合图片描述
            "answer": full_response,
            "context": final_context_docs,
            "sources": list(set(sources))  # 确保来源唯一
        }


def main():
    # 注意这里 ChatAgent 的初始化参数变化，现在包含 One API 相关的模型名称
    agent = ChatAgent(
        local_model_name_via_oneapi="qwen3:4B", # 确保与 One API 配置的 Ollama 渠道模型名称一致
        intent_model_name="qwen3:0.6b",             # 意图识别的本地 Ollama 模型
        online_model_name_via_oneapi="deepseek-chat",    # 确保与 One API 配置的在线模型渠道模型名称一致
        ollama_base_url="http://localhost:11434"   # 本地 Ollama 服务的基础 URL
    )
    session_id = "test_session"  # 固定的会话ID用于命令行测试

    print("=" * 50)
    print("RAG 智能体已启动 (通过 One API 路由，输入 'exit' 退出)")
    print("=" * 50)

    while True:
        user_input = input("\n你的问题: ").strip()

        if user_input.lower() in ["exit", "quit"]:#
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
                print(f"[文档 {i + 1}]: {doc[:200]}...") # 打印前200字
        else:
            print("无。")
        print("-" * 50)
        print(f"来源: {', '.join(response['sources']) if response['sources'] else '无'}")
        print("=" * 50)


if __name__ == "__main__":
    main()