import streamlit as st

from chat import ChatAgent
from embed import create_db, query_db

# --- 1. 页面基础设置 (领域适配) ---
st.set_page_config(
    page_title="新能源汽车智能助手",
    page_icon="🚗",
    layout="wide"
)

# --- 2. 应用标题和介绍 (领域适配) ---
st.title("🚗 新能源汽车智能诊断与知识助手")
st.caption("由本地大模型驱动，为您解答关于新能源汽车的使用、保养及故障诊断问题。")

# --- 后端初始化 ---

# 使用 Streamlit 的缓存机制，避免每次交互都重新加载模型
# 这会极大地提高应用的响应速度
@st.cache_resource
def get_chat_agent():
    """加载并缓存 ChatAgent 实例"""
    print("正在初始化 ChatAgent...")
    return ChatAgent()


# 在应用启动时，确保数据库已经创建
# 这个函数应该是幂等的，即多次运行不产生副作用
@st.cache_resource
def ensure_db_created():
    """检查并创建数据库"""
    print("正在检查并创建数据库...")
    create_db()
    return True


# 加载 Agent 并创建数据库
agent = get_chat_agent()
db_created = ensure_db_created()

# --- 会话状态管理 ---

# 初始化聊天历史
# st.session_state 是 Streamlit 用于在用户多次交互之间保持变量的机制
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属汽车助手。请问关于您的爱车，有什么可以帮您的吗？"}]


# --- 5. 侧边栏与示例问题 (提升用户体验) ---
with st.sidebar:
    st.header("💡 使用提示")
    st.info("您可以直接在下方的聊天框中提问，也可以点击下面的示例问题，快速开始体验。")

    example_questions = [
        "我的车最近续航掉了20%，可能是什么原因？",
        "如何为我的车辆进行首次保养？",
        "仪表盘上出现一个黄色的电池图标是什么意思？",
        "空调制冷效果不佳怎么办？"
    ]

    # 使用 st.button 来创建示例问题按钮
    selected_question = None
    for q in example_questions:
        if st.button(q, key=q):
            selected_question = q

    st.header("🧠 参考上下文")
    # 创建一个容器，用于后续显示检索到的上下文信息
    st.session_state.source_container = st.container()

# --- 6. 聊天界面渲染 (保持不变) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. 用户输入处理与 RAG 流程 (核心重构) ---

# 检查是否有示例问题被点击，或者用户是否有新的输入
if prompt := selected_question or st.chat_input("请在这里描述您的问题..."):
    # 1. 将用户输入添加到聊天历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG 核心流程，现在封装在 agent 中，UI 更简洁
    with st.chat_message("assistant"):
        # 使用 spinner 提升等待体验
        with st.spinner("正在知识库中检索并思考..."):
            # (1) 从向量数据库检索相关信息
            try:
                retrieved_results = query_db(prompt, n_results=3)
                context_docs = retrieved_results["documents"][0]
                context_metadatas = retrieved_results["metadatas"][0]

                # (2) 在侧边栏显示来源信息 (提升透明度)
                with st.session_state.source_container:
                    if not context_docs:
                        st.warning("未能从知识库中找到直接相关的信息。模型的回答将基于其通用知识。")
                    else:
                        st.info("以下是本次回答参考的主要知识片段：")
                        for i, (doc, meta) in enumerate(zip(context_docs, context_metadatas)):
                            # 使用 expander 创建可折叠的来源区域
                            with st.expander(
                                    f"来源 {i + 1}: {meta.get('source', '未知文档')} (片段 {meta.get('chunk_index', 'N/A')})"):
                                st.text(doc)

                context_str = "\n\n".join(context_docs)

            except Exception as e:
                st.error(f"检索知识库时出错: {e}")
                context_str = ""  # 出错时，上下文为空

        # (3) 生成并流式输出回答
        # 使用 st.write_stream 实现打字机效果
        response_stream = agent.chain.stream({
            "context": context_str,
            "question": prompt
        })

        # 将流式响应写入页面，并捕获完整响应
        full_response = st.write_stream(response_stream)

    # 4. 将完整的助手回答添加到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

