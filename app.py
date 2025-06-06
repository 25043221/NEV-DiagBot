import streamlit as st

from chat import ChatAgent
from embed import create_db, query_db

# --- 页面基础设置 ---
st.set_page_config(
    page_title="个人理财助手",
    page_icon="🤖",
    layout="wide"  # 使用宽布局，更像 Gemini
)

# --- 应用标题和介绍 ---
st.title("🤖 个人理财助手")
st.caption("一个由本地大模型驱动，能够回答您个人理财问题的智能助手。")


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
@st.cache_data
def ensure_db_created():
    """检查并创建数据库"""
    print("正在检查并创建数据库...")
    create_db()  # 假设 create_db 内部有检查是否需要重复创建的逻辑
    return True


# 加载 Agent 并创建数据库
agent = get_chat_agent()
db_created = ensure_db_created()

# --- 会话状态管理 ---

# 初始化聊天历史
# st.session_state 是 Streamlit 用于在用户多次交互之间保持变量的机制
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属理财助手，请问有什么可以帮您的吗？"}]

# --- 聊天界面渲染 ---

# 显示历史消息
for message in st.session_state.messages:
    # 使用 with st.chat_message() 为不同角色的消息创建专属容器
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 用户输入处理与 RAG 流程 ---

# st.chat_input 会在页面底部创建一个固定的输入框
if prompt := st.chat_input("请在这里输入您的问题..."):
    # 1. 将用户输入添加到聊天历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG 核心流程
    with st.spinner("正在思考中，请稍候..."):
        # (1) 从向量数据库检索相关信息
        # 我们将检索到的上下文显示在侧边栏，保持主聊天区清爽
        retrieved_results = query_db(prompt, n_results=3)  # 从数据库检索3个最相关的块
        context_docs = retrieved_results["documents"][0]
        context_metadatas = retrieved_results["metadatas"][0]

        # 在侧边栏显示来源信息
        with st.sidebar:
            st.subheader("🧠 参考上下文")
            for i, (doc, meta) in enumerate(zip(context_docs, context_metadatas)):
                with st.expander(f"来源 {i + 1}: {meta.get('source', 'N/A')} - 块 {meta.get('chunk_index', 'N/A')}"):
                    st.text(doc)

        context_str = "\n\n".join(context_docs)

    # 3. 生成并流式输出回答
    # 创建一个助手消息的容器
    with st.chat_message("assistant"):
        # st.write_stream 是实现打字机效果的关键
        # 它会接收一个生成器（generator），并将其产生的内容实时渲染到页面上
        # 我们的 agent.chain.stream() 正好返回一个生成器
        response_stream = agent.chain.stream({
            "context": context_str,
            "question": prompt
        })

        # 将流式响应写入页面，并保存完整响应
        full_response = st.write_stream(response_stream)

    # 4. 将完整的助手回答添加到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})