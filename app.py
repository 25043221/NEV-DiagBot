import streamlit as st
import os # 导入os
import uuid # 导入uuid用于生成session_id

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
st.markdown("---") # 分隔线

# --- 后端初始化 ---

@st.cache_resource
def get_chat_agent():
    """加载并缓存 ChatAgent 实例"""
    print("正在初始化 ChatAgent...")
    # 可以根据需要调整模型名称，确保Ollama中已下载
    return ChatAgent(model_name="qwen3:4B") # 保持与chat.py中的默认模型一致


@st.cache_resource
def ensure_db_created():
    """检查并创建数据库"""
    # 请根据您的实际PDF路径进行修改
    pdf_file_path = "./data/秦plusDMi用户手册.pdf" # 假设PDF文件在data目录下
    if not os.path.exists(pdf_file_path):
        st.error(f"错误：未找到用户手册PDF文件在路径 `{pdf_file_path}`. 请确保文件存在。")
        return False
    print("正在检查并创建数据库...")
    create_db(pdf_file_path)
    return True


# 加载 Agent 并创建数据库
agent = get_chat_agent()
db_created = ensure_db_created()

# --- 会话状态管理 ---

# 为每个用户会话生成一个唯一的 session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # 使用UUID生成唯一ID

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属新能源汽车助手。请问关于您的爱车，有什么可以帮您的吗？比如，您有什么故障码需要查询，或者想了解某个功能？"}]


# --- 5. 侧边栏与示例问题 (提升用户体验) ---
with st.sidebar:
    st.header("💡 使用提示")
    st.info("您可以直接在下方的聊天框中提问，也可以点击下面的示例问题，快速开始体验。")

    st.subheader("❓ 常用问题示例")
    example_questions = [
        "我的车最近续航掉了20%，可能是什么原因？",
        "如何为我的车辆进行首次保养？",
        "仪表盘上出现一个黄色的电池图标是什么意思？",
        "空调制冷效果不佳怎么办？",
        "P0420故障码是什么意思？",
        "冬季如何维护电池？",
        "最近电动汽车自燃事件多发，我的车安全吗？"
    ]

    selected_question = None
    for q in example_questions:
        if st.button(q, key=q):
            selected_question = q

    st.subheader("🛠️ 故障码快速查询")
    st.text_input("在这里输入故障码 (如 P0420)", key="fault_code_input",
                   placeholder="例如：P0420")
    if st.session_state.fault_code_input:
        if st.button("查询故障码"):
            selected_question = st.session_state.fault_code_input + "是什么意思？"
            st.session_state.fault_code_input = "" # 清空输入框


    st.markdown("---")
    st.header("🧠 参考上下文")
    st.session_state.source_container = st.container()

    st.markdown("---")
    st.header("⚙️ 更多功能 (规划中)")
    st.info("未来版本将支持：")
    st.markdown("""
        * 实时车辆数据诊断：连接OBD接口，实时监测车辆健康状况。
        * 语音交互：更自然的对话体验。
        * 图像识别：拍照识别仪表盘警告灯，直观获取信息。
        * 专业技师预约/社区互助：一键联系专业服务或获取车友帮助。
        * 视频教程库：提供常见保养和简单维修的视频指导。
    """)


# --- 6. 聊天界面渲染 (保持不变) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. 用户输入处理与 RAG 流程 (核心重构) ---

if prompt := selected_question or st.chat_input("请在这里描述您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在知识库中检索并思考..."):
            try:
                # 调用 RAG 核心流程，并传入 session_id
                response_data = agent.rag_chat(prompt, session_id=st.session_state.session_id, n_results=5)
                context_docs = response_data["context"]
                sources = response_data["sources"]

                with st.session_state.source_container:
                    if not context_docs:
                        st.warning("未能从知识库中找到直接相关的信息。模型的回答将基于其通用知识或网络搜索。")
                    else:
                        st.info("以下是本次回答参考的主要知识片段：")
                        for i, doc in enumerate(context_docs):
                            with st.expander(f"参考文档 {i + 1}"):
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                        if sources:
                            st.markdown("---")
                            st.info("来源信息:")
                            for source in sources:
                                st.markdown(f"- {source}")
                        else:
                            st.markdown("---")
                            st.warning("无明确来源信息。")

                full_response = response_data["answer"] # 直接获取Agent的最终回答

            except Exception as e:
                st.error(f"处理问题时出错: {e}")
                full_response = "抱歉，我在处理您的请求时遇到了问题。请稍后再试。"

        st.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})