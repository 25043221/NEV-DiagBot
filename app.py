import io
import time
from typing import Iterator

import streamlit as st
import os
import uuid

from PIL import Image

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
st.caption("由本地大模型驱动，为您解答关于新能源汽车的使用、保养及故障诊断问题")
st.markdown("---")

# --- 后端初始化 ---
@st.cache_resource
def get_chat_agent():
    """加载并缓存 ChatAgent 实例"""
    print("正在初始化 ChatAgent...")
    return ChatAgent(model_name="qwen3:4B")


@st.cache_resource
def ensure_db_created():
    """检查并创建数据库"""
    pdf_file_path = "./data/秦plusDMi用户手册.pdf"
    data_dir = os.path.dirname(pdf_file_path)

    if not os.path.exists(data_dir):
        st.error(f"错误：数据目录 `{data_dir}` 不存在。请创建该目录并将用户手册PDF文件放入其中。")
        return False

    if not os.path.exists(pdf_file_path):
        st.error(f"错误：未找到用户手册PDF文件在路径 `{pdf_file_path}`. 请确保文件存在。")
        return False

    print("正在检查并创建数据库...")
    try:
        create_db()
        return True
    except Exception as e:
        st.error(f"创建或连接数据库时发生错误: {e}")
        return False


agent = get_chat_agent()
db_created = ensure_db_created()

if not db_created:
    st.stop()

# --- 会话状态管理 ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "您好！我是您的专属新能源汽车助手。请问关于您的爱车，有什么可以帮您的吗？比如，您有什么故障码需要查询，或者想了解某个功能？"}]

# --- 用于存储被选中的示例问题 ---
if "selected_example_question" not in st.session_state:
    st.session_state.selected_example_question = None

if "current_fault_code_input" not in st.session_state:
    st.session_state.current_fault_code_input = ""

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def query_fault_code_callback():
    fault_code_to_query = st.session_state.fault_code_input_widget_key
    if fault_code_to_query:
        st.session_state.selected_example_question = fault_code_to_query + "是什么意思？"
        st.session_state.fault_code_input_widget_key = ""
    # st.rerun() # Rerun here after state is updated

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

    # 修改示例问题按钮的逻辑
    for i, q in enumerate(example_questions):
        # 使用 f-string 为每个按钮生成唯一的key，避免key冲突
        if st.button(q, key=f"example_q_{i}"):
            st.session_state.selected_example_question = q
            # 立即重新运行应用，确保示例问题被处理
            st.rerun()

    st.subheader("🛠️ 故障码快速查询")
    st.text_input(
        "在这里输入故障码 (如 P0420)",
        key="fault_code_input_widget_key",
        placeholder="例如：P0420",
        value=st.session_state.current_fault_code_input  # This value is now dynamically controlled
    )
    st.button("查询故障码", key="query_fault_code_button", on_click=query_fault_code_callback)

    st.subheader("📸 上传图片进行诊断")
    uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file.read() # Read image as bytes
        image = Image.open(io.BytesIO(st.session_state.uploaded_image)) # Open image from bytes
        st.image(image, caption='已上传的图片', use_container_width=True)
        st.success("图片上传成功！")

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


# --- 6. 聊天界面渲染 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and "text" in message["content"] and "image" in message["content"]:
            st.markdown(message["content"]["text"])
            st.image(Image.open(io.BytesIO(message["content"]["image"])), caption="用户上传图片", use_container_width=True)
        else:
            st.markdown(message["content"])


# --- 7. 用户输入处理与 RAG 流程 (核心重构) ---
# 始终显示聊天输入框
user_input_prompt = st.chat_input("请在这里描述您的问题...")

# 确定最终要处理的 prompt
prompt = None
image_to_process = None
# 优先处理通过示例问题或故障码设置的 session_state 变量
if st.session_state.selected_example_question:
    prompt = st.session_state.selected_example_question
    st.session_state.selected_example_question = None # Clear after use
elif user_input_prompt: # Otherwise, check for direct user input
    prompt = user_input_prompt

if st.session_state.uploaded_image:
    image_to_process = st.session_state.uploaded_image
    # 如果用户没有设置提示次，直接默认
    if not prompt:
        prompt = "请分析这张图片。"

if prompt:
    user_message_content = {"text": prompt}
    if image_to_process:
        user_message_content["image"] = image_to_process
    st.session_state.messages.append({"role": "user", "content": user_message_content})
    with st.chat_message("user"):
        st.markdown(prompt)
        if image_to_process:
            st.image(Image.open(io.BytesIO(image_to_process)), caption="用户上传图片", use_container_width=True)

    with st.chat_message("assistant"):
        with st.spinner("正在知识库中检索并思考..."):
            try:
                response_data = agent.rag_chat(prompt, session_id=st.session_state.session_id, n_results=5, image_bytes= image_to_process)
                context_docs = response_data.get("context", [])
                sources = response_data.get("sources", [])

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

                full_response = response_data.get("answer", "无法获取回答。")

            except Exception as e:
                st.error(f"处理问题时出错: {e}")
                full_response = "抱歉，我在处理您的请求时遇到了问题。请稍后再试。"
            finally:
                st.session_state.uploaded_image = None

        st.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})