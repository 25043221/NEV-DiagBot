import time
from typing import Iterator

import streamlit as st
import os
import uuid

from chat import ChatAgent
# 导入 create_db 和 query_db。query_db 在主聊天流程中没有直接使用，但可能用于其他调试或功能。
from embed import create_db, query_db

# --- 1. 页面基础设置 (领域适配) ---
# 设置 Streamlit 应用的页面配置，包括标题、图标和布局
st.set_page_config(
    page_title="新能源汽车智能助手",
    page_icon="🚗",
    layout="wide"
)

# --- 2. 应用标题和介绍 (领域适配) ---
# 设置应用的主标题
st.title("🚗 新能源汽车智能诊断与知识助手")
# 添加应用的副标题/说明
st.caption("由本地大模型驱动，为您解答关于新能源汽车的使用、保养及故障诊断问题。")
st.markdown("---") # 添加一条水平分隔线，用于视觉上的分隔

# --- 后端初始化 ---

# 使用 @st.cache_resource 装饰器缓存资源，避免在每次应用重新运行时重复初始化
@st.cache_resource
def get_chat_agent():
    """加载并缓存 ChatAgent 实例"""
    print("正在初始化 ChatAgent...")
    # 初始化 ChatAgent 实例，指定Ollama模型名称。
    # 确保在Ollama中已下载并运行此模型。
    return ChatAgent(model_name="qwen3:4B")


# 使用 @st.cache_resource 装饰器缓存资源，避免在每次应用重新运行时重复执行
@st.cache_resource
def ensure_db_created():
    """检查并创建数据库"""
    # 注意：embed.py 中的 create_db 函数目前不接受文件路径参数。
    # 它依赖于 chunk.py 中的 get_pdf_text 函数来查找 PDF 文件。
    # 在这里检查 PDF 文件的存在性是作为先决条件。
    pdf_file_path = "./data/秦plusDMi用户手册.pdf"
    # 从 PDF 文件路径中提取目录路径
    data_dir = os.path.dirname(pdf_file_path)

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        st.error(f"错误：数据目录 `{data_dir}` 不存在。请创建该目录并将用户手册PDF文件放入其中。")
        return False

    # 检查特定的 PDF 文件是否存在
    if not os.path.exists(pdf_file_path):
        st.error(f"错误：未找到用户手册PDF文件在路径 `{pdf_file_path}`. 请确保文件存在。")
        return False

    print("正在检查并创建数据库...")
    try:
        # 尝试创建 ChromaDB 数据库。
        # 请注意：embed.py 中的 create_db() 内部调用 chunk.py 中的 get_text_chunks()，
        # 而 get_text_chunks() 硬编码了 PDF 文件的查找路径（例如 '/home/yu/zjh/my_app/data'）。
        # 请确保此路径或其符号链接指向您实际的数据目录。
        create_db()
        return True
    except Exception as e:
        # 捕获创建或连接数据库时可能发生的任何异常
        st.error(f"创建或连接数据库时发生错误: {e}")
        return False # 指示失败


# 加载 ChatAgent 实例并确保数据库已设置
agent = get_chat_agent()
db_created = ensure_db_created()

# 如果数据库创建失败，停止 Streamlit 应用程序以防止进一步的错误
if not db_created:
    st.stop() # 这将阻止应用程序继续执行，如果数据库未准备好。

# --- 会话状态管理 ---

# 检查 Streamlit 会话状态中是否存在唯一的 session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # 使用 UUID 生成唯一的会话 ID

# 检查会话状态中是否存在聊天消息历史记录
if "messages" not in st.session_state:
    # 使用助手的开场白初始化消息列表
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的专属新能源汽车助手。请问关于您的爱车，有什么可以帮您的吗？比如，您有什么故障码需要查询，或者想了解某个功能？"}]


# --- 5. 侧边栏与示例问题 (提升用户体验) ---
with st.sidebar: # 此代码块中的内容将显示在 Streamlit 的侧边栏中
    st.header("💡 使用提示")
    st.info("您可以直接在下方的聊天框中提问，也可以点击下面的示例问题，快速开始体验。")

    st.subheader("❓ 常用问题示例")
    # 定义一个示例问题列表，方便用户快速交互
    example_questions = [
        "我的车最近续航掉了20%，可能是什么原因？",
        "如何为我的车辆进行首次保养？",
        "仪表盘上出现一个黄色的电池图标是什么意思？",
        "空调制冷效果不佳怎么办？",
        "P0420故障码是什么意思？",
        "冬季如何维护电池？",
        "最近电动汽车自燃事件多发，我的车安全吗？"
    ]

    selected_question = None # 初始化一个变量，用于存储从按钮中选择的问题
    for q in example_questions:
        if st.button(q, key=q): # 为每个示例问题创建一个按钮
            selected_question = q # 如果按钮被点击，则设置 selected_question

    st.subheader("🛠️ 故障码快速查询")
    # 用于用户输入故障码的文本输入框
    st.text_input("在这里输入故障码 (如 P0420)", key="fault_code_input",
                   placeholder="例如：P0420")
    # 如果故障码输入框中有内容
    if st.session_state.fault_code_input:
        if st.button("查询故障码"): # 触发故障码查询的按钮
            # 将输入格式化为聊天机器人可识别的问题
            selected_question = st.session_state.fault_code_input + "是什么意思？"
            st.session_state.fault_code_input = "" # 选择后清空输入字段，避免重复触发


    st.markdown("---")
    st.header("🧠 参考上下文")
    # 一个容器，用于动态显示检索到的上下文和来源信息
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
# 遍历会话状态中存储的消息并显示它们
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # 使用 Streamlit 的 chat_message 组件
        st.markdown(message["content"]) # 将消息内容作为 Markdown 格式显示



# --- 7. 用户输入处理与 RAG 流程 (核心重构) ---

# 检查是否有通过聊天输入框或示例问题/故障码提供的提示
if prompt := selected_question or st.chat_input("请在这里描述您的问题..."):
    # 将用户的提问添加到会话状态中的聊天历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天界面中显示用户的提问
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备助手的回复
    with st.chat_message("assistant"):
        with st.spinner("正在知识库中检索并思考..."): # 显示一个旋转动画，表示正在处理
            try:
                # 调用 RAG 核心流程（ChatAgent 的 rag_chat 方法）
                # 传递用户的提问、当前的 session_id 和所需的结果数量
                response_data = agent.rag_chat(prompt, session_id=st.session_state.session_id, n_results=5)
                # 从响应数据中提取上下文文档和来源信息
                context_docs = response_data.get("context", [])
                sources = response_data.get("sources", [])

                # 在侧边栏的专用容器中显示上下文和来源信息
                with st.session_state.source_container:
                    if not context_docs:
                        st.warning("未能从知识库中找到直接相关的信息。模型的回答将基于其通用知识或网络搜索。")
                    else:
                        st.info("以下是本次回答参考的主要知识片段：")
                        # 遍历并显示每个检索到的文档片段
                        for i, doc in enumerate(context_docs):
                            with st.expander(f"参考文档 {i + 1}"): # 使用可折叠的扩展器，方便查看
                                # 显示文档的前 500 个字符，如果文档较短则显示全部
                                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                        if sources:
                            st.markdown("---")
                            st.info("来源信息:")
                            for source in sources:
                                st.markdown(f"- {source}")
                        else:
                            # 理论上，如果找到了上下文文档，这里不应该出现“无明确来源信息”的情况
                            st.markdown("---")
                            st.warning("无明确来源信息。")

                # 从 Agent 的响应数据中获取最终的回答
                full_response = response_data.get("answer", "无法获取回答。")

            except Exception as e:
                # 捕获处理过程中可能发生的任何异常，并显示错误消息
                st.error(f"处理问题时出错: {e}")
                full_response = "抱歉，我在处理您的请求时遇到了问题。请稍后再试。"

        st.write(full_response) # 显示助手的最终回答
    # 将助手的回答添加到会话状态中的聊天历史记录
    st.session_state.messages.append({"role": "assistant", "content": full_response})