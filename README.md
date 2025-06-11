# NEV-DiagBot: 新能源汽车智能诊断与知识助手 🚗💡

## 项目简介

**NEV-DiagBot** 是一个基于本地大模型（LLM）和检索增强生成（RAG）技术的智能问答系统，专注于新能源汽车领域的故障诊断、使用指南和知识咨询。
![image](https://github.com/user-attachments/assets/4a03a155-37f6-46ef-908c-351595c5e389)
![image](https://github.com/user-attachments/assets/bf20cc52-fae0-4503-b65c-c7573233babb)

## 我们的目标

通过整合本地知识库和外部网络搜索能力，NEV-DiagBot 致力于为新能源车主提供专业、准确、易懂且富有同理心的解决方案与建议。

本项目旨在解决通用大模型在专业领域知识不足、容易出现“幻觉”的问题。通过多源信息融合，提升回答的可靠性和深度，成为您身边的专属新能源汽车专家。

---

## 核心功能

### 🛠️ 智能诊断与故障码解析

- 详尽解释各种新能源汽车故障码的含义。
- 提供可能的故障原因、初步排查建议及后续行动指南。
- 针对故障提供潜在维修建议及大致费用估算（基于现有信息）。
- **新增：** 支持发送图片，实现多模态输入，如仪表盘警告灯图像识别诊断。

### 🧠🌐 智能路由与多源知识融合

- **智能意图识别**：系统识别用户问题的意图，区分新能源汽车相关问题与通用性问题。
- **本地大模型**（车辆问题）：使用本地部署的模型（如 Ollama qwen3:4B）结合 RAG 从私有文档知识库中检索并生成回答，确保专业性和隐私。
- **在线大模型**（通用问题）：如为通用问题，智能路由至 GPT-4o、Gemini 等在线模型进行回答。
- **RAG 本地知识库**：从用户手册、维修文档中检索相关信息。
- **网络搜索**：集成 Tavily 搜索引擎，获取最新或外部信息，补充本地知识库。

### 💬 多轮对话支持

- 记忆对话历史，提供连贯自然的交流体验。

### 🖥️ 友好的用户界面

- 基于 Streamlit 构建交互式 Web 界面，清晰展示回答及信息来源。

---

## 技术栈

| 技术模块     | 工具/框架 |
|--------------|------------|
| 核心语言     | Python 3.10+ |
| 大模型框架   | LangChain |
| 大模型路由   | One API（统一管理本地与在线模型） |
| 路由日志    | sqlLite |
| 本地大模型   | Ollama（qwen3:4B、qwen2.5vl:3b、qwen:0.5b 等） |
| 在线大模型   | deepseek-chat |
| 嵌入模型     | Ollama（nomic-embed-text:latest） |
| 向量数据库   | ChromaDB |
| 网络搜索     | Tavily API |
| 文档处理     | PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader |
| 文本分割     | RecursiveCharacterTextSplitter |
| 重排序模型   | Sentence-transformers（如 BAAI/bge-reranker-base） |
| 前端界面     | Streamlit |

---

## 快速开始

1. 克隆项目仓库

    ```bash
    git clone https://github.com/25043221/NEV-DiagBot.git
    cd NEV-DiagBot
    ```

2. 安装 Python 依赖

    ```bash
    pip install -r requirements.txt
    ```

3. 安装并运行 Ollama

    - 访问 [Ollama 官网](https://ollama.com) 下载并安装。
    - 下载所需模型：

    ```bash
    ollama run qwen3:4B
    ollama run nomic-embed-text:latest
    ```

    - 确保服务运行在 `http://localhost:11434`。
4. 部署和配置 One API

- 部署 One API 服务： 参照 One API GitHub 仓库 或官方文档，将其部署在你的服务器或本地。推荐使用 Docker 部署。
- 配置 One API 渠道：
- 登录 One API 管理界面。
- 添加一个新渠道，类型选择 Ollama，基础地址填写 http://localhost:11434，并为其设置一个模型名称（例如：ollama-qwen3-4b）。
- 添加另一个新渠道，类型选择你希望使用的在线大模型，填入相应的 API Key，并为其设置一个模型名称。
- 获取 One API Key： 在 One API 后台生成一个用于访问 One API 服务的令牌 (Key)。
- 设置 API Keys

```bash
python -c "from load_key import load_key; load_key('ONE_API_BASE_URL'); load_key('ONE_API_KEY'); load_key('TAVILY_API_KEY')"
# 如果你的在线大模型在 One API 里需要额外的API Key，确保其也在 Keys.json 中配置，例如：
# python -c "from load_key import load_key; load_key('OPENAI_API_KEY')"
```

5. 准备本地知识库数据

    - 在项目根目录下创建 `data` 文件夹。
    - 放入您的新能源汽车用户手册、维修文档等 `.pdf`、`.md`、`.txt` 格式的文件，例如：

    ```bash
    ./data/秦plusDMi用户手册.pdf
    ```
6. 设置 Tavily API Key

    - 注册并获取您的 Tavily API Key：[Tavily 官网](https://www.tavily.com)
    - 运行以下命令并输入您的 API Key：

    ```bash
    python -c "from load_key import load_key; load_key('TAVILY_API_KEY')"
    ```

7. 创建 ChromaDB 向量数据库

    ```bash
    python -c "from embed import create_db; create_db()"
    ```

    这将读取 `data/` 目录文档，分块、嵌入并构建向量数据库（存储于 `./chroma_db/`）。

8. 启动应用

    ```bash
    streamlit run app.py
    ```

    浏览器中将自动打开页面，通常是 `http://localhost:8501`。

## 故障排除

- **Ollama 未启动**：确保本地服务运行，且模型已下载。
- **Tavily Key 错误**：确认 `keys.json` 中 `TAVILY_API_KEY` 是否正确。
- **知识库未识别**：确保 `data/` 目录存在且包含文档，`chunk.py` 会自动查找该目录。

## 未来规划

✅ 更深度的意图识别（支持车型/年份理解）
✅ ~~支持多模态输入（图片识别仪表盘警告灯）~~
✅ ~~集成 One API 实现大模型智能路由~~
✅ ~~用户意图识别（区分车辆问题与通用问题）~~
✅ 容器化部署（Docker支持）
✅实时诊断（对接 OBD 接口）
✅丰富的知识图谱源（对接行业结构化知识库）
✅模型微调（基于专业数据提升问答能力）
