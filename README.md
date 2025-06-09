# NEV-DiagBot: 新能源汽车智能诊断与知识助手 🚗💡

## 🌟 项目简介
**NEV-DiagBot** 是一个基于本地大模型（LLM）和检索增强生成（RAG）技术的智能问答系统，专注于新能源汽车领域的故障诊断、使用指南和知识咨询。

### 🎯 我们的目标
通过整合本地知识库、知识图谱（如维基百科）和外部网络搜索能力，NEV-DiagBot 致力于为新能源车主提供专业、准确、易懂且富有同理心的解决方案与建议。

本项目旨在解决通用大模型在专业领域知识不足、容易出现“幻觉”的问题。通过多源信息融合，提升回答的**可靠性**和**深度**，成为您身边的专属新能源汽车专家。

---

## ✨ 核心功能

### 🛠️ 智能诊断与故障码解析
- 详尽解释各种新能源汽车故障码的含义。
- 提供可能的故障原因、初步排查建议及后续行动指南。
- 针对故障提供潜在维修建议及大致费用估算（基于现有信息）。

### 📚🌐 多源知识融合
- **本地知识库**：通过 RAG 从上传的用户手册、维修文档中检索相关信息。
- **网络搜索**：集成 Tavily 搜索引擎，获取最新或本地未包含的信息。

### 💬 多轮对话支持
- 记忆对话历史，提供连贯自然的交流体验。

### 🖥️ 友好的用户界面
- 基于 **Streamlit** 构建交互式 Web 界面，清晰展示回答及信息来源。

---

## 🚀 技术栈

| 技术模块     | 工具/框架 |
|--------------|-----------|
| 核心语言     | Python 3.10+ |
| 大模型框架   | LangChain |
| 本地大模型   | Ollama（使用 `qwen3:4B` 或其他 Qwen 系列模型） |
| 嵌入模型     | Ollama（使用 `nomic-embed-text:latest`） |
| 向量数据库   | ChromaDB（用于本地知识库向量存储） |
| 网络搜索     | Tavily API |
| 文档处理     | PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader |
| 文本分割     | RecursiveCharacterTextSplitter |
| 重排序模型   | Sentence-transformers（如 `BAAI/bge-reranker-base`） |
| 前端界面     | Streamlit |

---

## 🧰 快速开始

### 1. 克隆项目仓库
```bash
git clone https://github.com/25043221/NEV-DiagBot.git
cd NEV-DiagBot
2. 安装 Python 依赖
bash
复制
编辑
pip install -r requirements.txt
3. 安装并运行 Ollama
访问 Ollama 官网 下载并安装。

下载所需模型：

bash
复制
编辑
ollama run qwen3:4B
ollama run nomic-embed-text:latest
确保服务运行在 http://localhost:11434。

4. 准备本地知识库数据
在项目根目录下创建 data 文件夹。

放入您的新能源汽车用户手册、维修文档等 .pdf、.md、.txt 格式的文件，例如：

bash
复制
编辑
./data/秦plusDMi用户手册.pdf
🛠️ 配置设置
5. 设置 Tavily API Key
注册并获取您的 Tavily API Key：Tavily 官网

运行以下命令并输入您的 API Key：

bash
复制
编辑
python -c "from load_key import load_key; load_key('TAVILY_API_KEY')"
🚀 启动应用
6. 创建 ChromaDB 向量数据库
bash
复制
编辑
python -c "from embed import create_db; create_db()"
将读取 data/ 目录文档，分块、嵌入并构建向量数据库（存储于 ./chroma_db/）。

7. 启动应用
bash
复制
编辑
streamlit run app.py
浏览器中将自动打开页面，通常是 http://localhost:8501。

🧯 故障排除
模型加载失败：检查 config.json 中 tiny_bert_model_path 是否正确。

Ollama 未启动：确保本地服务运行，且模型已下载。

Tavily Key 错误：确认 Keys.json 中 TAVILY_API_KEY 是否正确。

知识库未识别：确保 data/ 目录存在且包含文档，chunk.py 会自动查找该目录。

📅 未来规划
✅ 更深度的意图识别（支持车型/年份理解）

✅ 支持多模态输入（图片识别仪表盘警告灯）

✅ 实时诊断（对接 OBD 接口）

✅ 丰富的知识图谱源（对接行业结构化知识库）

✅ 模型微调（基于专业数据提升问答能力）

✅ 数据隐私保护（支持脱敏与安全机制）

✅ 容器化部署（Docker支持）

