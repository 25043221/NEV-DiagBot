NEV-DiagBot: 新能源汽车智能诊断与知识助手 🚗💡


🌟 项目简介
NEV-DiagBot 是一个基于本地大模型（LLM）和检索增强生成（RAG）技术的智能问答系统，专注于新能源汽车领域的故障诊断、使用指南和知识咨询。

🎯 我们的目标：
通过整合本地知识库、知识图谱（维基百科）和外部网络搜索能力，NEV-DiagBot 旨在为新能源车主提供专业、准确、易懂且富有同理心的解决方案和建议。本项目旨在解决通用大模型在专业领域知识不足、容易出现“幻觉”的问题，通过多源信息融合，提升回答的可靠性和深度，成为您身边的专属新能源汽车专家。

✨ 核心功能
智能诊断与故障码解析 🛠️
详尽解释各种新能源汽车故障码的含义。
提供可能的故障原因、初步排查建议及后续行动指南。
针对故障给出潜在的维修建议和大致费用估算（基于现有信息）。
多源知识融合 📚🌐
本地知识库：通过RAG从本地上传的用户手册、维修文档中检索相关信息。
知识图谱增强：利用BERT提取关键词，并通过维基百科API丰富相关概念解释。
网络搜索：集成Tavily搜索引擎，获取最新或本地知识库中未包含的信息。
用户意图识别 🧠
智能识别用户提问意图（如故障诊断、保养指南、功能咨询、通用知识），优化信息检索和回答策略。
多轮对话支持 💬
记忆对话历史，提供连贯自然的交流体验。
友好的用户界面 🖥️
基于Streamlit构建的交互式Web界面，清晰展示回答和信息来源。
🚀 技术栈
核心语言：Python 3.10+
大模型框架：LangChain
本地大模型：Ollama (本项目使用 qwen3:4B 或其他Qwen系列模型)
嵌入模型：Ollama (nomic-embed-text:latest 或兼容模型)
向量数据库：ChromaDB (持久化存储本地知识库向量)
知识图谱提取：
jieba (中文分词)
scikit-learn (TF-IDF)
transformers & PyTorch (BERT模型，用于关键词嵌入和相似度计算)
wikipedia (维基百科API)
网络搜索：Tavily API
文档处理：PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader
文本分割：RecursiveCharacterTextSplitter
Reranker：sentence-transformers (如 BAAI/bge-reranker-base，用于重排检索结果，提高相关性)
前端界面：Streamlit
快速开始
1. 环境准备 准备环境，就像给爱车做保养！ 🔧
克隆项目仓库：
Bash

git clone https://github.com/25043221/NEV-DiagBot.git
cd NEV-DiagBot
安装Python依赖：
Bash

pip install -r requirements.txt
(确保 requirements.txt 中包含所有必需库，如：jieba, numpy, scikit-learn, transformers, wikipedia, torch, langchain-community, langchain-core, langchain-ollama, chromadb, streamlit, tavily-python, pymupdf, unstructured, sentence-transformers)
安装并运行Ollama：
访问 Ollama官网 下载并安装Ollama。
下载所需模型。本项目默认使用 qwen3:4B 作为LLM，nomic-embed-text:latest 作为嵌入模型。您可以在终端运行：
Bash

ollama run qwen3:4B
ollama run nomic-embed-text:latest
确保Ollama服务在 http://localhost:11434 运行。
下载BERT模型：
本项目知识图谱模块使用BERT模型进行关键词提取。请下载一个中文BERT模型，例如Hugging Face上的 hfl/chinese-macbert-base。
将下载的模型文件（config.json, pytorch_model.bin, tokenizer.json, vocab.txt 等）解压到项目根目录下的 ./bert_models/chinese-macbert-base 文件夹中。请确保 config.json 中 tiny_bert_model_path 的路径与实际路径一致。
准备本地知识库数据：
在项目根目录下创建 data 文件夹。
将您的新能源汽车用户手册、维修文档等PDF、Markdown或TXT格式的文件放入 data 文件夹中。例如：./data/秦plusDMi用户手册.pdf。
2. 配置设置 🛠️ 简单配置，让助手更懂你！
设置Tavily API Key：
本项目使用Tavily进行网络搜索。请访问 Tavily官网 注册并获取您的API Key。
运行 load_key.py 文件，它将提示您输入Tavily API Key 并保存到 Keys.json 中：
Bash

python -c "from load_key import load_key; load_key('TAVILY_API_KEY')"
（输入您的Tavily API Key 后回车）
3. 运行应用 🚀 一键启动您的智能助手！
创建ChromaDB向量数据库：
首次运行前，需要构建本地知识库的向量数据库。
在终端运行：
Bash

python -c "from embed import create_db; create_db()"
此步骤会读取 data 目录下的文档，进行分块、嵌入并存储到 ./chroma_db 目录中。
启动Streamlit应用：
Bash

streamlit run app.py
应用将在您的浏览器中自动打开（通常是 http://localhost:8501）。
4. 故障排除 🚧 遇到问题？别担心，我们来帮你！
如果遇到模型加载失败，请检查 config.json 中 tiny_bert_model_path 是否正确，以及BERT模型文件是否完整放置。
确保Ollama服务正在运行，且所需模型（qwen3:4B, nomic-embed-text:latest）已下载。
检查 Keys.json 中 TAVILY_API_KEY 是否正确设置。
如果 data 目录下没有PDF文件或者路径不对，请确认 data 目录及其内容是否存在。chunk.py 会自动寻找当前项目下的 data 文件夹。
📅 未来规划
我们对 NEV-DiagBot 的未来充满期待，以下是部分规划：

更深度的意图识别：结合车辆型号、年份等信息进行更精确的意图判断。
多模态输入：支持图片输入（例如：识别仪表盘警告灯），进行诊断。
实时数据接口：对接OBD接口，实时获取车辆数据进行辅助诊断。
更丰富的知识图谱源：整合更多汽车行业的结构化知识库。
模型微调：基于专业数据集对Qwen3模型进行领域微调，进一步提升专业问答能力和降低幻觉。
安全与隐私：增强数据脱敏和隐私保护机制。
部署优化：提供更便捷的部署方案，如Docker容器化。
