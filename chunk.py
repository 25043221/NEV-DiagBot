import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 获取pdf文件内容
def get_pdf_text():
    folder_path = "/home/yu/zjh/my_app/data"
    # 获取folder_path下所有⽂件路径，储存在file_paths⾥
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths[:3])
    # 遍历⽂件路径并把实例化的loader存放在loaders⾥
    loaders = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))
        elif file_type == 'txt':
            loaders.append(TextLoader(file_path, encoding='utf-8'))

    texts = []
    for loader in loaders: texts.extend(loader.load())
    return texts


# 拆分文本
def get_text_chunks():
    content = get_pdf_text()
    full_text = "\n".join(doc.page_content for doc in content)
    chunks = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50).split_text(full_text)
    return chunks

if __name__ == "__main__":
    data = get_pdf_text()
    print(data)
    chunks:  List[str] = get_text_chunks()
    for datas in chunks:
        print(datas)
        print("----------------")
