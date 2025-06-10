import base64
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama


class MultimodalModel:
    def __init__(self, model_name: str = "qwen2.5vl:3b", base_url: str = "http://localhost:11434"):
        """
        初始化用于图像描述的多模态模型。

        参数：
            model_name: Ollama 多模态模型的名称（例如 "qwen2.5vl:3b"、"llava:latest"）。
            base_url: Ollama 服务的基础 URL。
        """
        self.model = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)  # 使用较低温度以获得一致性描述
        print(f"已初始化多模态模型：{model_name}")

    def describe_image(self, image_bytes: bytes) -> str:
        """
        使用多模态模型生成图像的文字描述。

        参数：
            image_bytes: 图像的原始字节数据。

        返回：
            图像的文字描述字符串；如果失败，则返回错误信息。
        """
        print("正在使用多模态模型生成图像描述...")
        if not image_bytes:
            return "没有图片数据可供描述。"

        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "请概述这张图片的内容。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            )
            response = self.model.invoke([message])
            print("图像描述生成成功。")
            return response.content
        except Exception as e:
            print(f"生成图像描述时出错：{e}")
            return f"无法生成图片描述：{e}"


if __name__ == "__main__":
    try:
        with open("./data/test2.jpg", "rb") as f:
            test_image_bytes = f.read()

        multimodal_agent = MultimodalModel()
        description = multimodal_agent.describe_image(test_image_bytes)
        print("\n生成的描述：")
        print(description)
    except FileNotFoundError:
        print("请创建一个名为 'data' 的文件夹，并将一张名为 'test.jpg' 的图片放入其中以进行测试。")
    except Exception as e:
        print(f"测试过程中发生错误：{e}。请确保 Ollama 正在运行，并且已经拉取 'qwen2.5vl:3b' 模型。")