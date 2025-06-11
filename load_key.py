import os
import json
import getpass

def load_key(keyname: str) -> object:
    """加载或创建密钥/配置信息

    参数:
        keyname (str): 要获取的配置键名

    返回:
        object: 对应的配置值
    """
    file_name = "Keys.json"  # 配置存储文件名

    # 如果配置文件已存在
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            Key = json.load(file)  # 加载现有配置

        # 如果键存在且不为空
        if keyname in Key and Key[keyname]:
            return Key[keyname]
        else:
            # 提示用户输入缺失的配置
            keyval = getpass.getpass("配置文件中没有相应键，请输入对应配置信息：").strip()
            Key[keyname] = keyval  # 添加新配置

            # 保存更新后的配置
            with open(file_name, "w") as file:
                json.dump(Key, file, indent=4)
            return keyval
    else:
        # 配置文件不存在时创建新配置
        keyval = getpass.getpass("配置文件中没有相应键，请输入对应配置信息：").strip()
        Key = {keyname: keyval}  # 创建新配置字典

        # 写入新配置文件
        with open(file_name, "w") as file:
            json.dump(Key, file, indent=4)
        return keyval

if __name__ == "__main__":
    one_api_base_url = load_key("ONE_API_BASE_URL")
    one_api_key = load_key("ONE_API_KEY")
    print(f"ONE_API_BASE_URL: {one_api_base_url}")
    print(f"ONE_API_KEY: {one_api_key}")
    print(load_key("TAVILY_API_KEY"))

