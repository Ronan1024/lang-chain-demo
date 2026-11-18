
"""
LangChain Models - 模型初始化

LangChain 提供两种常用的聊天模型初始化方式：

1. Model Class
   直接使用模型提供商的集成类创建实例。
   例如：ChatDeepSeek、ChatOpenAI、ChatAnthropic。

2. init_chat_model()
   通过 LangChain 的统一接口初始化模型。
   根据 model_provider 选择对应的模型集成类。

两种方式创建的模型均支持 LangChain 标准调用接口，
例如 invoke()、stream() 和 batch()。

运行环境：
    Python >= 3.10

安装依赖：
    pip install -U langchain
    pip install -U langchain-deepseek

环境变量：
    DEEP_SEEK_API_KEY   DeepSeek API 密钥
    DEEP_SEEK_BASE_URL  DeepSeek API 地址

参考文档：
    https://docs.langchain.com/oss/python/langchain/models
    https://docs.langchain.com/oss/python/integrations/providers/overview
    https://docs.langchain.com/oss/python/integrations/chat/deepseek
"""

import os

# LangChain 统一模型初始化接口
from langchain.chat_models import init_chat_model

# DeepSeek 模型集成类
from langchain_deepseek import ChatDeepSeek


# ============================================================
# 1. 模型基础配置
# ============================================================

# 从环境变量中读取 API Key，避免在代码中硬编码密钥
DEEP_SEEK_API_KEY = os.getenv("API_KEY")

# DeepSeek API 地址，未配置时使用官方地址
DEEP_SEEK_BASE_URL = os.getenv(
    "BASE_URL",
    "https://api.deepseek.com"
)

# 使用 DeepSeek 官方支持的模型名称
# 模型名称及版本以 DeepSeek 官方 API 文档为准
MODEL_NAME = "deepseek-flash"

# 初始化前检查必要配置
if not DEEP_SEEK_API_KEY:
    raise ValueError(
        "未配置 DEEP_SEEK_API_KEY 环境变量"
    )


# ============================================================
# 2. Model Class 方式初始化
# ============================================================

# 直接使用 ChatDeepSeek 创建模型实例。
#
# 这种方式明确指定模型提供商的实现类，
# 可以直接使用该模型集成类支持的初始化参数。
#
# 不同模型提供商可能具有额外的专属参数，
# 具体参数应查看对应的 LangChain Integrations 文档。

deepseek_llm_by_class = ChatDeepSeek(

    # 模型名称
    model=MODEL_NAME,

    # API 身份认证密钥
    api_key=DEEP_SEEK_API_KEY,

    # 模型服务地址
    base_url=DEEP_SEEK_BASE_URL,

    # 采样温度
    # 控制支持该参数的生成模式下的随机性。
    # DeepSeek 思考模式下此参数不生效。
    temperature=0.2
)


# ============================================================
# 3. init_chat_model 方式初始化
# ============================================================

# init_chat_model() 是 LangChain 提供的统一初始化入口。
#
# 通过 model_provider 指定模型提供商，
# LangChain 会选择对应的模型集成类。
#
# 当前 model_provider="deepseek"，
# 因此实际创建的仍然是 ChatDeepSeek 实例。
#
# 该方式适合需要统一管理模型配置、
# 或者后续可能接入多个模型提供商的项目。
#
# 注意：
# init_chat_model() 不会自动安装模型集成包，
# 使用 DeepSeek 仍需安装 langchain-deepseek。

deepseek_llm = init_chat_model(

    # 模型名称
    model=MODEL_NAME,

    # 模型提供商
    # 指定为 deepseek 时使用 ChatDeepSeek 实现
    model_provider="deepseek",

    # API 身份认证密钥
    api_key=DEEP_SEEK_API_KEY,

    # 模型服务地址
    base_url=DEEP_SEEK_BASE_URL,

    # 模型采样温度
    # 参数会传递给底层模型实现
    temperature=0.2
)


# ============================================================
# 4. 模型初始化结果
# ============================================================

# 两种方式只负责创建模型客户端。
# 初始化阶段通常不会向 DeepSeek 发送推理请求。

print("Model Class 初始化结果：")
print(type(deepseek_llm_by_class).__name__)

print("\ninit_chat_model 初始化结果：")
print(type(deepseek_llm).__name__)

# ============================================================
# 补充说明
# ============================================================

# 1. Model Class 与 init_chat_model 的区别主要在初始化方式。
#
#    Model Class：
#        直接实例化模型提供商对应的类。
#
#    init_chat_model：
#        根据模型名称和提供商创建模型实例。
#
# 2. 两种方式都可以使用 invoke()、stream() 等标准接口。
#
# 3. init_chat_model 并不会统一所有提供商的专属参数。
#    实际支持的参数仍然取决于底层模型实现。
#
# 4. init_chat_model 还支持以下模型命名方式：
#
#    init_chat_model(
#        model="deepseek:deepseek-flash",
#        api_key=DEEP_SEEK_API_KEY,
#        base_url=DEEP_SEEK_BASE_URL
#    )
#
#    此时可以不再单独指定 model_provider。
#
# 5. 本例沿用 DEEP_SEEK_API_KEY 环境变量名称，
#    与 langchain-deepseek 默认识别的 DEEPSEEK_API_KEY 不同。
#    由于显式传入了 api_key，因此不影响模型初始化。
