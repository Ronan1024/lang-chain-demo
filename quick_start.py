
"""
LangChain 快速入门：DeepSeek + Agent + Tool

一、示例说明
通过 LangChain 接入 DeepSeek 大语言模型，
创建一个停车场管理智能体（Agent），
使模型能够根据用户的问题调用指定工具，
查询停车场的车位使用情况。

二、基础环境
1. 安装 LangChain
   pip install -U langchain

2. 安装 DeepSeek 模型集成包
   pip install -U langchain-deepseek

三、执行流程
用户输入问题
    ↓
Agent 接收消息
    ↓
DeepSeek 分析用户意图
    ↓
判断是否需要调用工具
    ↓
调用 get_car_number 工具
    ↓
获取停车场车位信息
    ↓
DeepSeek 结合工具结果生成回答
    ↓
返回最终结果

注意：
当前停车场数据为模拟数据，
不涉及真实数据库或停车场管理系统。
"""

import os

# LangChain Agent 创建接口
# 用于创建具备工具调用能力的智能体
from langchain.agents import create_agent

# LangChain 工具装饰器
# 将普通 Python 函数注册为 Agent 可以调用的工具
from langchain_core.tools import tool

# DeepSeek 模型集成
# 提供与 DeepSeek 大语言模型交互的统一接口
from langchain_deepseek import ChatDeepSeek


# ============================================================
# 1. 读取模型连接配置
# ============================================================

# 从操作系统环境变量中读取 API Key
# 避免将密钥直接写入代码，防止敏感信息泄露
API_KEY = os.getenv("API_KEY")

# 自定义模型 API 地址
# 例如：https://api.deepseek.com
BASE_URL = os.getenv("BASE_URL")

# ============================================================
# 2. 定义 Agent 可以调用的工具
# ============================================================

@tool
def get_car_number(name: str) -> str:
    """
    查询指定停车场的车位使用情况。

    该函数通过 @tool 装饰器注册为 LangChain 工具，
    Agent 可以根据用户的问题决定是否调用此工具。

    Args:
        name (str): 停车场名称。

    Returns:
        str: 停车场的车位使用情况。

    注意：
        当前使用模拟数据。
        实际项目中可以替换为数据库查询或 API 调用。
    """

    # 模拟停车场车位数据
    # 实际业务中可以通过 MySQL、Redis 或 HTTP API 获取
    occupied_spaces = 500
    available_spaces = 10

    # 将查询结果返回给 Agent
    # 模型会根据此结果组织自然语言回答
    return (
        f"{name} 当前已使用车位 {occupied_spaces} 个，"
        f"剩余可用车位 {available_spaces} 个。"
    )


# ============================================================
# 3. 初始化 DeepSeek 大语言模型
# ============================================================

# 创建 DeepSeek 模型客户端
# ChatDeepSeek 封装了模型请求及响应处理过程
deep_seek = ChatDeepSeek(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url=BASE_URL,
    # 控制生成结果的随机性 较低的 temperature 适合业务查询类场景
    temperature=0.2,
)


# ============================================================
# 4. 创建 Agent 智能体
# ============================================================

# create_agent 用于构建能够调用工具的智能体
#
# Agent 的主要职责：
# 1. 接收并理解用户问题
# 2. 根据模型判断是否需要调用工具
# 3. 选择适当的工具并传入参数
# 4. 获取工具执行结果
# 5. 根据工具返回的信息生成最终回答

agent = create_agent(

    # 指定 Agent 使用的大语言模型
    model=deep_seek,

    # 注册 Agent 可以调用的工具
    # 这里传入的是由 @tool 装饰的工具对象
    # 可以同时注册多个不同类型的工具
    tools=[get_car_number],

    # 系统提示词
    # 定义 Agent 的身份、职责和行为规则
    system_prompt=(
        """你是一个专业的停车场管理助手。
        你可以帮助用户查询指定停车场的车位使用情况。
        当用户询问停车场车位信息时，
        应调用 get_car_number 工具获取数据。
        回答应当准确、简洁，并以工具返回的数据为依据。
        不要自行编造停车场数据。"""
    ),
)


# ============================================================
# 5. 调用 Agent 执行任务
# ============================================================

# invoke() 用于同步执行 Agent
#
# messages 是模型接收的消息列表：
# role="user" 表示用户消息
# content 表示用户输入的具体内容
#
# Agent 接收到消息后：
# 1. 将用户消息发送给 DeepSeek
# 2. 模型根据问题决定调用 get_car_number
# 3. LangChain 执行工具并获得结果
# 4. 将工具结果交给模型
# 5. 模型生成最终自然语言回答

resp = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "牛马停车场的车位情况"
        }
    ]
})


# ============================================================
# 6. 输出 Agent 执行结果
# ============================================================

# 返回结果中包含 Agent 执行期间产生的消息
# 可能包括：
# - HumanMessage：用户输入的消息
# - AIMessage：模型返回的消息或工具调用请求
# - ToolMessage：工具执行后的返回结果
#
# 打印全部消息，便于观察 Agent 的工具调用过程
print("========== Agent 执行过程 ==========")

for message in resp["messages"]:
    print(f"\n消息类型：{message.type}")
    print(f"消息内容：{message.content}")

    # 如果模型发起了工具调用，额外输出工具调用信息
    if getattr(message, "tool_calls", None):
        print(f"工具调用：{message.tool_calls}")


# ============================================================
# 7. 获取 Agent 最终回答
# ============================================================

# 正常完成执行时，最后一条消息通常是模型最终回答
# 此处使用 content 获取模型生成的文本内容

print("\n========== Agent 最终回答 ==========")

print(resp["messages"][-1].content)
