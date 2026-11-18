"""
LangChain Models - 模型调用

演示 BaseChatModel 的常用调用方式：

1. invoke()              同步调用：字符串消息
2. invoke()              同步调用：字典消息列表
3. invoke()              同步调用：消息对象列表
4. stream()              流式调用
5. batch()               批量调用
6. batch_as_completed()  按完成顺序获取批量结果

模型初始化：
    复用 models.init_llm 中已创建的 deepseek_llm。

参考文档：
    https://docs.langchain.com/oss/python/langchain/models
    https://docs.langchain.com/oss/python/langchain/messages
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage)

from models import init_llm


def single_message(chat_model: BaseChatModel) -> None:
    """
    使用字符串执行单次模型调用。

    字符串会被 LangChain 转换为用户消息（HumanMessage）。

    Args:
        chat_model: 已初始化的聊天模型实例。
    """
    # invoke() 为同步调用，等待模型返回完整结果
    response = chat_model.invoke("请简单说明一下 LangChain 的作用")

    # 返回值通常为 AIMessage，content 保存消息内容
    print(f"单消息结果：{response.content}")


def dict_message(chat_model: BaseChatModel) -> None:
    """
    使用字典列表向模型传递多条消息。

    消息格式：
        role: 消息角色
        content: 消息内容

    常见 role：
        system: 系统消息，定义角色和行为要求
        user: 用户消息
        assistant: 助手的历史回复
        tool: 工具执行结果，需符合工具调用协议

    Args:
        chat_model: 已初始化的聊天模型实例。
    """
    # 模拟一段历史对话，并在末尾提出新问题
    conversation = [
        {
            "role": "system",
            "content": "你是一位诗歌专家"
        },
        {
            "role": "user",
            "content": "帮我写一篇描述春天的50字左右的短文"
        },
        {
            "role": "assistant",
            "content": "春风拂面，柳枝轻摇，花香随风飘散。"
        },
        {
            "role": "user",
            "content": "请在刚才的内容基础上继续完善，控制在50字左右"
        }
    ]

    # 模型会结合传入的消息列表生成新一轮回复
    response = chat_model.invoke(conversation)
    print(f"字典消息结果：{response.content}")


def class_message(chat_model: BaseChatModel) -> None:
    """
    使用 LangChain 消息对象构建模型输入。

    SystemMessage: 系统消息
    HumanMessage: 用户消息
    AIMessage: 助手消息

    与字典消息相比，消息对象具有明确的 Python 类型，
    也便于后续处理工具调用、元数据和消息状态。

    Args:
        chat_model: 已初始化的聊天模型实例。
    """
    conversation = [
        SystemMessage(content="你是一位诗歌专家"),
        HumanMessage(
            content="帮我写一篇描述秋天的50字左右的短文"
        ),
        # 模拟之前一轮对话中助手已经给出的回复
        AIMessage(content="秋叶飘落，晚风微凉，金黄的田野铺向远方。"),
        # 当前轮次的用户消息
        HumanMessage(
            content="请继续完善刚才的短文，增加一些秋天的景物描写"
        )
    ]

    response = chat_model.invoke(conversation)
    print(f"消息对象结果：{response.content}")


def stream_invoke(chat_model: BaseChatModel) -> None:
    """
    使用 stream() 流式调用模型。

    返回值是迭代器，通常逐次产生 AIMessageChunk。
    适用于终端实时显示、聊天页面和增量输出。

    Args:
        chat_model: 已初始化的聊天模型实例。
    """
    # stream() 返回迭代器，遍历时逐步获取生成的片段
    response = chat_model.stream("请你介绍一下你自己")

    print("流式调用结果：", end="", flush=True)

    for chunk in response:
        # chunk.content 在普通文本调用中通常是字符串
        # end=""：连续输出，不自动换行
        # flush=True：立即刷新终端输出缓冲区
        print(chunk.content, end="", flush=True)

    print()


def batch_invoke(chat_model: BaseChatModel) -> None:
    """
    批量调用模型。

    batch():
        等待批量处理完成，结果按输入顺序返回。

    batch_as_completed():
        返回迭代器，按照任务实际完成顺序返回结果。
        每个结果为 (index, result)。

    max_concurrency:
        限制客户端同时执行的任务数量，
        不代表模型服务端的最大并发能力。

    Args:
        chat_model: 已初始化的聊天模型实例。
    """
    # --------------------------------------------------------
    # 1. batch()：按输入顺序返回
    # --------------------------------------------------------
    questions = [
        "请你介绍一下你自己",
        "汽车是如何工作的",
        "火车是如何工作的"
    ]

    responses = chat_model.batch( questions,
        config={"max_concurrency": 2}
    )

    # 无论请求的实际完成先后顺序如何，
    # responses 与 questions 的索引保持对应
    for index, response in enumerate(responses):
        print("-" * 60)
        print(f"问题：{questions[index]}")
        print(f"回答：{response.content}")

    # --------------------------------------------------------
    # 2. batch_as_completed()：按任务完成顺序返回
    # --------------------------------------------------------
    questions = [
        "大语言模型是如何工作的",
        "什么是LLM"
    ]

    results = chat_model.batch_as_completed(
        questions,
        config={"max_concurrency": 2},
        # 让单个失败任务以异常对象形式返回，
        # 避免调用方无法区分失败结果
        return_exceptions=True
    )

    # index 是原始输入索引，不代表完成次序
    for index, result in results:
        print("-" * 60)
        print(f"问题索引：{index}")
        print(f"问题：{questions[index]}")

        if isinstance(result, Exception):
            print(f"调用失败：{result}")
            continue

        print(f"回答：{result.content}")


if __name__ == "__main__":
    # 复用之前初始化的模型，不在当前示例重复创建
    # 注意与 models/init_llm.py 中实际定义的变量名一致
    deepseek = init_llm.deepseek_llm

    print("=" * 20 + " 单条消息调用 " + "=" * 20)

    # 1. 字符串消息
    single_message(deepseek)

    # 2. 字典消息
    dict_message(deepseek)

    # 3. 消息对象
    class_message(deepseek)

    print("\n" + "=" * 20 + " 流式调用 " + "=" * 20)
    stream_invoke(deepseek)

    print("\n" + "=" * 20 + " 批量调用 " + "=" * 20)
    batch_invoke(deepseek)
