
"""
LangChain Models - 模型调用

演示 BaseChatModel 的同步与异步调用方式。

一、同步调用
1. invoke()               字符串消息
2. invoke()               字典消息列表
3. invoke()               消息对象列表
4. stream()               流式调用
5. batch()                批量调用
6. batch_as_completed()   按完成顺序获取结果

二、异步调用
1. ainvoke()              异步字符串消息
2. ainvoke()              异步字典消息
3. ainvoke()              异步消息对象
4. astream()              异步流式调用
5. abatch()               异步批量调用
6. abatch_as_completed()  按完成顺序获取异步结果

模型初始化：
    复用 models.init_llm 中已定义的 deepseek_llm。

参考文档：
    https://docs.langchain.com/oss/python/langchain/models
    https://docs.langchain.com/oss/python/langchain/messages
"""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from models import init_llm


# ============================================================
# 一、同步调用
# ============================================================

def single_message(chat_model: BaseChatModel) -> None:
    """
    使用字符串执行同步模型调用。

    字符串会被作为用户消息处理。

    Args:
        chat_model: 已初始化的聊天模型。
    """
    # invoke() 等待模型生成完整结果后返回
    response = chat_model.invoke(
        "请简单说明一下 LangChain 的作用"
    )

    # 聊天模型通常返回 AIMessage
    print(f"单消息结果：{response.content}")


def dict_message(chat_model: BaseChatModel) -> None:
    """
    使用字典消息列表调用模型。

    常见 role：
        system: 系统消息
        user: 用户消息
        assistant: 助手历史回复
        tool: 工具消息，需符合工具调用协议

    Args:
        chat_model: 已初始化的聊天模型。
    """
    # 构造历史对话及本轮用户问题
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
            "content": "请完善刚才的内容，控制在50字左右"
        }
    ]

    response = chat_model.invoke(conversation)
    print(f"字典消息结果：{response.content}")


def class_message(chat_model: BaseChatModel) -> None:
    """
    使用 LangChain 消息对象调用模型。

    SystemMessage: 系统消息
    HumanMessage: 用户消息
    AIMessage: 助手消息

    Args:
        chat_model: 已初始化的聊天模型。
    """
    conversation = [
        SystemMessage(
            content="你是一位诗歌专家"
        ),
        HumanMessage(
            content="帮我写一篇描述秋天的50字左右的短文"
        ),
        # 模拟助手在上一轮对话中的回复
        AIMessage(
            content="秋叶飘落，晚风微凉，金黄的田野铺向远方。"
        ),
        # 本轮用户问题
        HumanMessage(
            content="请完善刚才的短文，增加一些秋天的景物描写"
        )
    ]

    response = chat_model.invoke(conversation)
    print(f"消息对象结果：{response.content}")


def stream_invoke(chat_model: BaseChatModel) -> None:
    """
    使用 stream() 同步流式调用模型。

    返回迭代器，通常逐次产生 AIMessageChunk。
    """
    response = chat_model.stream(
        "请你介绍一下你自己"
    )

    print("流式调用结果：", end="", flush=True)

    # 逐步获取模型生成的消息片段
    for chunk in response:
        # 当前示例使用纯文本模型输出
        print(chunk.content, end="", flush=True)

    print()


def batch_invoke(chat_model: BaseChatModel) -> None:
    """
    同步批量调用。

    batch():
        按输入顺序返回结果列表。

    batch_as_completed():
        按任务完成顺序返回 (index, result)。
    """

    # 1. batch()
    questions = [
        "请你介绍一下你自己",
        "汽车是如何工作的",
        "火车是如何工作的"
    ]

    responses = chat_model.batch(
        questions,
        config={"max_concurrency": 2}
    )

    # 返回结果顺序与输入顺序一致
    for index, response in enumerate(responses):
        print("-" * 60)
        print(f"问题：{questions[index]}")
        print(f"回答：{response.content}")

    # 2. batch_as_completed()
    questions = [
        "大语言模型是如何工作的",
        "什么是 LLM"
    ]

    results = chat_model.batch_as_completed(
        questions,
        config={"max_concurrency": 2},
        return_exceptions=True
    )

    # index 为原始输入索引，并非完成次序
    for index, result in results:
        print("-" * 60)
        print(f"问题索引：{index}")
        print(f"问题：{questions[index]}")

        if isinstance(result, Exception):
            print(f"调用失败：{result}")
            continue

        print(f"回答：{result.content}")


# ============================================================
# 二、异步调用
# ============================================================

async def async_single_message(chat_model: BaseChatModel) -> None:
    """
    使用 ainvoke() 异步调用模型。

    与 invoke() 不同，ainvoke() 返回可等待对象，
    通过 await 等待模型执行结果。

    Args:
        chat_model: 已初始化的聊天模型。
    """

    # await 会等待当前调用完成，等待期间事件循环可以调度其他异步任务
    response = await chat_model.ainvoke("请简单说明一下 LangChain 的作用")

    print(f"异步单消息结果：{response.content}")


async def async_dict_message(chat_model: BaseChatModel) -> None:
    """
    使用字典消息列表异步调用模型。

    ainvoke() 支持与 invoke() 相同的消息输入格式。
    """
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
            "content": "春风轻拂，细雨润花，柳枝在河畔舒展。"
        },
        {
            "role": "user",
            "content": "请在此基础上完善短文，控制在50字左右"
        }
    ]

    response = await chat_model.ainvoke(conversation)

    print(f"异步字典消息结果：{response.content}")


async def async_class_message(chat_model: BaseChatModel) -> None:
    """
    使用 LangChain 消息对象异步调用模型。

    消息对象与同步调用相同，
    区别是使用 await 等待模型响应。
    """
    conversation = [
        SystemMessage(
            content="你是一位诗歌专家"
        ),
        HumanMessage(
            content="帮我写一篇描述秋天的50字左右的短文"
        ),
        # 模拟历史回复
        AIMessage(
            content="秋风吹过田野，金黄的落叶随风飘落。"
        ),
        HumanMessage(
            content="请在刚才的内容基础上补充秋天的景物描写"
        )
    ]

    response = await chat_model.ainvoke(conversation)

    print(f"异步消息对象结果：{response.content}")


async def async_stream_invoke(chat_model: BaseChatModel) -> None:
    """
    使用 astream() 执行异步流式调用。

    astream() 返回异步迭代器，
    使用 async for 逐步获取模型输出。

    普通文本场景下，每个片段通常为 AIMessageChunk。
    """
    print("异步流式结果：", end="", flush=True)

    # astream() 返回异步迭代器
    # 不需要对 astream() 本身使用 await
    async for chunk in chat_model.astream("请你介绍一下你自己" ):
        # 模型生成一个片段，就输出一个片段
        print(chunk.content, end="", flush=True)

    print()


async def async_batch_invoke( chat_model: BaseChatModel) -> None:
    """
    异步批量调用。

    abatch():
        并发执行多个异步调用，
        等待全部处理完成后按输入顺序返回结果。

    abatch_as_completed():
        并发执行多个异步调用，
        按任务实际完成顺序逐个返回结果。

    max_concurrency:
        限制客户端同时执行的任务数量。

    return_exceptions:
        为 True 时，单个任务异常会作为结果返回。
    """

    # --------------------------------------------------------
    # 1. abatch()：异步批量处理
    # --------------------------------------------------------

    questions = [
        "请你介绍一下你自己",
        "汽车是如何工作的",
        "火车是如何工作的"
    ]

    # abatch() 是异步方法，需要使用 await
    responses = await chat_model.abatch(
        questions,
        config={"max_concurrency": 2},
        return_exceptions=True
    )

    # 结果列表与 questions 的输入顺序保持一致
    for index, result in enumerate(responses):
        print("-" * 60)
        print(f"问题：{questions[index]}")

        if isinstance(result, Exception):
            print(f"调用失败：{result}")
            continue

        print(f"回答：{result.content}")

    # --------------------------------------------------------
    # 2. abatch_as_completed()：按完成顺序处理
    # --------------------------------------------------------

    questions = [
        "大语言模型是如何工作的",
        "什么是 LLM",
        "LangChain 和 LangGraph 有什么区别"
    ]

    # 返回异步迭代器
    # 每项结果包含原始输入索引及模型响应
    results = chat_model.abatch_as_completed(
        questions,
        config={
            "max_concurrency": 2
        },
        return_exceptions=True
    )

    # 异步迭代：哪个请求先完成，就先处理哪个结果
    async for index, result in results:
        print("-" * 60)
        print(f"问题索引：{index}")
        print(f"问题：{questions[index]}")

        if isinstance(result, Exception):
            print(f"调用失败：{result}")
            continue

        print(f"回答：{result.content}")


# ============================================================
# 三、异步执行入口
# ============================================================

async def run_async_calls(
    chat_model: BaseChatModel
) -> None:
    """
    统一执行异步调用示例。

    此处顺序使用 await，便于观察每种调用方式。
    各调用之间并未同时执行。

    abatch() 和 abatch_as_completed() 内部可以并发处理任务。
    """

    print("\n" + "=" * 20 + " 异步消息调用 " + "=" * 20)

    await async_single_message(chat_model)
    await async_dict_message(chat_model)
    await async_class_message(chat_model)

    print("\n" + "=" * 20 + " 异步流式调用 " + "=" * 20)

    await async_stream_invoke(chat_model)

    print("\n" + "=" * 20 + " 异步批量调用 " + "=" * 20)

    await async_batch_invoke(chat_model)


# ============================================================
# 四、主程序入口
# ============================================================

if __name__ == "__main__":

    # 复用 models/init_llm.py 中已创建的模型实例
    # 注意：变量名称需与实际初始化代码保持一致
    deepseek = init_llm.deepseek_llm

    # --------------------------------------------------------
    # 同步调用
    # --------------------------------------------------------

    print("=" * 20 + " 单条消息调用 " + "=" * 20)

    single_message(deepseek)
    dict_message(deepseek)
    class_message(deepseek)

    print("\n" + "=" * 20 + " 流式调用 " + "=" * 20)

    stream_invoke(deepseek)

    print("\n" + "=" * 20 + " 批量调用 " + "=" * 20)

    batch_invoke(deepseek)

    # --------------------------------------------------------
    # 异步调用
    # --------------------------------------------------------

    # asyncio.run() 创建并运行事件循环，
    # 等待 run_async_calls() 中的异步调用完成。
    #
    # 该写法适用于普通 Python 脚本。
    # 如果已有运行中的事件循环（例如 Jupyter），
    # 应在现有异步环境中直接 await run_async_calls(...)。
    asyncio.run(run_async_calls(deepseek))
