
"""
LangChain Models - 模型结构化输出

一、模型结构化输出
1. Pydantic
2. TypedDict
3. JSON Schema

二、输出解析器
1. JsonOutputParser
2. PydanticOutputParser
3. OutputFixingParser

三、Agent 结构化输出
1. ToolStrategy
2. ProviderStrategy
3. 自动选择策略
4. 结构化输出错误处理

参考：
https://docs.langchain.com/oss/python/langchain/models
https://docs.langchain.com/oss/python/langchain/structured-output
"""

from typing import List, TypedDict, Annotated, Union

from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_classic.output_parsers import OutputFixingParser
from models.init_llm import deepseek_llm


# ============================================================
# 1. Pydantic 模型
# ============================================================

class Actor(BaseModel):
    """
    演员信息。

    BaseModel 提供数据解析与校验能力。
    Field 用于声明字段描述及约束。
    """
    name: str = Field(description="演员名称")
    role: str = Field(description="出演角色")
    age: int = Field(description="年龄")


class Movie(BaseModel):
    """
    电影信息。

    actor 使用 List[Actor] 定义嵌套结构，
    表示一个电影包含多个演员对象。
    """
    name: str = Field(description="电影名称")  # 基础简单格式
    desc: str = Field(description="电影描述")
    time: str = Field(description="上映时间")
    star: float = Field(description="电影评分")
    actor: List[Actor] = Field(description="演员列表")  # 嵌套结构


# ============================================================
# 2. TypedDict 模型
# ============================================================

class Car(TypedDict):
    """
    汽车信息。

    TypedDict 描述字典字段及类型。
    Annotated 为字段附加描述信息。

    TypedDict 本身不执行运行时数据校验。
    """
    brand: Annotated[str, '汽车品牌']
    type: Annotated[str, '类型']
    seats: Annotated[int, '座位数']
    name: Annotated[str, '名称']
    sales: Annotated[List[int], '截止目前的销量']
    year: Annotated[str, '上市年份']


# ============================================================
# 3. JSON Schema
# ============================================================

"""
使用 JSON Schema 定义电脑信息。

type：定义数据类型
properties：定义对象字段
description：字段描述
items：定义数组元素结构
required：定义必填字段
"""

project_schema = {
    "title": "ComputerInfo",
    "description": "电脑信息",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "电脑名称"},
        "brand": {"type": "string", "description": "电脑品牌"},
        "configuration": {
            "type": "array",
            "description": "配置信息",
            "items": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "内存大小"
                    },
                    "disk": {
                        "type": "string",
                        "description": "磁盘大小"
                    }
                },
                "required": ["memory", "disk"]
            }
        }
    },
    "required": ["name", "brand", "configuration"]
}


# ============================================================
# 4. JsonOutputParser
# ============================================================

def output_parser(deepseek):
    """
    使用 JsonOutputParser 解析模型输出。

    1. 根据 Movie 生成 JSON 格式说明。
    2. 通过 ChatPromptTemplate 组织提示词。
    3. 使用 LCEL 组合 prompt、model 和 parser。

    返回结果是 dict，不是 Movie 对象。
    """

    # 指定 Pydantic 模型，用于生成 JSON 格式说明
    parser = JsonOutputParser(pydantic_object=Movie)

    # format_instructions 提供 Movie 对应的 JSON Schema
    prompt = ChatPromptTemplate.from_template("""
    请根据用户问题介绍电影。

    问题：{question}

    请严格按照以下格式返回 JSON：

    {format_instructions}
    """)

    # 将格式说明固定到提示词模板中
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # LCEL：
    # prompt 负责组织输入
    # deepseek 负责生成回答
    # parser 负责解析 JSON
    chain = prompt | deepseek | parser

    resp = chain.invoke({
        "question": "详细介绍一下电影《黑客帝国3》"
    })

    print(f"JSON 解析结果：{resp}")
    print(f"返回类型：{type(resp)}")


# ============================================================
# 5. PydanticOutputParser
# ============================================================

def pydantic_output_parser(deepseek):
    """
    使用 PydanticOutputParser 解析模型输出。

    与 JsonOutputParser 不同：
    返回经过 Pydantic 校验的 Movie 对象。
    """

    parser = PydanticOutputParser(pydantic_object=Movie)

    prompt = ChatPromptTemplate.from_template("""
    请根据用户问题介绍电影。

    问题：{question}

    请严格按照以下格式返回 JSON：

    {format_instructions}
    """)

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | deepseek | parser

    resp = chain.invoke({
        "question": "详细介绍一下电影《黑客帝国3》"
    })

    print(f"Pydantic 解析结果：{resp}")
    print(f"返回类型：{type(resp)}")

    # Pydantic 返回对象，可直接使用属性访问
    print(f"电影名称：{resp.name}")


# ============================================================
# 6. include_raw：查看原始模型输出
# ============================================================

def structured_output_raw(deepseek):
    """
    查看结构化输出的原始消息和解析结果。

    include_raw=True 返回：
        raw：模型原始消息
        parsed：解析后的结果
        parsing_error：解析异常

    仅能捕获输出解析阶段的异常，
    不会将所有 API 请求错误转换为 parsing_error。
    """

    # 当前示例使用 JSON Mode，
    # 避免强制工具选择与 DeepSeek 思考模式冲突
    structured_llm = deepseek.with_structured_output(
        Movie,
        method="json_mode",
        include_raw=True
    )

    parser = JsonOutputParser(pydantic_object=Movie)

    # JSON Mode 中需要明确提供 JSON 格式要求
    prompt = (
        "介绍电影《复仇者联盟4》。"
        "只返回符合下列格式的 JSON，不要附加其他内容。\n"
        + parser.get_format_instructions()
    )

    resp = structured_llm.invoke(prompt)

    print(f"原始消息：{resp['raw']}")
    print(f"解析结果：{resp['parsed']}")
    print(f"解析异常：{resp['parsing_error']}")


# ============================================================
# 7. OutputFixingParser：解析失败修复
# ============================================================

def output_fixing_parser(deepseek):
    """
    使用 OutputFixingParser 修复不符合格式的模型输出。

    修复时会再次调用模型，产生额外 Token 消耗。
    """

    # 原始解析器：负责解析和校验 Movie
    parser = PydanticOutputParser(pydantic_object=Movie)

    # 包装解析器：
    # 当解析失败时，让模型尝试修复输出
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser,
        llm=deepseek,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_template("""
    请介绍电影。

    问题：{question}

    请按照以下格式返回 JSON：

    {format_instructions}
    """)

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | deepseek | fixing_parser

    resp = chain.invoke({
        "question": "详细介绍一下电影《黑客帝国3》"
    })

    print(f"修复后解析结果：{resp}")


# ============================================================
# 8. Agent：ToolStrategy
# ============================================================

def agent_tool_strategy(deepseek):
    """
    使用 ToolStrategy 实现 Agent 结构化输出。

    需要模型支持工具调用。
    当前 DeepSeek 若处于思考模式，
    强制工具选择可能返回 HTTP 400。

    使用本示例前，需要确认模型配置兼容。
    """

    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    agent = create_agent(
        model=deepseek,
        tools=[],
        response_format=ToolStrategy(
            schema=Movie,

            # 结构化结果生成后，工具消息中使用的内容
            tool_message_content="电影信息已完成结构化提取",

            # 出现支持重试的结构化输出错误时进行反馈与重试
            handle_errors=True
        )
    )

    resp = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "描述电影《复仇者联盟4》"
            }
        ]
    })

    # Agent 结构化结果保存在 structured_response 中
    print(resp["structured_response"])


# ============================================================
# 9. Agent：ProviderStrategy
# ============================================================

def agent_provider_strategy(provider_model):
    """
    使用 ProviderStrategy 实现 Agent 结构化输出。

    provider_model 必须是支持原生结构化输出的模型。

    本函数用于说明 ProviderStrategy 的调用方式；
    不直接将当前 ChatDeepSeek 视为原生 Schema 模型。
    """

    from langchain.agents import create_agent
    from langchain.agents.structured_output import ProviderStrategy

    agent = create_agent(
        model=provider_model,
        tools=[],
        response_format=ProviderStrategy(Movie)
    )

    resp = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "描述电影《复仇者联盟4》"
            }
        ]
    })

    print(resp["structured_response"])


# ============================================================
# 10. Agent：自动选择结构化输出策略
# ============================================================

def agent_auto_strategy(model):
    """
    直接向 response_format 传入 Pydantic 模型。

    LangChain 根据模型能力选择：
        ProviderStrategy 或 ToolStrategy。

    具体策略取决于模型集成和能力信息。
    """

    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[],
        response_format=Movie
    )

    resp = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "描述电影《复仇者联盟4》"
            }
        ]
    })

    print(resp["structured_response"])


# ============================================================
# 11. Agent：结构化输出错误处理
# ============================================================

def agent_error_handling(deepseek):
    """
    自定义 ToolStrategy 的错误处理逻辑。

    主要处理：
        StructuredOutputValidationError：数据结构校验异常
        MultipleStructuredOutputsError：返回多个结构化结果

    仅适用于 ToolStrategy 的相应错误处理流程。
    """

    from langchain.agents import create_agent
    from langchain.agents.structured_output import (
        ToolStrategy,
        StructuredOutputValidationError,
        MultipleStructuredOutputsError,
    )

    def error_handler(error: Exception) -> str:
        # 结构化数据不符合 Schema
        if isinstance(error, StructuredOutputValidationError):
            return "返回数据不符合字段定义，请重新生成正确的结构。"

        # 模型同时返回了多个结构化结果
        if isinstance(error, MultipleStructuredOutputsError):
            return "只能返回一个结构化结果，请选择最符合问题的一个。"

        return f"结构化输出发生错误：{error}"

    agent = create_agent(
        model=deepseek,
        tools=[],

        # 允许模型根据问题选择 Movie 或 Car
        response_format=ToolStrategy(
            schema=Union[Movie, Car],
            handle_errors=error_handler
        )
    )

    resp = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "介绍小米YU7汽车"
            }
        ]
    })

    print(resp["structured_response"])


# ============================================================
# 12. 主程序入口
# ============================================================

if __name__ == '__main__':

    # 1. Pydantic 结构化输出
    resp = deepseek_llm.with_structured_output(Movie).invoke(
        "描述一下电影《复仇者联盟4》"
    )
    print(f'电影介绍返回: {resp}')

    # 2. TypedDict 结构化输出
    car = deepseek_llm.with_structured_output(Car).invoke(
        "帮我介绍一下小米YU7"
    )
    print(f'汽车介绍返回： {car}')

    # 3. JSON Schema 结构化输出
    computer = deepseek_llm.with_structured_output(project_schema).invoke(
        "详细介绍一下苹果MacBookPro16寸2026年的笔记本电脑"
    )
    print(f'电脑介绍：{computer}')

    # 4. JsonOutputParser（保留你的原函数）
    output_parser(deepseek_llm)

    # PydanticOutputParser
    pydantic_output_parser(deepseek_llm)

    # 查看原始模型输出
    structured_output_raw(deepseek_llm)

    output_fixing_parser(deepseek_llm)

    # Agent ToolStrategy：需确认思考模式与工具调用兼容
    agent_tool_strategy(deepseek_llm)

    # Agent 自动选择策略
    agent_auto_strategy(deepseek_llm)

    # Agent 自定义错误处理
    agent_error_handling(deepseek_llm)

    # ProviderStrategy：
    # 需要传入支持原生结构化输出的其他模型实例，
    # 不直接使用当前 deepseek_llm。
    # agent_provider_strategy(provider_model)
