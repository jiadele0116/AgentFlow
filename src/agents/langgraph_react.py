"""LangGraph ReAct Agent 模块。

使用 LangGraph 内置的 create_react_agent 创建单 Agent 工具调用循环。
ReAct 模式：LLM 推理 → 工具调用 → 观察结果 → 继续推理 → 直到得出最终答案。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from src.core.logging import get_logger
from src.core.prompts import REACT_PROMPT

logger = get_logger("agents.react")


def create_react_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str | None = None,
) -> CompiledStateGraph:
    """创建 LangGraph ReAct Agent。

    Args:
        llm: LangChain ChatModel 实例
        tools: LangChain Tool 列表
        system_prompt: 系统提示词（可选）

    Returns:
        编译好的 LangGraph StateGraph，可直接调用
    """
    if system_prompt is None:
        system_prompt = REACT_PROMPT

    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    logger.info(f"ReAct Agent 已创建，包含 {len(tools)} 个工具")
    return graph
