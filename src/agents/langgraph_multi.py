"""LangGraph 多 Agent 协作图模块。

构建 Planner → Executor → Critic 的多 Agent 工作流。

流程：
  用户输入 → Planner(分解任务) → Executor(执行，可调用工具) → Critic(审查)
                                                ↑                      │
                                                └──── 不通过则修订 ────┘
"""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.core.logging import get_logger
from src.core.prompts import (
    get_planner_prompt, get_executor_prompt,
    EXECUTOR_TOOL_RESULT_PROMPT, CRITIC_PROMPT,
)

logger = get_logger("agents.multi")


# ============ 状态定义 ============

class MultiAgentState(TypedDict):
    """多 Agent 协作图的状态。"""
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史（自动追加）
    task: str                       # 原始用户任务
    plan: str                       # Planner 生成的计划
    execution_result: str           # Executor 的执行结果
    review_status: str              # "pending" / "approved" / "rejected"
    iterations: int                 # 修订轮次计数
    context: str                    # 记忆上下文（短期历史 + 长期召回）


# ============ 节点函数 ============

async def planner_node(state: MultiAgentState, llm: BaseChatModel) -> dict:
    """Planner 节点：分析任务，生成执行计划。"""
    logger.info("规划器: 正在分析任务")
    task = state["task"]
    context = state.get("context", "")

    system_prompt = get_planner_prompt(context)

    # 只用当前任务，不重复注入历史（context 已包含）
    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=task))

    plan = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            plan += chunk.content

    logger.info(f"规划器: 计划已生成 ({len(plan)} 字符)")
    return {
        "plan": plan,
        "messages": [SystemMessage(content=f"[计划]\n{plan}")],
    }


async def executor_node(state: MultiAgentState, llm: BaseChatModel, tools: list[BaseTool]) -> dict:
    """Executor 节点：根据计划执行任务，可调用工具。"""
    logger.info("执行器: 正在执行任务")
    task = state["task"]
    plan = state.get("plan", "")
    context = state.get("context", "")
    is_revision = state.get("review_status") == "rejected"

    system_prompt = get_executor_prompt(context, revision=is_revision)

    # 只发任务和计划，不重复历史
    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=f"任务：{task}\n计划：{plan}"))

    # 带工具的 LLM 调用（流式）
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    response_content = ""
    tool_calls = None
    async for chunk in llm_with_tools.astream(messages):
        if chunk.content:
            response_content += chunk.content
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            if tool_calls is None:
                tool_calls = chunk.tool_calls
            else:
                for tc in chunk.tool_calls:
                    tool_calls.append(tc)

    # 如果 LLM 请求了工具调用，执行工具
    if tool_calls:
        logger.info(f"执行器: 请求调用 {len(tool_calls)} 个工具")
        tool_results = []
        for tc in tool_calls:
            for tool in tools:
                if tool.name == tc["name"]:
                    try:
                        result = await tool.ainvoke(tc["args"])
                        tool_results.append(f"[{tc['name']}] → {result}")
                        logger.info(f"执行器: 工具 {tc['name']} 执行成功")
                    except Exception as e:
                        tool_results.append(f"[{tc['name']}] → 错误: {e}")
                        logger.warning(f"执行器: 工具 {tc['name']} 执行失败: {e}")
                    break

        # 工具结果反馈给 LLM（精简 prompt）
        tool_context = "\n".join(tool_results)
        result = ""
        async for chunk in llm.astream([
            SystemMessage(content=EXECUTOR_TOOL_RESULT_PROMPT),
            HumanMessage(content=f"任务：{task}\n工具结果：\n{tool_context}"),
        ]):
            if chunk.content:
                result += chunk.content
    else:
        result = response_content

    logger.info(f"执行器: 结果已生成 ({len(result)} 字符)")
    return {
        "execution_result": result,
        "messages": [SystemMessage(content=f"[执行结果]\n{result}")],
    }


async def critic_node(state: MultiAgentState, llm: BaseChatModel) -> dict:
    """Critic 节点：审查执行结果质量。"""
    logger.info("审查器: 正在审查结果")
    task = state["task"]
    result = state.get("execution_result", "")

    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"任务：{task}\n结果：{result}"),
    ]

    review = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            review += chunk.content
    review = review.strip()

    if review.upper().startswith("APPROVED"):
        status = "approved"
        logger.info("审查器: 通过")
    else:
        status = "rejected"
        logger.info(f"审查器: 拒绝 - {review[:50]}")

    return {
        "review_status": status,
        "messages": [SystemMessage(content=f"[审查] {review}")],
        "iterations": state.get("iterations", 0) + 1,
    }


# ============ 条件路由 ============

def should_continue(state: MultiAgentState) -> str:
    """决定是继续修订还是结束。"""
    # 审查通过 → 结束
    if state.get("review_status") == "approved":
        return "end"
    # 超过 2 轮修订 → 强制结束
    if state.get("iterations", 0) >= 2:
        logger.info("已达最大修订轮次，结束")
        return "end"
    # 未审查 → 去审查
    if state.get("review_status") == "pending":
        return "critic"
    # 被拒绝 → 重新执行
    return "executor"


# ============ 图构建 ============

def create_multi_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
) -> StateGraph:
    """创建多 Agent 协作图。

    Args:
        llm: LangChain ChatModel
        tools: LangChain Tool 列表

    Returns:
        编译好的 LangGraph StateGraph
    """
    graph = StateGraph(MultiAgentState)

    # 添加节点（用 async wrapper 绑定 llm 和 tools 参数）
    async def _planner(state):
        return await planner_node(state, llm)

    async def _executor(state):
        return await executor_node(state, llm, tools)

    async def _critic(state):
        return await critic_node(state, llm)

    graph.add_node("planner", _planner)
    graph.add_node("executor", _executor)
    graph.add_node("critic", _critic)

    # 定义边
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    graph.add_conditional_edges("critic", should_continue, {
        "end": END,
        "executor": "executor",
    })

    compiled = graph.compile()
    logger.info("多Agent协作图编译完成")
    return compiled
