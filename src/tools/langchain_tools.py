"""LangChain 工具转换模块。

将项目 ToolRegistry 中注册的工具转换为 LangChain StructuredTool。
"""
import asyncio
from typing import Any

from langchain_core.tools import StructuredTool, BaseTool

from src.tools.registry import ToolRegistry
from src.core.logging import get_logger

logger = get_logger("tools.langchain")


def _wrap_handler(handler):
    """将异步 handler 包装为同步可调用（LangChain 需要）。

    StructuredTool.from_function 的 coroutine 参数接受异步函数，
    但 func 参数需要同步函数。这里提供一个同步包装。
    """
    def sync_wrapper(**kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(handler(**kwargs))
        finally:
            loop.close()
    return sync_wrapper


def convert_to_langchain_tools(registry: ToolRegistry) -> list[BaseTool]:
    """将 ToolRegistry 中的工具转换为 LangChain Tool 列表。

    Args:
        registry: 项目工具注册表

    Returns:
        LangChain BaseTool 列表
    """
    tools: list[BaseTool] = []

    for name in registry.list_names():
        tool = registry.get(name)
        if tool is None:
            continue

        try:
            lc_tool = StructuredTool.from_function(
                coroutine=tool.handler,
                name=tool.name,
                description=tool.description,
                return_direct=False,
            )
            tools.append(lc_tool)
            logger.debug(f"Converted tool: {tool.name}")
        except Exception as e:
            logger.warning(f"Failed to convert tool {tool.name}: {e}")

    logger.info(f"Converted {len(tools)} tools to LangChain format")
    return tools
