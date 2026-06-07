"""
多Agent系统入口模块 (Multi-Agent System Entry Point)

系统启动流程：
1. 加载配置 (config.yaml)
2. 初始化日志系统
3. 创建 LangChain LLM 实例
4. 注册内置工具并转换为 LangChain 格式
5. 初始化记忆系统（短期 + 长期）
6. 构建 LangGraph 多 Agent 协作图
7. 启动 Web 服务 (FastAPI + WebSocket)

使用示例：
    python -m src.main
"""
import asyncio
from pathlib import Path

from src.core.config import load_config
from src.core.logging import setup_logging, get_logger
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_all_builtin_tools


async def main():
    """系统主入口：初始化所有组件并启动 Web 服务。"""
    # 步骤1：加载配置
    config_path = Path(__file__).parent.parent / "config.yaml"
    settings = load_config(config_path)

    # 步骤2：初始化日志
    setup_logging(level=settings.system.log_level)
    logger = get_logger("main")

    logger.info(f"Starting {settings.system.name} v{settings.system.version}")

    # 步骤3：创建 LangChain LLM
    from src.llm.langchain_llm import create_langchain_llm
    langchain_llm = create_langchain_llm(settings)
    logger.info(f"LangChain LLM: {settings.llm.provider} / {settings.llm.model}")

    # 步骤4：注册内置工具并转换为 LangChain 格式
    registry = ToolRegistry()
    register_all_builtin_tools(registry, workspace_dir="./data/workspace")
    logger.info(f"Registered {len(registry.list_tools())} tools")

    from src.tools.langchain_tools import convert_to_langchain_tools
    lc_tools = convert_to_langchain_tools(registry)
    logger.info(f"Converted {len(lc_tools)} tools to LangChain format")

    # 步骤5：初始化记忆系统
    from src.memory.short_term import ShortTermMemory
    from src.memory.long_term import LongTermMemory
    from src.memory.manager import MemoryManager

    short_term = ShortTermMemory(max_turns=settings.agents.memory.short_term_max)
    memory = MemoryManager(short_term=short_term, long_term=None, llm=langchain_llm)
    logger.info("Memory system initialized (short-term + LLM summarization)")

    # 步骤6：构建 LangGraph 多 Agent 协作图
    from src.agents.langgraph_multi import create_multi_agent_graph
    graph = create_multi_agent_graph(langchain_llm, lc_tools)
    logger.info("LangGraph multi-agent graph compiled")

    # 步骤7：启动 Web 服务
    import uvicorn
    from src.api.server import create_app

    app = create_app(graph=graph, memory=memory)
    config = uvicorn.Config(app, host="0.0.0.0", port=8880, log_level="info")
    server = uvicorn.Server(config)
    logger.info("Web server starting at http://localhost:8880")
    logger.info("System ready. Press Ctrl+C to exit.")

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
