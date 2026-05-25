"""Multi-Agent System entry point."""
import asyncio
from pathlib import Path

from src.core.config import load_config
from src.core.logging import setup_logging, get_logger
from src.core.events import EventBus
from src.tools.registry import ToolRegistry
from src.tools.engine import ToolEngine
from src.tools.builtin import register_all_builtin_tools
from src.agents.memory import ShortTermMemory


async def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    settings = load_config(config_path)
    setup_logging(level=settings.system.log_level)
    logger = get_logger("main")

    logger.info(f"Starting {settings.system.name} v{settings.system.version}")

    # Initialize tools
    registry = ToolRegistry()
    register_all_builtin_tools(registry, workspace_dir="./data/workspace")
    engine = ToolEngine(registry)
    logger.info(f"Registered {len(registry.list_tools())} tools")

    # Initialize event bus
    bus = EventBus()
    await bus.start()

    logger.info("System ready. Press Ctrl+C to exit.")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(main())
