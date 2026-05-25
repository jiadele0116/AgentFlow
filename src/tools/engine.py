"""Tool execution engine with timeout, retry, and audit."""
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.tools.registry import ToolRegistry


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0


class ToolEngine:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        start = time.monotonic()
        last_error = ""
        for attempt in range(1 + tool.retry):
            try:
                result = await asyncio.wait_for(tool.handler(**params), timeout=tool.timeout)
                duration = (time.monotonic() - start) * 1000
                return ToolResult(success=True, data=result, duration_ms=duration)
            except asyncio.TimeoutError:
                duration = (time.monotonic() - start) * 1000
                return ToolResult(success=False, error=f"Tool '{name}' timed out after {tool.timeout}s", duration_ms=duration)
            except Exception as e:
                last_error = str(e)
                if attempt < tool.retry:
                    await asyncio.sleep(0.1 * (2 ** attempt))

        duration = (time.monotonic() - start) * 1000
        return ToolResult(success=False, error=last_error, duration_ms=duration)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._registry.list_tools()
