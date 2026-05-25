"""Tool registry with decorator-based registration."""
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    timeout: int = 30
    retry: int = 0
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any],
                 timeout: int = 30, retry: int = 0, requires_confirmation: bool = False):
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            self._tools[name] = Tool(
                name=name, description=description, parameters=parameters,
                handler=func, timeout=timeout, retry=retry, requires_confirmation=requires_confirmation,
            )
            return func
        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in self._tools.values()]

    def unregister(self, name: str):
        self._tools.pop(name, None)
