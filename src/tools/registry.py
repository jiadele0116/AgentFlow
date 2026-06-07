"""
工具注册表模块 (Tool Registry)

提供装饰器风格的工具注册机制。每个工具通过 @register 装饰器注册到全局注册表，
包含名称、描述、JSON Schema 参数定义和异步处理函数。

工具定义结构：
  Tool
  ├── name: 工具唯一标识符
  ├── description: 工具功能描述（LLM 用于决定何时调用）
  ├── parameters: JSON Schema 格式的参数定义（LLM 用于生成正确参数）
  ├── handler: 异步执行函数 (async def handler(**params) -> Any)
  ├── timeout: 执行超时（秒），默认 30
  ├── retry: 失败重试次数，默认 0
  └── requires_confirmation: 是否需要用户确认（危险操作标记）

使用示例：
    registry = ToolRegistry()

    @registry.register(
        name="read_file",
        description="Read content from a file path",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        }
    )
    async def read_file(path: str) -> str:
        with open(path) as f:
            return f.read()
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    """工具定义数据类：包含工具的完整元信息和执行逻辑。

    Attributes:
        name: 工具唯一标识符，用于注册和调用
        description: 工具功能描述，LLM 会根据此描述决定何时使用该工具
        parameters: JSON Schema 格式的参数定义，约束工具接受的输入
        handler: 异步执行函数，接收参数字典返回执行结果
        timeout: 执行超时时间（秒），超时后抛出 TimeoutError
        retry: 失败重试次数（0 表示不重试）
        requires_confirmation: 危险操作标记，True 时需要用户确认后才执行
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    timeout: int = 30
    retry: int = 0
    requires_confirmation: bool = False


class ToolRegistry:
    """工具注册表：管理所有已注册工具的生命周期。

    提供注册、查询、列举和注销功能。
    注册表是全局单例模式，通常在系统启动时初始化一次。
    """

    def __init__(self):
        # 内部存储：工具名 → Tool 对象
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any],
                 timeout: int = 30, retry: int = 0, requires_confirmation: bool = False):
        """注册工具装饰器。

        将异步函数注册为可被 LLM 调用的工具。装饰器会自动创建 Tool 对象
        并存入注册表，同时保留原函数的可调用性。

        Args:
            name: 工具唯一标识符
            description: 功能描述（LLM 可见）
            parameters: JSON Schema 参数定义
            timeout: 执行超时（秒）
            retry: 重试次数
            requires_confirmation: 是否需要确认

        Returns:
            装饰器函数
        """
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            self._tools[name] = Tool(
                name=name, description=description, parameters=parameters,
                handler=func, timeout=timeout, retry=retry, requires_confirmation=requires_confirmation,
            )
            return func
        return decorator

    def get(self, name: str) -> Tool | None:
        """根据名称获取工具定义。

        Args:
            name: 工具名称

        Returns:
            Tool 对象，未找到返回 None
        """
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """返回所有已注册工具的名称列表。"""
        return list(self._tools.keys())

    def list_tools(self) -> list[dict[str, Any]]:
        """返回所有工具的 JSON Schema 描述列表。

        该列表直接传递给 LLM 的 tools 参数，让 LLM 知道有哪些工具可用。
        每个工具包含 name、description 和 input_schema 三个字段。

        Returns:
            工具描述列表，格式符合 Anthropic/OpenAI 的 tools 参数规范
        """
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in self._tools.values()]

    def unregister(self, name: str):
        """注销指定工具。

        Args:
            name: 要注销的工具名称
        """
        self._tools.pop(name, None)
