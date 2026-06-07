"""
MCP 服务器基础模块 (MCP Server Base)

提供 MCP 服务器的核心实现，包括装饰器和请求处理逻辑。

核心概念：
1. @tool 装饰器：将类方法标记为 MCP 工具
   - 自动从函数签名推断参数 JSON Schema
   - 可手动指定 description 和 parameters

2. @resource 装饰器：将类方法标记为 MCP 资源
   - 资源通过 URI scheme 访问（如 file://, https://）

3. @mcp_server 类装饰器：将普通类转换为 MCP 服务器
   - 扫描类中的 @tool 和 @resource 方法
   - 自动生成 handle_request() 方法处理 MCP 请求

支持的请求方法：
  - tools/list: 列出所有可用工具
  - tools/call: 调用指定工具
  - resources/list: 列出所有可用资源

使用示例：
    @mcp_server(name="my_server", version="1.0")
    class MyServer:
        @tool(description="Add two numbers")
        async def add(self, a: int, b: int) -> int:
            return a + b

    server = MyServer()
    request = MCPRequest(method="tools/call", params={"name": "add", "arguments": {"a": 1, "b": 2}}, request_id="1")
    response = await server.handle_request(request)
"""
import inspect
from typing import Any, Callable
from src.mcp.protocol import MCPRequest, MCPResponse, MCPError


def tool(description: str = "", parameters: dict | None = None):
    """MCP 工具装饰器：将方法标记为可被 Agent 调用的工具。

    装饰器会在函数上添加元信息属性：
    - _mcp_tool: True（标记为工具）
    - _mcp_tool_description: 工具描述
    - _mcp_tool_parameters: JSON Schema 参数定义

    Args:
        description: 工具功能描述
        parameters: JSON Schema 参数定义（None 则自动推断）
    """
    def decorator(func):
        func._mcp_tool = True
        func._mcp_tool_description = description
        func._mcp_tool_parameters = parameters or _infer_parameters(func)
        return func
    return decorator


def resource(uri_scheme: str = ""):
    """MCP 资源装饰器：将方法标记为可通过 URI 访问的资源。

    Args:
        uri_scheme: URI 协议方案（如 "file", "https"）
    """
    def decorator(func):
        func._mcp_resource = True
        func._mcp_uri_scheme = uri_scheme
        return func
    return decorator


def _infer_parameters(func: Callable) -> dict[str, Any]:
    """从函数签名自动推断 JSON Schema 参数定义。

    支持的类型映射：
    - str → {"type": "string"}
    - int → {"type": "integer"}
    - float → {"type": "number"}
    - bool → {"type": "boolean"}
    - 其他 → {"type": "string"}（默认）

    无默认值的参数标记为 required。

    Args:
        func: 待推断的函数

    Returns:
        JSON Schema 格式的参数定义
    """
    sig = inspect.signature(func)
    properties, required = {}, []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        prop: dict[str, Any] = {}
        if param.annotation == str: prop["type"] = "string"
        elif param.annotation == int: prop["type"] = "integer"
        elif param.annotation == float: prop["type"] = "number"
        elif param.annotation == bool: prop["type"] = "boolean"
        else: prop["type"] = "string"
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def mcp_server(name: str, version: str = "1.0"):
    """MCP 服务器类装饰器：将普通类转换为 MCP 服务器。

    装饰后的类会获得：
    - _mcp_name, _mcp_version: 服务器元信息
    - _tools: 工具注册表（name → handler/description/parameters）
    - _resources: 资源注册表（name → handler/uri_scheme）
    - handle_request(): MCP 请求处理方法

    Args:
        name: 服务器名称
        version: 服务器版本
    """
    def decorator(cls):
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mcp_name, self._mcp_version = name, version
            self._tools, self._resources = {}, {}
            # 扫描类方法，注册 @tool 和 @resource
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if attr and getattr(attr, "_mcp_tool", False):
                    self._tools[attr_name] = {"handler": attr, "description": attr._mcp_tool_description, "parameters": attr._mcp_tool_parameters}
                if attr and getattr(attr, "_mcp_resource", False):
                    self._resources[attr_name] = {"handler": attr, "uri_scheme": attr._mcp_uri_scheme}
        cls.__init__ = new_init

        async def handle_request(self, request: MCPRequest) -> MCPResponse:
            """处理 MCP 请求：根据 method 分发到对应的处理器。"""
            if request.method == "tools/list":
                # 列出所有工具
                return MCPResponse(request_id=request.request_id, result={"tools": [
                    {"name": n, "description": i["description"], "parameters": i["parameters"]} for n, i in self._tools.items()]})
            elif request.method == "tools/call":
                # 调用指定工具
                tn = request.params.get("name")
                if tn not in self._tools:
                    return MCPResponse(request_id=request.request_id, error=MCPError(code=-32601, message=f"Tool not found: {tn}"))
                try:
                    result = await self._tools[tn]["handler"](**request.params.get("arguments", {}))
                    return MCPResponse(request_id=request.request_id, result={"content": result})
                except Exception as e:
                    return MCPResponse(request_id=request.request_id, error=MCPError(code=-32603, message=str(e)))
            elif request.method == "resources/list":
                # 列出所有资源
                return MCPResponse(request_id=request.request_id, result={"resources": [
                    {"name": n, "uri_scheme": i["uri_scheme"]} for n, i in self._resources.items()]})
            else:
                # 未知方法
                return MCPResponse(request_id=request.request_id, error=MCPError(code=-32601, message=f"Method not found: {request.method}"))
        cls.handle_request = handle_request
        return cls
    return decorator
