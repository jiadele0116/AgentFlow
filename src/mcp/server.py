"""MCP server base with @tool and @resource decorators."""
import inspect
from typing import Any, Callable
from src.mcp.protocol import MCPRequest, MCPResponse, MCPError


def tool(description: str = "", parameters: dict | None = None):
    def decorator(func):
        func._mcp_tool = True
        func._mcp_tool_description = description
        func._mcp_tool_parameters = parameters or _infer_parameters(func)
        return func
    return decorator


def resource(uri_scheme: str = ""):
    def decorator(func):
        func._mcp_resource = True
        func._mcp_uri_scheme = uri_scheme
        return func
    return decorator


def _infer_parameters(func: Callable) -> dict[str, Any]:
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
    def decorator(cls):
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mcp_name, self._mcp_version = name, version
            self._tools, self._resources = {}, {}
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if attr and getattr(attr, "_mcp_tool", False):
                    self._tools[attr_name] = {"handler": attr, "description": attr._mcp_tool_description, "parameters": attr._mcp_tool_parameters}
                if attr and getattr(attr, "_mcp_resource", False):
                    self._resources[attr_name] = {"handler": attr, "uri_scheme": attr._mcp_uri_scheme}
        cls.__init__ = new_init

        async def handle_request(self, request: MCPRequest) -> MCPResponse:
            if request.method == "tools/list":
                return MCPResponse(request_id=request.request_id, result={"tools": [
                    {"name": n, "description": i["description"], "parameters": i["parameters"]} for n, i in self._tools.items()]})
            elif request.method == "tools/call":
                tn = request.params.get("name")
                if tn not in self._tools:
                    return MCPResponse(request_id=request.request_id, error=MCPError(code=-32601, message=f"Tool not found: {tn}"))
                try:
                    result = await self._tools[tn]["handler"](**request.params.get("arguments", {}))
                    return MCPResponse(request_id=request.request_id, result={"content": result})
                except Exception as e:
                    return MCPResponse(request_id=request.request_id, error=MCPError(code=-32603, message=str(e)))
            elif request.method == "resources/list":
                return MCPResponse(request_id=request.request_id, result={"resources": [
                    {"name": n, "uri_scheme": i["uri_scheme"]} for n, i in self._resources.items()]})
            else:
                return MCPResponse(request_id=request.request_id, error=MCPError(code=-32601, message=f"Method not found: {request.method}"))
        cls.handle_request = handle_request
        return cls
    return decorator
