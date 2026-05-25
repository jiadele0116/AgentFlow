"""MCP client for communicating with MCP servers."""
import uuid
from typing import Any
from src.mcp.protocol import MCPRequest


class MCPClient:
    def __init__(self):
        self._servers: dict[str, Any] = {}

    def register_server(self, name: str, server: Any):
        self._servers[name] = server

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")
        request = MCPRequest(method="tools/call", params={"name": tool_name, "arguments": arguments}, request_id=str(uuid.uuid4()))
        response = await server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")
        return response.result.get("content") if response.result else None

    async def list_tools(self, server_name: str) -> list[dict]:
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")
        request = MCPRequest(method="tools/list", params={}, request_id=str(uuid.uuid4()))
        response = await server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")
        return response.result.get("tools", []) if response.result else []
