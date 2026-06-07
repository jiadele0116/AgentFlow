"""
MCP 客户端模块 (MCP Client)

提供与 MCP 服务器通信的客户端接口。客户端维护已注册的服务器列表，
通过 handle_request() 方法与服务器交互。

使用示例：
    client = MCPClient()
    client.register_server("filesystem", FileSystemServer("/workspace"))

    # 列出工具
    tools = await client.list_tools("filesystem")

    # 调用工具
    content = await client.call_tool("filesystem", "read_file", {"path": "test.txt"})
"""
import uuid
from typing import Any
from src.mcp.protocol import MCPRequest


class MCPClient:
    """MCP 客户端：管理多个 MCP 服务器的通信。

    支持同时连接多个 MCP 服务器，通过 server_name 区分。
    """

    def __init__(self):
        # 已注册的服务器：name → server instance
        self._servers: dict[str, Any] = {}

    def register_server(self, name: str, server: Any):
        """注册一个 MCP 服务器。

        Args:
            name: 服务器名称（用于后续调用）
            server: MCP 服务器实例（需有 handle_request 方法）
        """
        self._servers[name] = server

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用指定服务器上的工具。

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 服务器未找到
            RuntimeError: MCP 调用失败
        """
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")
        request = MCPRequest(method="tools/call", params={"name": tool_name, "arguments": arguments}, request_id=str(uuid.uuid4()))
        response = await server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")
        return response.result.get("content") if response.result else None

    async def list_tools(self, server_name: str) -> list[dict]:
        """列出指定服务器上的所有工具。

        Args:
            server_name: 服务器名称

        Returns:
            工具描述列表
        """
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")
        request = MCPRequest(method="tools/list", params={}, request_id=str(uuid.uuid4()))
        response = await server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")
        return response.result.get("tools", []) if response.result else []
