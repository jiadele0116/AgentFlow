"""Tests for MCP server."""
import pytest
from src.mcp.server import mcp_server, tool
from src.mcp.protocol import MCPRequest


@pytest.mark.asyncio
async def test_mcp_server_tools_list():
    @mcp_server(name="test", version="1.0")
    class TestServer:
        @tool(description="Echo")
        async def echo(self, text: str) -> str: return text

    server = TestServer()
    resp = await server.handle_request(MCPRequest(method="tools/list", params={}, request_id="1"))
    assert resp.result is not None
    assert len(resp.result["tools"]) == 1
    assert resp.result["tools"][0]["name"] == "echo"


@pytest.mark.asyncio
async def test_mcp_server_tool_call():
    @mcp_server(name="calc", version="1.0")
    class CalcServer:
        @tool(description="Add")
        async def add(self, a: int, b: int) -> int: return a + b

    server = CalcServer()
    resp = await server.handle_request(MCPRequest(method="tools/call", params={"name": "add", "arguments": {"a": 2, "b": 3}}, request_id="2"))
    assert resp.result["content"] == 5


@pytest.mark.asyncio
async def test_mcp_server_unknown_method():
    @mcp_server(name="test", version="1.0")
    class TestServer: pass

    server = TestServer()
    resp = await server.handle_request(MCPRequest(method="unknown", params={}, request_id="3"))
    assert resp.error is not None
    assert resp.error.code == -32601
