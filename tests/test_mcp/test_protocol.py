"""Tests for MCP protocol."""
import pytest
from src.mcp.protocol import MCPRequest, MCPResponse, MCPError


def test_mcp_request():
    req = MCPRequest(method="tools/call", params={"name": "read_file"}, request_id="1")
    assert req.method == "tools/call"
    assert req.request_id == "1"


def test_mcp_response_success():
    resp = MCPResponse(request_id="1", result={"content": "data"})
    assert resp.result is not None
    assert resp.error is None


def test_mcp_response_error():
    resp = MCPResponse(request_id="1", error=MCPError(code=-32600, message="Bad"))
    assert resp.error.code == -32600
