"""MCP protocol message definitions."""
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPError:
    code: int
    message: str
    data: dict[str, Any] | None = None


@dataclass
class MCPRequest:
    method: str
    params: dict[str, Any]
    request_id: str


@dataclass
class MCPResponse:
    request_id: str
    result: dict[str, Any] | None = None
    error: MCPError | None = None
