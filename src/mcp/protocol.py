"""
MCP 协议消息定义模块 (MCP Protocol Messages)

定义 MCP (Model Context Protocol) 通信中使用的消息数据结构。

MCP 是 Agent 与外部工具/数据源通信的标准化协议，类似于 LSP (Language Server Protocol)
为编辑器和语言服务器提供统一接口。

消息结构：
  MCPRequest  - 客户端发送的请求
  ├── method: 请求方法（"tools/list", "tools/call", "resources/list"）
  ├── params: 请求参数
  └── request_id: 请求唯一标识（用于关联响应）

  MCPResponse - 服务器返回的响应
  ├── request_id: 对应的请求ID
  ├── result: 成功时的结果数据
  └── error: 失败时的错误信息

  MCPError - 错误信息
  ├── code: 错误码（遵循 JSON-RPC 错误码规范）
  ├── message: 错误描述
  └── data: 额外错误数据（可选）

标准错误码（JSON-RPC）：
  -32600: Invalid Request
  -32601: Method Not Found
  -32602: Invalid Params
  -32603: Internal Error
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPError:
    """MCP 错误信息。

    Attributes:
        code: 错误码（整数，遵循 JSON-RPC 规范）
        message: 人类可读的错误描述
        data: 额外的错误上下文数据（可选）
    """
    code: int
    message: str
    data: dict[str, Any] | None = None


@dataclass
class MCPRequest:
    """MCP 请求消息。

    Attributes:
        method: 请求方法名（如 "tools/list", "tools/call"）
        params: 请求参数字典
        request_id: 请求唯一标识（用于关联响应）
    """
    method: str
    params: dict[str, Any]
    request_id: str


@dataclass
class MCPResponse:
    """MCP 响应消息。

    Attributes:
        request_id: 对应的请求ID
        result: 成功时的结果数据（method 依赖）
        error: 失败时的错误信息（成功时为 None）
    """
    request_id: str
    result: dict[str, Any] | None = None
    error: MCPError | None = None
