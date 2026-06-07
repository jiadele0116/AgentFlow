"""
网络请求工具模块 (Network Tools)

提供 HTTP 请求能力，供 Agent 访问外部 API 和网页。

注册的工具：
  - fetch_url(url, method, headers, body): 发送 HTTP 请求并返回响应

使用 httpx 异步客户端，支持 GET/POST/PUT/DELETE 等方法。
"""
import httpx
from typing import Any
from src.tools.registry import ToolRegistry


def register_network_tools(registry: ToolRegistry):
    """注册网络请求工具到注册表。"""

    @registry.register(
        name="fetch_url", description="Fetch content from a URL",
        parameters={"type": "object", "properties": {
            "url": {"type": "string"}, "method": {"type": "string", "default": "GET"},
            "headers": {"type": "object"}, "body": {"type": "string"},
        }, "required": ["url"]},
        timeout=30,
    )
    async def fetch_url(url: str, method: str = "GET", headers: dict[str, str] | None = None, body: str | None = None) -> dict[str, Any]:
        """发送 HTTP 请求并返回响应。

        Args:
            url: 请求 URL
            method: HTTP 方法（默认 GET）
            headers: 请求头字典（可选）
            body: 请求体字符串（可选，用于 POST/PUT）

        Returns:
            包含 status（状态码）、body（响应体）、headers（响应头）的字典
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method=method, url=url, headers=headers or {}, content=body)
            return {"status": resp.status_code, "body": resp.text, "headers": dict(resp.headers)}
