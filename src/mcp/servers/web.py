"""
MCP Web 服务器 (Web Server)

提供网络请求能力的 MCP 服务器实现。

提供的工具：
  - fetch_url(url, method): 发送 HTTP 请求
  - search_web(query): 网页搜索（当前为存根实现）

提供的资源：
  - https_resource(uri): 通过 https:// URI 获取网页内容

安全特性：
  - allowed_domains: 域名白名单（预留，当前未强制执行）
"""
import httpx
from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="web", version="1.0")
class WebServer:
    """Web MCP 服务器。

    Args:
        allowed_domains: 允许访问的域名列表（安全限制，预留）
    """

    def __init__(self, allowed_domains: list[str] | None = None):
        self._allowed = allowed_domains

    @tool(description="Fetch content from a URL")
    async def fetch_url(self, url: str, method: str = "GET") -> dict:
        """发送 HTTP 请求并返回响应。"""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url)
            return {"status": resp.status_code, "body": resp.text}

    @tool(description="Search the web (stub)")
    async def search_web(self, query: str) -> list[dict]:
        """网页搜索（当前为存根实现，返回模拟数据）。"""
        return [{"title": f"Result for: {query}", "url": "https://example.com"}]

    @resource(uri_scheme="https")
    async def https_resource(self, uri: str) -> str:
        """通过 https:// URI 获取网页内容。"""
        result = await self.fetch_url(uri)
        return result["body"]
