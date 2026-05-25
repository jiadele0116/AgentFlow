"""MCP WebServer."""
import httpx
from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="web", version="1.0")
class WebServer:
    def __init__(self, allowed_domains: list[str] | None = None):
        self._allowed = allowed_domains

    @tool(description="Fetch content from a URL")
    async def fetch_url(self, url: str, method: str = "GET") -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url)
            return {"status": resp.status_code, "body": resp.text}

    @tool(description="Search the web (stub)")
    async def search_web(self, query: str) -> list[dict]:
        return [{"title": f"Result for: {query}", "url": "https://example.com"}]

    @resource(uri_scheme="https")
    async def https_resource(self, uri: str) -> str:
        result = await self.fetch_url(uri)
        return result["body"]
