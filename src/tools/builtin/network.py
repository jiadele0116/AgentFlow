"""Network tools."""
import httpx
from typing import Any
from src.tools.registry import ToolRegistry


def register_network_tools(registry: ToolRegistry):
    @registry.register(
        name="fetch_url", description="Fetch content from a URL",
        parameters={"type": "object", "properties": {
            "url": {"type": "string"}, "method": {"type": "string", "default": "GET"},
            "headers": {"type": "object"}, "body": {"type": "string"},
        }, "required": ["url"]},
        timeout=30,
    )
    async def fetch_url(url: str, method: str = "GET", headers: dict[str, str] | None = None, body: str | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method=method, url=url, headers=headers or {}, content=body)
            return {"status": resp.status_code, "body": resp.text, "headers": dict(resp.headers)}
