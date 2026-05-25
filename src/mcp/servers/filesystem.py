"""MCP FileSystemServer."""
import fnmatch
import os
from pathlib import Path
from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="filesystem", version="1.0")
class FileSystemServer:
    def __init__(self, root_dir: str = "."):
        self._root = Path(root_dir)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self._root / p
        return p.resolve()

    @tool(description="Read file contents")
    async def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @tool(description="Write content to a file")
    async def write_file(self, path: str, content: str) -> bool:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @tool(description="List directory contents")
    async def list_dir(self, path: str = ".") -> list[str]:
        p = self._resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [str(item.name) for item in p.iterdir()]

    @tool(description="Search for files matching a pattern")
    async def search_files(self, pattern: str, directory: str = ".") -> list[str]:
        d = self._resolve(directory)
        results = []
        for root, dirs, files in os.walk(d):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)

    @resource(uri_scheme="file")
    async def file_resource(self, uri: str) -> str:
        return await self.read_file(uri.replace("file://", ""))
