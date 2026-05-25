"""File operation tools."""
import fnmatch
import os
from pathlib import Path
from src.tools.registry import ToolRegistry


def register_file_tools(registry: ToolRegistry):
    @registry.register(
        name="read_file", description="Read the contents of a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    async def read_file(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @registry.register(
        name="write_file", description="Write content to a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    )
    async def write_file(path: str, content: str) -> bool:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @registry.register(
        name="search_files", description="Search for files matching a glob pattern",
        parameters={"type": "object", "properties": {"pattern": {"type": "string"}, "directory": {"type": "string"}}, "required": ["pattern", "directory"]},
    )
    async def search_files(pattern: str, directory: str) -> list[str]:
        results = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)
