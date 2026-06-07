"""
文件操作工具模块 (File Operation Tools)

提供文件系统的基本操作能力，供 Agent 读写和搜索文件。

注册的工具：
  - read_file(path): 读取文件内容，返回 UTF-8 文本
  - write_file(path, content): 写入文件，自动创建父目录
  - search_files(pattern, directory): 按 glob 模式递归搜索文件
"""
import fnmatch
import os
from pathlib import Path
from src.tools.registry import ToolRegistry


def register_file_tools(registry: ToolRegistry):
    """注册文件操作工具到注册表。"""

    @registry.register(
        name="read_file", description="Read the contents of a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    async def read_file(path: str) -> str:
        """读取指定路径的文件内容。

        Args:
            path: 文件路径（绝对或相对路径）

        Returns:
            文件的 UTF-8 文本内容

        Raises:
            FileNotFoundError: 文件不存在
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @registry.register(
        name="write_file", description="Write content to a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    )
    async def write_file(path: str, content: str) -> bool:
        """写入内容到指定路径的文件。

        如果父目录不存在会自动创建。

        Args:
            path: 目标文件路径
            content: 要写入的文本内容

        Returns:
            True 表示写入成功
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @registry.register(
        name="search_files", description="Search for files matching a glob pattern",
        parameters={"type": "object", "properties": {"pattern": {"type": "string"}, "directory": {"type": "string"}}, "required": ["pattern", "directory"]},
    )
    async def search_files(pattern: str, directory: str) -> list[str]:
        """递归搜索匹配 glob 模式的文件。

        Args:
            pattern: glob 模式（如 "*.py", "test_*.txt"）
            directory: 搜索根目录

        Returns:
            匹配的文件路径列表（排序后返回）
        """
        results = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)
