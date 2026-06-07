"""
MCP 文件系统服务器 (FileSystem Server)

提供文件系统操作的 MCP 服务器实现。通过 @tool 装饰器暴露文件操作工具，
通过 @resource 装饰器暴露文件资源。

提供的工具：
  - read_file(path): 读取文件内容
  - write_file(path, content): 写入文件
  - list_dir(path): 列出目录内容
  - search_files(pattern, directory): 按模式搜索文件

提供的资源：
  - file_resource(uri): 通过 file:// URI 读取文件

安全特性：
  - 路径解析：相对路径基于 root_dir 解析
  - 路径规范化：使用 resolve() 防止路径遍历
"""
import fnmatch
import os
from pathlib import Path
from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="filesystem", version="1.0")
class FileSystemServer:
    """文件系统 MCP 服务器。

    Args:
        root_dir: 文件操作的根目录（相对路径基于此目录解析）
    """

    def __init__(self, root_dir: str = "."):
        self._root = Path(root_dir)

    def _resolve(self, path: str) -> Path:
        """解析路径：相对路径基于 root_dir，绝对路径直接使用。

        Args:
            path: 输入路径

        Returns:
            规范化的绝对路径
        """
        p = Path(path)
        if not p.is_absolute():
            p = self._root / p
        return p.resolve()

    @tool(description="Read file contents")
    async def read_file(self, path: str) -> str:
        """读取文件内容。"""
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @tool(description="Write content to a file")
    async def write_file(self, path: str, content: str) -> bool:
        """写入内容到文件（自动创建父目录）。"""
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @tool(description="List directory contents")
    async def list_dir(self, path: str = ".") -> list[str]:
        """列出目录下的所有文件和子目录。"""
        p = self._resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [str(item.name) for item in p.iterdir()]

    @tool(description="Search for files matching a pattern")
    async def search_files(self, pattern: str, directory: str = ".") -> list[str]:
        """递归搜索匹配 glob 模式的文件。"""
        d = self._resolve(directory)
        results = []
        for root, dirs, files in os.walk(d):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)

    @resource(uri_scheme="file")
    async def file_resource(self, uri: str) -> str:
        """通过 file:// URI 读取文件资源。"""
        return await self.read_file(uri.replace("file://", ""))
