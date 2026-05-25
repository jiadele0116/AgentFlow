"""Tests for file operation tools."""
import pytest
from src.tools.registry import ToolRegistry
from src.tools.builtin.file_ops import register_file_tools


@pytest.mark.asyncio
async def test_read_file(tmp_path):
    registry = ToolRegistry()
    register_file_tools(registry)
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    result = await registry.get("read_file").handler(path=str(f))
    assert result == "hello world"


@pytest.mark.asyncio
async def test_write_file(tmp_path):
    registry = ToolRegistry()
    register_file_tools(registry)
    f = tmp_path / "out.txt"
    result = await registry.get("write_file").handler(path=str(f), content="new content")
    assert result is True
    assert f.read_text() == "new content"


@pytest.mark.asyncio
async def test_search_files(tmp_path):
    registry = ToolRegistry()
    register_file_tools(registry)
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "c.txt").write_text("c")
    result = await registry.get("search_files").handler(pattern="*.py", directory=str(tmp_path))
    assert len(result) == 2
