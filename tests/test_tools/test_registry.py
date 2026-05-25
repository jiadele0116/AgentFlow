"""Tests for tool registry."""
import pytest
from src.tools.registry import ToolRegistry


def test_register_and_list():
    registry = ToolRegistry()
    @registry.register(name="greet", description="Say hello", parameters={"type": "object", "properties": {"name": {"type": "string"}}})
    async def greet(name: str) -> str: return f"Hello, {name}!"
    assert "greet" in registry.list_names()
    assert registry.get("greet").description == "Say hello"


def test_list_tools_returns_schemas():
    registry = ToolRegistry()
    @registry.register(name="add", description="Add", parameters={"type": "object", "properties": {"a": {"type": "number"}}})
    async def add(a: float): return a
    tools = registry.list_tools()
    assert tools[0]["name"] == "add"


def test_unregister():
    registry = ToolRegistry()
    @registry.register(name="temp", description="temp", parameters={})
    async def temp(): pass
    registry.unregister("temp")
    assert "temp" not in registry.list_names()
