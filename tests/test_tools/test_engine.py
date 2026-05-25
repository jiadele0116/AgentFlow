"""Tests for tool engine."""
import asyncio
import pytest
from src.tools.registry import ToolRegistry
from src.tools.engine import ToolEngine


@pytest.mark.asyncio
async def test_engine_execute():
    registry = ToolRegistry()
    @registry.register(name="add", description="Add", parameters={"type": "object"})
    async def add(a: float, b: float) -> float: return a + b
    engine = ToolEngine(registry)
    result = await engine.execute("add", {"a": 2, "b": 3})
    assert result.success is True
    assert result.data == 5.0


@pytest.mark.asyncio
async def test_engine_missing_tool():
    registry = ToolRegistry()
    engine = ToolEngine(registry)
    result = await engine.execute("nope", {})
    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_engine_timeout():
    registry = ToolRegistry()
    @registry.register(name="slow", description="Slow", parameters={}, timeout=1)
    async def slow(): await asyncio.sleep(10)
    engine = ToolEngine(registry)
    result = await engine.execute("slow", {})
    assert result.success is False
