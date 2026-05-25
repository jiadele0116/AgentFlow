"""Tests for data tools."""
import pytest
from src.tools.registry import ToolRegistry
from src.tools.builtin.data_ops import register_data_tools


@pytest.mark.asyncio
async def test_parse_json():
    registry = ToolRegistry()
    register_data_tools(registry)
    result = await registry.get("parse_json").handler(text='{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}


@pytest.mark.asyncio
async def test_csv_to_dict():
    registry = ToolRegistry()
    register_data_tools(registry)
    result = await registry.get("csv_to_dict").handler(csv_text="name,age\nAlice,30\nBob,25")
    assert len(result) == 2
    assert result[0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_json_to_csv():
    registry = ToolRegistry()
    register_data_tools(registry)
    result = await registry.get("json_to_csv").handler(data=[{"name": "Alice", "age": "30"}])
    assert "Alice" in result
