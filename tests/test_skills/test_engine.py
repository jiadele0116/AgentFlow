"""Tests for skill engine."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.llm.provider import LLMResponse, TokenUsage
from src.skills.engine import SkillEngine


@pytest.mark.asyncio
async def test_skill_engine_tool_call_step():
    mock_tool = MagicMock()
    mock_tool.execute = AsyncMock(return_value=MagicMock(success=True, data="file content"))
    mock_tool.list_tools = MagicMock(return_value=[])
    mock_llm = MagicMock()
    engine = SkillEngine(tool_engine=mock_tool, llm=mock_llm)
    skill = {"name": "test", "steps": [{"id": "read", "type": "tool_call", "tool": "read_file", "params": {"path": "{{ input.file_path }}"}, "output": "content"}]}
    result = await engine.execute_skill(skill, {"file_path": "/tmp/test.txt"})
    assert result["content"] == "file content"


@pytest.mark.asyncio
async def test_skill_engine_llm_step():
    mock_tool = MagicMock()
    mock_tool.list_tools = MagicMock(return_value=[])
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(content="Analysis: good", tool_calls=None, usage=TokenUsage(), model="test"))
    engine = SkillEngine(tool_engine=mock_tool, llm=mock_llm)
    skill = {"name": "analyze", "steps": [{"id": "think", "type": "llm_reasoning", "prompt": "Analyze: {{ input.text }}", "output": "analysis"}]}
    result = await engine.execute_skill(skill, {"text": "some code"})
    assert "good" in result["analysis"]
