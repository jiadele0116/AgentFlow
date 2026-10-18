"""Tests for Skills engine."""
import pytest
from unittest.mock import AsyncMock

from src.skills.engine import (
    Skill, SkillStep, SkillContext, SkillResult, SkillEngine, StepType,
)
from src.skills.loader import load_skill_from_yaml, _parse_skill, _parse_step
from src.core.exceptions import SkillError


# ============ SkillContext Tests ============

def test_context_get_set():
    ctx = SkillContext(input={"file_path": "test.py"})
    ctx.set("code_content", "print('hello')")
    assert ctx.get("code_content") == "print('hello')"
    assert ctx.get("input.file_path") == "test.py"
    assert ctx.get("nonexistent", "default") == "default"


def test_context_render_template():
    ctx = SkillContext(input={"name": "world"})
    ctx.set("greeting", "hello")
    result = ctx.render_template("{{ greeting }} {{ input.name }}")
    assert result == "hello world"


def test_context_record():
    ctx = SkillContext()
    ctx.record("step1", "tool_call", "result", 100.5)
    assert len(ctx.history) == 1
    assert ctx.history[0]["step_id"] == "step1"
    assert ctx.history[0]["duration_ms"] == 100.5


# ============ SkillEngine Tests ============

@pytest.fixture
def mock_engine():
    """创建带 mock 回调的 SkillEngine。"""
    tool_caller = AsyncMock(side_effect=lambda name, params: f"tool:{name}:ok")
    llm_caller = AsyncMock(side_effect=lambda prompt: "llm:response")
    engine = SkillEngine(tool_caller=tool_caller, llm_caller=llm_caller)
    return engine, tool_caller, llm_caller


@pytest.mark.asyncio
async def test_engine_execute_tool_call(mock_engine):
    engine, tool_caller, _ = mock_engine
    skill = Skill(
        name="test_tool",
        description="test",
        steps=[SkillStep(id="s1", type=StepType.TOOL_CALL, tool="read_file", params={"path": "test.py"}, output="content")],
    )
    engine.register(skill)
    result = await engine.execute("test_tool")
    assert result.success
    assert "tool:read_file:ok" in str(result.output)
    tool_caller.assert_called_once_with("read_file", {"path": "test.py"})


@pytest.mark.asyncio
async def test_engine_execute_llm_reasoning(mock_engine):
    engine, _, llm_caller = mock_engine
    skill = Skill(
        name="test_llm",
        description="test",
        steps=[SkillStep(id="s1", type=StepType.LLM_REASONING, prompt="分析代码", output="analysis")],
    )
    engine.register(skill)
    result = await engine.execute("test_llm")
    assert result.success
    assert result.output == "llm:response"
    llm_caller.assert_called_once()


@pytest.mark.asyncio
async def test_engine_execute_conditional_then(mock_engine):
    engine, _, llm_caller = mock_engine
    skill = Skill(
        name="test_cond",
        description="test",
        steps=[
            SkillStep(id="cond", type=StepType.CONDITIONAL, condition="true",
                       then_steps=[SkillStep(id="then_s", type=StepType.LLM_REASONING, prompt="yes", output="r")]),
        ],
    )
    engine.register(skill)
    result = await engine.execute("test_cond")
    assert result.success
    assert llm_caller.call_count == 1


@pytest.mark.asyncio
async def test_engine_execute_conditional_else(mock_engine):
    engine, _, llm_caller = mock_engine
    skill = Skill(
        name="test_cond_else",
        description="test",
        steps=[
            SkillStep(id="cond", type=StepType.CONDITIONAL, condition="false",
                       then_steps=[SkillStep(id="then_s", type=StepType.LLM_REASONING, prompt="yes")],
                       else_steps=[SkillStep(id="else_s", type=StepType.LLM_REASONING, prompt="no", output="r")]),
        ],
    )
    engine.register(skill)
    result = await engine.execute("test_cond_else")
    assert result.success
    assert llm_caller.call_count == 1


@pytest.mark.asyncio
async def test_engine_execute_loop(mock_engine):
    engine, _, llm_caller = mock_engine
    skill = Skill(
        name="test_loop",
        description="test",
        steps=[
            SkillStep(
                id="loop", type=StepType.LOOP,
                condition="{{ iteration }} == 0",
                max_iterations=3,
                body_steps=[SkillStep(id="body", type=StepType.LLM_REASONING, prompt="iter", output="r")],
            ),
        ],
    )
    engine.register(skill)
    result = await engine.execute("test_loop")
    assert result.success


@pytest.mark.asyncio
async def test_engine_execute_not_found(mock_engine):
    engine, _, _ = mock_engine
    result = await engine.execute("nonexistent")
    assert not result.success
    assert "未注册" in result.error


@pytest.mark.asyncio
async def test_engine_template_params(mock_engine):
    engine, tool_caller, _ = mock_engine
    skill = Skill(
        name="test_template",
        description="test",
        steps=[
            SkillStep(id="s1", type=StepType.TOOL_CALL, tool="read_file",
                       params={"path": "{{ input.file_path }}"}, output="content"),
        ],
    )
    engine.register(skill)
    result = await engine.execute("test_template", input_data={"file_path": "main.py"})
    assert result.success
    tool_caller.assert_called_once_with("read_file", {"path": "main.py"})


@pytest.mark.asyncio
async def test_engine_multi_step(mock_engine):
    engine, tool_caller, llm_caller = mock_engine
    skill = Skill(
        name="test_multi",
        description="test",
        steps=[
            SkillStep(id="s1", type=StepType.TOOL_CALL, tool="read_file", params={"path": "a.py"}, output="code"),
            SkillStep(id="s2", type=StepType.LLM_REASONING, prompt="分析 {{ code }}", output="analysis"),
        ],
    )
    engine.register(skill)
    result = await engine.execute("test_multi")
    assert result.success
    assert tool_caller.call_count == 1
    assert llm_caller.call_count == 1
    assert result.context.variables["code"] == "tool:read_file:ok"


def test_engine_list_skills(mock_engine):
    engine, _, _ = mock_engine
    engine.register(Skill(name="a", description="skill a"))
    engine.register(Skill(name="b", description="skill b"))
    skills = engine.list_skills()
    assert len(skills) == 2
    assert skills[0]["name"] == "a"


def test_engine_get_skill(mock_engine):
    engine, _, _ = mock_engine
    engine.register(Skill(name="my_skill", description="test"))
    assert engine.get("my_skill") is not None
    assert engine.get("other") is None


# ============ Condition Evaluation Tests ============

def test_evaluate_condition_bool():
    ctx = SkillContext()
    assert SkillEngine._evaluate_condition("true", ctx) is True
    assert SkillEngine._evaluate_condition("false", ctx) is False
    assert SkillEngine._evaluate_condition("", ctx) is True


def test_evaluate_condition_equality():
    ctx = SkillContext(input={"status": "ok"})
    assert SkillEngine._evaluate_condition("ok == ok", ctx) is True
    assert SkillEngine._evaluate_condition("ok == fail", ctx) is False
    assert SkillEngine._evaluate_condition("ok != fail", ctx) is True


# ============ YAML Loader Tests ============

def test_parse_skill():
    data = {
        "name": "test_skill",
        "description": "A test skill",
        "required_tools": ["read_file"],
        "steps": [
            {"id": "s1", "type": "tool_call", "tool": "read_file", "params": {"path": "test.py"}, "output": "content"},
            {"id": "s2", "type": "llm_reasoning", "prompt": "Analyze {{ content }}", "output": "result"},
        ],
    }
    skill = _parse_skill(data)
    assert skill.name == "test_skill"
    assert len(skill.steps) == 2
    assert skill.steps[0].type == StepType.TOOL_CALL
    assert skill.steps[1].type == StepType.LLM_REASONING


def test_parse_step_conditional():
    data = {
        "id": "cond",
        "type": "conditional",
        "condition": "true",
        "then_steps": [{"id": "t", "type": "llm_reasoning", "prompt": "yes"}],
        "else_steps": [{"id": "e", "type": "llm_reasoning", "prompt": "no"}],
    }
    step = _parse_step(data)
    assert step.type == StepType.CONDITIONAL
    assert len(step.then_steps) == 1
    assert len(step.else_steps) == 1


def test_parse_step_loop():
    data = {
        "id": "loop",
        "type": "loop",
        "condition": "true",
        "max_iterations": 5,
        "body_steps": [{"id": "body", "type": "llm_reasoning", "prompt": "loop body"}],
    }
    step = _parse_step(data)
    assert step.type == StepType.LOOP
    assert step.max_iterations == 5
    assert len(step.body_steps) == 1


def test_parse_skill_missing_name():
    with pytest.raises(SkillError, match="name"):
        _parse_skill({"steps": []})


def test_parse_step_missing_id():
    with pytest.raises(SkillError, match="id"):
        _parse_step({"type": "tool_call"})


def test_parse_step_invalid_type():
    with pytest.raises(SkillError, match="类型无效"):
        _parse_step({"id": "s1", "type": "invalid_type"})


def test_load_skill_from_yaml(tmp_path):
    yaml_content = """
name: test_yaml_skill
description: "Test skill from YAML"
required_tools:
  - read_file
steps:
  - id: read
    type: tool_call
    tool: read_file
    params:
      path: "{{ input.file_path }}"
    output: content
"""
    yaml_file = tmp_path / "test_skill.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    skill = load_skill_from_yaml(yaml_file)
    assert skill.name == "test_yaml_skill"
    assert len(skill.steps) == 1
    assert skill.steps[0].tool == "read_file"


def test_load_skill_file_not_found():
    with pytest.raises(SkillError, match="不存在"):
        load_skill_from_yaml("/nonexistent/path.yaml")
