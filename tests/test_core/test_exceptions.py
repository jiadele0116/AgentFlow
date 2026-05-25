import pytest


def test_base_error_has_code_and_context():
    from src.core.exceptions import MultiAgentError

    err = MultiAgentError("something broke", code="ERR_001", context={"key": "val"})
    assert str(err) == "something broke"
    assert err.code == "ERR_001"
    assert err.context == {"key": "val"}


def test_agent_error_inherits():
    from src.core.exceptions import AgentError, MultiAgentError

    err = AgentError("agent failed", agent_id="agent-1")
    assert isinstance(err, MultiAgentError)
    assert err.agent_id == "agent-1"


def test_tool_error_inherits():
    from src.core.exceptions import ToolError, MultiAgentError

    err = ToolError("tool failed", tool_name="read_file")
    assert isinstance(err, MultiAgentError)
    assert err.tool_name == "read_file"


def test_skill_error_inherits():
    from src.core.exceptions import SkillError, MultiAgentError

    err = SkillError("skill failed", skill_name="code_review")
    assert isinstance(err, MultiAgentError)
    assert err.skill_name == "code_review"


def test_rag_error_inherits():
    from src.core.exceptions import RAGError, MultiAgentError

    err = RAGError("rag failed")
    assert isinstance(err, MultiAgentError)


def test_mcp_error_inherits():
    from src.core.exceptions import MCPError, MultiAgentError

    err = MCPError("mcp failed", server_name="filesystem")
    assert isinstance(err, MultiAgentError)
    assert err.server_name == "filesystem"


def test_config_error_inherits():
    from src.core.exceptions import ConfigError, MultiAgentError

    err = ConfigError("config bad")
    assert isinstance(err, MultiAgentError)
