from typing import Any


class MultiAgentError(Exception):
    """Base exception for the multi-agent system."""

    def __init__(self, message: str, code: str = "", context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class AgentError(MultiAgentError):
    """Error originating from an agent."""

    def __init__(self, message: str, agent_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id


class ToolError(MultiAgentError):
    """Error originating from a tool execution."""

    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class SkillError(MultiAgentError):
    """Error originating from a skill execution."""

    def __init__(self, message: str, skill_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.skill_name = skill_name


class RAGError(MultiAgentError):
    """Error originating from the RAG module."""


class MCPError(MultiAgentError):
    """Error originating from the MCP module."""

    def __init__(self, message: str, server_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.server_name = server_name


class ConfigError(MultiAgentError):
    """Error originating from configuration."""
