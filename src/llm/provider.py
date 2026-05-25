"""LLM provider interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: TokenUsage
    model: str


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs) -> LLMResponse: ...

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs) -> AsyncIterator[str]: ...
