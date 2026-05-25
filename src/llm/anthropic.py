"""Anthropic Claude LLM provider."""
import asyncio
from typing import Any, AsyncIterator

import anthropic

from src.llm.provider import LLMProvider, LLMResponse, TokenUsage, ToolCall


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 max_tokens: int = 4096, temperature: float = 0.7,
                 timeout: int = 60, max_retries: int = 3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]] | None = None, **kwargs) -> LLMResponse:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                params: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "messages": messages,
                }
                if tools:
                    params["tools"] = tools
                response = await self._client.messages.create(**params)
                return self._parse_response(response)
            except (anthropic.APIError, anthropic.RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_error  # type: ignore

    async def stream(self, messages: list[dict[str, Any]],
                     tools: list[dict[str, Any]] | None = None, **kwargs) -> AsyncIterator[str]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    def _parse_response(self, response: Any) -> LLMResponse:
        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
        usage = TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage, model=response.model)
