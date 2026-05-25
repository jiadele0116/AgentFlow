"""LLM-based query rewriting for better retrieval."""
from typing import Any
from src.llm.provider import LLMProvider


class QueryRewriter:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def rewrite(self, query: str) -> str:
        response = await self._llm.complete(messages=[
            {"role": "system", "content": "Rewrite the user's query to be more specific and searchable. Return only the rewritten query."},
            {"role": "user", "content": query},
        ])
        return response.content.strip() or query
