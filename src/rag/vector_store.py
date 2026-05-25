"""Vector store abstract base class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from src.rag.chunker import Chunk


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[SearchResult]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...
