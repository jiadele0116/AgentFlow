"""Hybrid retrieval pipeline with optional reranking."""
from typing import Any
from src.rag.vector_store import VectorStore, SearchResult


class RetrievalPipeline:
    def __init__(self, vector_store: VectorStore, embedder: Any, reranker: Any | None = None):
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(self, query: str, top_k: int = 5, rerank: bool = False, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        query_embedding = await self._embedder.embed(query)
        fetch_k = top_k * 2 if rerank and self._reranker else top_k
        results = await self._vector_store.search(query_embedding=query_embedding, top_k=fetch_k, filters=filters)
        if rerank and self._reranker and results:
            results = await self._reranker.rerank(query, results)
        return results[:top_k]
