"""ChromaDB vector store implementation."""
import uuid
from typing import Any
import chromadb
from src.rag.chunker import Chunk
from src.rag.vector_store import VectorStore, SearchResult


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str = "./data/chroma", collection: str = "default"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{**c.metadata, "index": c.index, "source": c.source} for c in chunks]
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    async def search(self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        count = self._collection.count()
        n = min(top_k, count) if count > 0 else 0
        if n == 0:
            return []
        kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": n}
        if filters:
            kwargs["where"] = filters
        results = self._collection.query(**kwargs)
        return [
            SearchResult(id=results["ids"][0][i], text=results["documents"][0][i],
                         score=1 - results["distances"][0][i],
                         metadata=results["metadatas"][0][i] if results["metadatas"] else {})
            for i in range(len(results["ids"][0]))
        ]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)
