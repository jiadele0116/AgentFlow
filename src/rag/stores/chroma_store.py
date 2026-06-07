"""
ChromaDB 向量存储实现 (ChromaDB Vector Store)

使用 ChromaDB 作为向量存储后端，实现 VectorStore 抽象接口。

ChromaDB 特点：
  - 轻量级开源向量数据库，适合本地开发
  - 支持持久化存储（PersistentClient）
  - 内置 HNSW 索引，支持高效的近似最近邻搜索
  - 支持余弦距离（cosine）等相似度度量

距离到分数的转换：
  ChromaDB 返回的距离是 0-2 的值（余弦距离），
  通过 score = 1 - distance 转换为 0-1 的相似度分数。
  - distance = 0 → score = 1（完全相同）
  - distance = 1 → score = 0（正交）
  - distance = 2 → score = -1（完全相反）
"""
import uuid
from typing import Any
import chromadb
from src.rag.chunker import Chunk
from src.rag.vector_store import VectorStore, SearchResult


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现。

    使用 chromadb.PersistentClient 进行数据持久化，
    集合使用余弦距离（cosine）作为相似度度量。

    Args:
        persist_dir: 数据持久化目录
        collection: 集合名称
    """

    def __init__(self, persist_dir: str = "./data/chroma", collection: str = "default"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        # 使用余弦距离，HNSW 索引
        self._collection = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """插入或更新文档块到 ChromaDB。

        为每个块生成 UUID 作为唯一标识，将文本、嵌入向量和元数据一起存储。

        Args:
            chunks: 文档块列表
            embeddings: 嵌入向量列表（与 chunks 等长）
        """
        if not chunks:
            return
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{**c.metadata, "index": c.index, "source": c.source} for c in chunks]
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    async def search(self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """在 ChromaDB 中进行向量相似度搜索。

        ChromaDB 返回的 distances 是余弦距离（0-2），
        通过 1 - distance 转换为相似度分数（-1 到 1）。

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件（ChromaDB where 语法）

        Returns:
            按相似度降序排列的搜索结果
        """
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
        """从 ChromaDB 中删除指定 ID 的文档块。

        Args:
            ids: 要删除的文档块 ID 列表
        """
        if ids:
            self._collection.delete(ids=ids)
