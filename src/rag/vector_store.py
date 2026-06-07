"""
向量存储抽象基类模块 (Vector Store Abstract Base Class)

定义向量数据库的统一接口。所有向量存储实现（ChromaDB、FAISS、Pinecone 等）
都必须实现此接口，确保上层代码可以透明切换存储后端。

核心操作：
  - upsert: 插入或更新文档块及其嵌入向量
  - search: 根据查询向量进行相似度搜索
  - delete: 根据 ID 删除文档块

使用示例：
    store = ChromaVectorStore(persist_dir="./data/chroma")
    await store.upsert(chunks, embeddings)
    results = await store.search(query_embedding, top_k=5)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from src.rag.chunker import Chunk


@dataclass
class SearchResult:
    """向量搜索结果。

    Attributes:
        id: 文档块唯一标识符
        text: 文档块文本内容
        score: 相似度分数（0-1，越高越相关）
        metadata: 文档块的元数据
    """
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """向量存储抽象基类。

    定义向量数据库必须实现的三个核心操作。
    """

    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """插入或更新文档块及其嵌入向量。

        chunks 和 embeddings 必须一一对应。

        Args:
            chunks: 文档块列表
            embeddings: 嵌入向量列表（与 chunks 等长）
        """

        ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """根据查询向量进行相似度搜索。

        Args:
            query_embedding: 查询文本的嵌入向量
            top_k: 返回最相似的 K 个结果
            filters: 可选的元数据过滤条件

        Returns:
            按相似度降序排列的搜索结果列表
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """根据 ID 列表删除文档块。

        Args:
            ids: 要删除的文档块 ID 列表
        """
        ...
