"""长期记忆模块 (Long-Term Memory)

基于向量数据库的跨会话记忆，保存重要的事实和知识。
使用语义检索来回忆与当前话题相关的记忆。

存储结构：
  - 每条记忆是一个文本块，附带 metadata（session_id, timestamp, type）
  - 通过 embedding 向量化后存入 ChromaDB
  - 查询时通过语义相似度匹配最相关的记忆
"""
import time
import uuid
from typing import Any

from src.rag.vector_store import VectorStore, SearchResult
from src.rag.chunker import Chunk
from src.core.logging import get_logger

logger = get_logger("memory.long_term")


class LongTermMemory:
    """长期记忆：基于向量数据库的跨会话记忆。

    Args:
        vector_store: 向量存储实例（如 ChromaVectorStore）
        embedder: 文本嵌入模型实例
    """

    def __init__(self, vector_store: VectorStore, embedder):
        self._vector_store = vector_store
        self._embedder = embedder

    async def save(self, session_id: str, content: str, memory_type: str = "conversation"):
        """保存一条记忆到长期存储。

        Args:
            session_id: 来源会话 ID
            content: 记忆内容（一段文本）
            memory_type: 记忆类型 ("conversation" / "fact" / "summary")
        """
        try:
            embedding = await self._embedder.embed(content)
            chunk = Chunk(
                text=content,
                index=0,
                source=f"memory:{session_id}",
                metadata={
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "timestamp": time.time(),
                },
            )
            await self._vector_store.upsert([chunk], [embedding])
            logger.info(f"Memory saved: {content[:50]}...")
        except Exception as e:
            logger.warning(f"保存记忆失败: {e}")

    async def recall(self, query: str, session_id: str | None = None, top_k: int = 3) -> list[SearchResult]:
        """根据查询回忆相关的长期记忆。

        Args:
            query: 查询文本
            session_id: 限定在某个会话的记忆中搜索（None 搜索全部）
            top_k: 返回最相关的 K 条记忆

        Returns:
            相关记忆列表
        """
        try:
            query_embedding = await self._embedder.embed(query)
            filters = None
            if session_id:
                filters = {"session_id": session_id}

            results = await self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            logger.info(f" {len(results)} 条记忆: {query[:30]}...")
            return results
        except Exception as e:
            logger.warning(f"回忆记忆失败: {e}")
            return []

    async def get_context_prompt(self, query: str, session_id: str | None = None, top_k: int = 3) -> str:
        """回忆与查询相关的记忆，拼成 LLM 可用的上下文字符串。

        Args:
            query: 查询文本
            session_id: 限定搜索范围
            top_k: 返回条数

        Returns:
            格式化的记忆上下文
        """
        results = await self.recall(query, session_id=session_id, top_k=top_k)
        if not results:
            return ""

        lines = ["以下是与当前话题相关的记忆："]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.text}")
        return "\n".join(lines)
