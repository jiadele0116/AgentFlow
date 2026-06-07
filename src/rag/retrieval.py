"""
混合检索管道模块 (Hybrid Retrieval Pipeline)

实现完整的 RAG 检索流程：查询 → 嵌入 → 向量搜索 → (可选)重排序 → 结果

检索流程：
  1. 将查询文本通过 Embedder 转换为向量
  2. 如果启用重排序，先检索 2*top_k 个候选结果（多取一些供精排）
  3. 在 VectorStore 中进行向量相似度搜索
  4. 如果启用重排序，使用 Reranker 对候选结果精排
  5. 返回 top_k 个最相关的结果

重排序 (Reranking) 的价值：
  - 向量检索使用双编码器（bi-encoder），速度快但精度有限
  - 重排序使用交叉编码器（cross-encoder），对 query-document 对精确打分
  - 两阶段检索：先粗召回再精排，平衡速度和精度
"""
from typing import Any
from src.rag.vector_store import VectorStore, SearchResult


class RetrievalPipeline:
    """混合检索管道：整合嵌入、向量搜索和重排序。

    Args:
        vector_store: 向量存储实例
        embedder: 文本嵌入器实例
        reranker: 重排序器实例（可选，None 则不重排序）
    """

    def __init__(self, vector_store: VectorStore, embedder: Any, reranker: Any | None = None):
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker

    async def retrieve(self, query: str, top_k: int = 5, rerank: bool = False, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """执行混合检索。

        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            rerank: 是否启用重排序
            filters: 元数据过滤条件

        Returns:
            按相关性降序排列的搜索结果列表
        """
        # 步骤1：将查询文本嵌入为向量
        query_embedding = await self._embedder.embed(query)

        # 步骤2：如果启用重排序，多取一些候选结果
        fetch_k = top_k * 2 if rerank and self._reranker else top_k

        # 步骤3：向量相似度搜索
        results = await self._vector_store.search(query_embedding=query_embedding, top_k=fetch_k, filters=filters)

        # 步骤4：可选的重排序精排
        if rerank and self._reranker and results:
            results = await self._reranker.rerank(query, results)

        return results[:top_k]
