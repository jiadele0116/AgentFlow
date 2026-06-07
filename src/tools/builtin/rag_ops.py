"""
RAG 操作工具模块 (RAG Operation Tools)

桥接 Tools 模块和 RAG 模块，将 RAG 能力暴露为可被 Agent 调用的工具。

注册的工具：
  - search_knowledge(query, collection, top_k): 在知识库中语义搜索
  - index_document(source, collection): 将文档索引到知识库

这些工具是 Tools 模块和 RAG 模块之间的桥梁：
  Agent → ToolEngine → rag_ops → RetrievalPipeline/VectorStore
"""
from typing import Any
from src.tools.registry import ToolRegistry


def register_rag_tools(registry: ToolRegistry, retrieval_pipeline: Any = None, document_processor: Any = None):
    """注册 RAG 操作工具到注册表。

    Args:
        registry: 工具注册表
        retrieval_pipeline: RAG 检索管道实例（search_knowledge 依赖）
        document_processor: 文档处理器实例（index_document 依赖）
    """

    @registry.register(
        name="search_knowledge", description="Search the knowledge base",
        parameters={"type": "object", "properties": {
            "query": {"type": "string"}, "collection": {"type": "string", "default": "default"}, "top_k": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    )
    async def search_knowledge(query: str, collection: str = "default", top_k: int = 5) -> list[dict]:
        """在知识库中进行语义搜索。

        通过 RAG 检索管道将查询向量化后与存储的文档块进行相似度匹配。

        Args:
            query: 搜索查询文本
            collection: 目标集合名称（当前未使用，预留多集合支持）
            top_k: 返回最相关的 K 个结果

        Returns:
            结果列表，每项包含 text（文本）、score（相似度分数）、metadata（元数据）

        Raises:
            RuntimeError: RAG 检索管道未配置
        """
        if retrieval_pipeline is None:
            raise RuntimeError("RAG retrieval pipeline not configured")
        results = await retrieval_pipeline.retrieve(query, top_k=top_k)
        return [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    @registry.register(
        name="index_document", description="Index a document into the knowledge base",
        parameters={"type": "object", "properties": {
            "source": {"type": "string"}, "collection": {"type": "string", "default": "default"},
        }, "required": ["source"]},
    )
    async def index_document(source: str, collection: str = "default") -> int:
        """将文档索引到知识库。

        文档会被分块、嵌入后存储到向量数据库中。

        Args:
            source: 文档来源（文件路径或文本）
            collection: 目标集合名称

        Returns:
            索引的文档块数量

        Raises:
            RuntimeError: 文档处理器未配置
        """
        if document_processor is None:
            raise RuntimeError("Document processor not configured")
        chunks = await document_processor.ingest(source, metadata={"collection": collection})
        return len(chunks)
