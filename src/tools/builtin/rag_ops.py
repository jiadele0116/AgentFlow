"""RAG-powered tools bridging Tools and RAG modules."""
from typing import Any
from src.tools.registry import ToolRegistry


def register_rag_tools(registry: ToolRegistry, retrieval_pipeline: Any = None, document_processor: Any = None):
    @registry.register(
        name="search_knowledge", description="Search the knowledge base",
        parameters={"type": "object", "properties": {
            "query": {"type": "string"}, "collection": {"type": "string", "default": "default"}, "top_k": {"type": "integer", "default": 5},
        }, "required": ["query"]},
    )
    async def search_knowledge(query: str, collection: str = "default", top_k: int = 5) -> list[dict]:
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
        if document_processor is None:
            raise RuntimeError("Document processor not configured")
        chunks = await document_processor.ingest(source, metadata={"collection": collection})
        return len(chunks)
