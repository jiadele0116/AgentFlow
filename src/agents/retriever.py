"""RetrieverAgent: runs RAG retrieval."""
from typing import Any
from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory


class RetrieverAgent(BaseAgent):
    def __init__(self, id: str, retrieval_pipeline: Any, memory: ShortTermMemory | None = None):
        super().__init__(id=id, role="retriever", memory=memory)
        self._retrieval = retrieval_pipeline

    async def run(self, query: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        self.transition(AgentState.EXECUTING)
        results = await self._retrieval.retrieve(query, top_k=top_k)
        knowledge = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
        self.memory.add({"role": "user", "content": query})
        self.memory.add({"role": "assistant", "content": f"Found {len(knowledge)} results"})
        self.transition(AgentState.COMPLETED)
        return knowledge
