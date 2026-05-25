"""CriticAgent: reviews output and provides feedback."""
from typing import Any
from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory
from src.llm.provider import LLMProvider


class CriticAgent(BaseAgent):
    def __init__(self, id: str, llm: LLMProvider, memory: ShortTermMemory | None = None):
        super().__init__(id=id, role="critic", memory=memory)
        self._llm = llm

    async def run(self, content_to_review: str, original_task: str = "", **kwargs) -> dict[str, Any]:
        self.transition(AgentState.REVIEWING)
        messages = [
            {"role": "system", "content": "You are a review agent. Review the work for correctness, performance, security, maintainability. Output APPROVED if good, or REJECTED with specific feedback."},
            {"role": "user", "content": f"Original task: {original_task}\n\nWork to review:\n{content_to_review}"},
        ]
        response = await self._llm.complete(messages=messages)
        approved = "APPROVED" in response.content.upper()
        self.transition(AgentState.COMPLETED)
        return {"approved": approved, "feedback": response.content}
