"""PlannerAgent: decomposes tasks into subtasks."""
from src.agents.base import BaseAgent, AgentState, PlannerResult
from src.agents.memory import ShortTermMemory
from src.llm.provider import LLMProvider


class PlannerAgent(BaseAgent):
    def __init__(self, id: str, llm: LLMProvider, memory: ShortTermMemory | None = None):
        super().__init__(id=id, role="planner", memory=memory)
        self._llm = llm

    async def run(self, task_description: str, **kwargs) -> PlannerResult:
        self.transition(AgentState.PLANNING)
        self.memory.add({"role": "user", "content": task_description})
        messages = [
            {"role": "system", "content": "You are a task planning agent. Decompose complex tasks into clear, actionable subtasks. Output each subtask on a numbered line."},
            {"role": "user", "content": task_description},
        ]
        response = await self._llm.complete(messages=messages)
        subtasks = [l.strip() for l in response.content.split("\n") if l.strip() and l.strip()[0].isdigit()]
        self.memory.add({"role": "assistant", "content": response.content})
        self.transition(AgentState.COMPLETED)
        return PlannerResult(success=True, subtasks=subtasks, content=response.content)
