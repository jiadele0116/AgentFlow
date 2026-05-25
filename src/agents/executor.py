"""ExecutorAgent: executes tasks via strategy."""
from src.agents.base import BaseAgent, AgentState, Result
from src.agents.memory import ShortTermMemory
from src.agents.strategy import ExecutionStrategy


class ExecutorAgent(BaseAgent):
    def __init__(self, id: str, strategy: ExecutionStrategy, memory: ShortTermMemory | None = None):
        super().__init__(id=id, role="executor", memory=memory)
        self._strategy = strategy

    async def run(self, task_description: str, context: list[dict] | None = None, **kwargs) -> Result:
        self.transition(AgentState.EXECUTING)
        self.memory.add({"role": "user", "content": task_description})
        result = await self._strategy.execute(task_description=task_description, context=context or self.memory.get_all())
        self.memory.add({"role": "assistant", "content": result.content})
        self.transition(AgentState.COMPLETED)
        return Result(task_id=self.id, success=True, data={"content": result.content, "tool_calls": result.tool_calls_made})
