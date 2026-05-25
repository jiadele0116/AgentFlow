"""CoordinatorAgent: manages inter-agent communication."""
from typing import Any
from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory
from src.agents.message_bus import AgentMessageBus


class CoordinatorAgent(BaseAgent):
    def __init__(self, id: str, message_bus: AgentMessageBus, memory: ShortTermMemory | None = None):
        super().__init__(id=id, role="coordinator", memory=memory)
        self._bus = message_bus
        self._task_status: dict[str, str] = {}

    async def run(self, tasks: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.transition(AgentState.EXECUTING)
        results = {}
        for task in tasks:
            task_id = task.get("id", "")
            agent_id = task.get("assigned_to", "")
            description = task.get("description", "")
            self._task_status[task_id] = "pending"
            await self._bus.send(sender=self.id, receiver=agent_id, msg_type="task",
                                 payload={"task_id": task_id, "description": description}, correlation_id=task_id)
            self._task_status[task_id] = "dispatched"
            results[task_id] = {"status": "dispatched", "agent": agent_id}
        self.transition(AgentState.COMPLETED)
        return {"tasks_dispatched": len(results), "status": results}

    async def handle_result(self, message: Any):
        task_id = message.payload.get("task_id", message.correlation_id)
        self._task_status[task_id] = "completed"
        self.memory.add({"role": "system", "content": f"Task {task_id} completed"})
