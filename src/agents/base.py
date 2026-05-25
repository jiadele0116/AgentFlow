"""Base agent with state machine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from src.agents.memory import ShortTermMemory


class AgentState(Enum):
    INIT = "init"
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETED = "completed"


@dataclass
class Task:
    id: str
    description: str
    input_data: dict[str, Any] = field(default_factory=dict)
    assigned_to: str | None = None
    parent_id: str | None = None


@dataclass
class Result:
    task_id: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0


@dataclass
class PlannerResult:
    success: bool
    subtasks: list[str] = field(default_factory=list)
    content: str = ""


class BaseAgent(ABC):
    def __init__(self, id: str, role: str = "", goal: str = "",
                 capabilities: list[str] | None = None, memory: ShortTermMemory | None = None):
        self.id = id
        self.role = role
        self.goal = goal
        self.capabilities = capabilities or []
        self.memory = memory or ShortTermMemory()
        self.state = AgentState.IDLE

    def transition(self, new_state: AgentState):
        self.state = new_state

    @abstractmethod
    async def run(self, **kwargs) -> Any: ...
