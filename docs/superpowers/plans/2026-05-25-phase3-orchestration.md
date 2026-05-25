# Phase 3: Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Agents (base, 5 types, memory, strategies, message bus), MCP (protocol, client, servers), and Skills (engine, loader, YAML definitions) modules — the orchestration layer that ties everything together.

**Architecture:** Agents are autonomous units that communicate via the EventBus. Each agent has a role, strategy, and memory. MCP provides a standardized protocol for tool/resource access. Skills compose Tools + LLM into higher-level capabilities.

**Tech Stack:** Python 3.11+, PyYAML, Jinja2, aiosqlite, Core/LLM/Tools/RAG from Phases 1-2

---

## File Structure

```
src/agents/
├── __init__.py
├── base.py              # BaseAgent ABC, Task, Result, AgentState
├── memory.py            # Short-term (list) + Long-term (SQLite) memory
├── strategy.py          # ReAct, PlanExecute, CoT strategies
├── message_bus.py       # AgentMessageBus wrapping Core EventBus
├── planner.py           # PlannerAgent
├── executor.py          # ExecutorAgent
├── critic.py            # CriticAgent
├── retriever.py         # RetrieverAgent
└── coordinator.py       # CoordinatorAgent

src/mcp/
├── __init__.py
├── protocol.py          # MCPRequest, MCPResponse, MCPError
├── server.py            # MCPServer ABC + decorators
├── client.py            # MCPClient
└── servers/
    ├── __init__.py
    ├── filesystem.py    # FileSystemServer
    └── web.py           # WebServer

src/skills/
├── __init__.py
├── engine.py            # SkillEngine
├── loader.py            # YAML skill loader
└── definitions/
    ├── code_review.yaml
    └── data_analysis.yaml

tests/
├── test_agents/
│   ├── __init__.py
│   ├── test_memory.py
│   ├── test_strategy.py
│   ├── test_message_bus.py
│   └── test_planner.py
├── test_mcp/
│   ├── __init__.py
│   ├── test_protocol.py
│   └── test_server.py
├── test_skills/
│   ├── __init__.py
│   ├── test_engine.py
│   └── test_loader.py
```

---

## Task 1: Agent Memory

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/memory.py`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_agents/test_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_memory.py
import pytest


@pytest.mark.asyncio
async def test_short_term_memory_add_and_get():
    from src.agents.memory import ShortTermMemory

    mem = ShortTermMemory(max_turns=5)
    mem.add({"role": "user", "content": "hello"})
    mem.add({"role": "assistant", "content": "hi there"})

    messages = mem.get_all()
    assert len(messages) == 2
    assert messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_short_term_memory_eviction():
    from src.agents.memory import ShortTermMemory

    mem = ShortTermMemory(max_turns=3)
    for i in range(5):
        mem.add({"role": "user", "content": f"msg {i}"})

    messages = mem.get_all()
    assert len(messages) == 3
    assert messages[0]["content"] == "msg 2"  # oldest evicted


@pytest.mark.asyncio
async def test_short_term_memory_clear():
    from src.agents.memory import ShortTermMemory

    mem = ShortTermMemory(max_turns=5)
    mem.add({"role": "user", "content": "hello"})
    mem.clear()
    assert len(mem.get_all()) == 0


@pytest.mark.asyncio
async def test_long_term_memory_store_and_retrieve(tmp_path):
    from src.agents.memory import LongTermMemory

    db_path = str(tmp_path / "test.db")
    mem = LongTermMemory(db_path=db_path)

    await mem.store(agent_id="agent-1", key="fact1", value="Paris is the capital of France")
    await mem.store(agent_id="agent-1", key="fact2", value="Python is a language")

    results = await mem.retrieve(agent_id="agent-1", query="capital")
    assert len(results) >= 1
    assert any("Paris" in r["value"] for r in results)


@pytest.mark.asyncio
async def test_long_term_memory_agent_isolation(tmp_path):
    from src.agents.memory import LongTermMemory

    db_path = str(tmp_path / "test.db")
    mem = LongTermMemory(db_path=db_path)

    await mem.store(agent_id="agent-1", key="secret", value="agent 1 data")
    await mem.store(agent_id="agent-2", key="secret", value="agent 2 data")

    results_1 = await mem.retrieve(agent_id="agent-1", query="data")
    results_2 = await mem.retrieve(agent_id="agent-2", query="data")

    assert any("agent 1" in r["value"] for r in results_1)
    assert any("agent 2" in r["value"] for r in results_2)
    assert not any("agent 2" in r["value"] for r in results_1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents/test_memory.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement memory**

```python
# src/agents/__init__.py
```

```python
# src/agents/memory.py
import aiosqlite
from collections import deque
from typing import Any


class ShortTermMemory:
    """In-memory message buffer with max turns limit."""

    def __init__(self, max_turns: int = 20):
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_turns)

    def add(self, message: dict[str, Any]):
        """Add a message to short-term memory."""
        self._messages.append(message)

    def get_all(self) -> list[dict[str, Any]]:
        """Get all messages in memory."""
        return list(self._messages)

    def clear(self):
        """Clear all messages."""
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)


class LongTermMemory:
    """SQLite-backed persistent memory with per-agent isolation."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self):
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent
                ON memories(agent_id)
            """)
            await self._db.commit()

    async def store(self, agent_id: str, key: str, value: str):
        """Store a memory entry for an agent."""
        await self._ensure_db()
        await self._db.execute(
            "INSERT INTO memories (agent_id, key, value) VALUES (?, ?, ?)",
            (agent_id, key, value),
        )
        await self._db.commit()

    async def retrieve(self, agent_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve memories matching a query for a specific agent."""
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT key, value, created_at FROM memories WHERE agent_id = ? AND value LIKE ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [{"key": r[0], "value": r[1], "created_at": r[2]} for r in rows]

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_memory.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/__init__.py src/agents/memory.py tests/test_agents/__init__.py tests/test_agents/test_memory.py
git commit -m "feat: agent memory — ShortTermMemory (deque) + LongTermMemory (SQLite)"
```

---

## Task 2: Agent Strategies

**Files:**
- Create: `src/agents/strategy.py`
- Create: `tests/test_agents/test_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_strategy.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.llm.provider import LLMResponse, TokenUsage


@pytest.mark.asyncio
async def test_react_strategy_single_step():
    """ReAct strategy executes thought-action-observation loop."""
    from src.agents.strategy import ReActStrategy

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=[
        # Thought + Action
        LLMResponse(
            content="I need to read the file.",
            tool_calls=[MagicMock(id="c1", name="read_file", arguments={"path": "/tmp/test.txt"})],
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            model="test",
        ),
        # Final answer after observation
        LLMResponse(
            content="The file contains hello world.",
            tool_calls=None,
            usage=TokenUsage(input_tokens=10, output_tokens=15),
            model="test",
        ),
    ])

    mock_tool_engine = MagicMock()
    mock_tool_engine.execute = AsyncMock(return_value=MagicMock(success=True, data="hello world"))
    mock_tool_engine.list_tools = MagicMock(return_value=[{"name": "read_file"}])

    strategy = ReActStrategy(llm=mock_llm, tool_engine=mock_tool_engine, max_iterations=3)
    result = await strategy.execute(
        task_description="Read the file and summarize it",
        context=[],
    )

    assert result.content == "The file contains hello world."
    assert mock_llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_react_strategy_max_iterations():
    """ReAct strategy stops at max iterations."""
    from src.agents.strategy import ReActStrategy

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Still thinking...",
        tool_calls=[MagicMock(id="c1", name="think", arguments={})],
        usage=TokenUsage(input_tokens=5, output_tokens=5),
        model="test",
    ))

    mock_tool_engine = MagicMock()
    mock_tool_engine.execute = AsyncMock(return_value=MagicMock(success=True, data="ok"))
    mock_tool_engine.list_tools = MagicMock(return_value=[])

    strategy = ReActStrategy(llm=mock_llm, tool_engine=mock_tool_engine, max_iterations=2)
    result = await strategy.execute(task_description="Complex task", context=[])

    # Should stop after 2 iterations
    assert mock_llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_cot_strategy():
    """CoT strategy does chain-of-thought without tools."""
    from src.agents.strategy import CoTStrategy

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Step 1: Analyze. Step 2: Conclude. Result: 42.",
        tool_calls=None,
        usage=TokenUsage(input_tokens=20, output_tokens=30),
        model="test",
    ))

    strategy = CoTStrategy(llm=mock_llm)
    result = await strategy.execute(task_description="What is the answer?", context=[])

    assert "42" in result.content
    mock_llm.complete.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents/test_strategy.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement strategies**

```python
# src/agents/strategy.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.llm.provider import LLMProvider, LLMResponse


@dataclass
class StrategyResult:
    content: str
    tool_calls_made: int = 0
    iterations: int = 0


class ExecutionStrategy(ABC):
    """Base class for agent execution strategies."""

    @abstractmethod
    async def execute(
        self,
        task_description: str,
        context: list[dict[str, Any]],
    ) -> StrategyResult:
        ...


class ReActStrategy(ExecutionStrategy):
    """Thought → Action → Observation loop."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_engine: Any,
        max_iterations: int = 10,
    ):
        self._llm = llm
        self._tool_engine = tool_engine
        self._max_iterations = max_iterations

    async def execute(
        self,
        task_description: str,
        context: list[dict[str, Any]],
    ) -> StrategyResult:
        tools = self._tool_engine.list_tools()
        messages = list(context)
        messages.append({"role": "user", "content": task_description})

        tool_calls_made = 0

        for iteration in range(self._max_iterations):
            response = await self._llm.complete(messages=messages, tools=tools if tools else None)

            # If no tool calls, we have the final answer
            if not response.tool_calls:
                return StrategyResult(
                    content=response.content,
                    tool_calls_made=tool_calls_made,
                    iterations=iteration + 1,
                )

            # Execute tool calls
            messages.append({"role": "assistant", "content": response.content or ""})

            for tool_call in response.tool_calls:
                result = await self._tool_engine.execute(tool_call.name, tool_call.arguments)
                tool_calls_made += 1

                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_call.name}: {result.data if result.success else result.error}",
                })

        # Max iterations reached
        return StrategyResult(
            content=response.content if response else "Max iterations reached",
            tool_calls_made=tool_calls_made,
            iterations=self._max_iterations,
        )


class CoTStrategy(ExecutionStrategy):
    """Chain-of-thought reasoning without tools."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def execute(
        self,
        task_description: str,
        context: list[dict[str, Any]],
    ) -> StrategyResult:
        messages = list(context)
        messages.append({
            "role": "user",
            "content": f"Think step by step about this task:\n\n{task_description}",
        })

        response = await self._llm.complete(messages=messages)

        return StrategyResult(
            content=response.content,
            tool_calls_made=0,
            iterations=1,
        )


class PlanExecuteStrategy(ExecutionStrategy):
    """Plan first, then execute step by step."""

    def __init__(self, llm: LLMProvider, tool_engine: Any, max_iterations: int = 10):
        self._llm = llm
        self._tool_engine = tool_engine
        self._max_iterations = max_iterations

    async def execute(
        self,
        task_description: str,
        context: list[dict[str, Any]],
    ) -> StrategyResult:
        tools = self._tool_engine.list_tools()

        # Step 1: Generate plan
        plan_messages = list(context)
        plan_messages.append({
            "role": "user",
            "content": (
                f"Create a step-by-step plan for this task. "
                f"Output each step on a new line numbered 1, 2, 3, etc.\n\n"
                f"Task: {task_description}"
            ),
        })

        plan_response = await self._llm.complete(messages=plan_messages)
        plan_steps = [
            line.strip() for line in plan_response.content.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

        # Step 2: Execute each step
        results = []
        tool_calls_made = 0

        for i, step in enumerate(plan_steps[:self._max_iterations]):
            step_messages = list(context)
            step_messages.append({
                "role": "user",
                "content": f"Execute this step: {step}\n\nPrevious results: {results}",
            })

            step_response = await self._llm.complete(messages=step_messages, tools=tools if tools else None)

            if step_response.tool_calls:
                for tc in step_response.tool_calls:
                    result = await self._tool_engine.execute(tc.name, tc.arguments)
                    tool_calls_made += 1
                    results.append(f"Step {i+1}: {result.data if result.success else result.error}")
            else:
                results.append(f"Step {i+1}: {step_response.content}")

        return StrategyResult(
            content="\n".join(results),
            tool_calls_made=tool_calls_made,
            iterations=len(plan_steps),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_strategy.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/strategy.py tests/test_agents/test_strategy.py
git commit -m "feat: agent strategies — ReAct, CoT, PlanExecute"
```

---

## Task 3: Agent Message Bus

**Files:**
- Create: `src/agents/message_bus.py`
- Create: `tests/test_agents/test_message_bus.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_message_bus.py
import asyncio
import pytest


@pytest.mark.asyncio
async def test_agent_message_bus_send_receive():
    from src.agents.message_bus import AgentMessageBus

    bus = AgentMessageBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.register_agent("agent-1", handler)
    await bus.start()

    await bus.send(
        sender="agent-0",
        receiver="agent-1",
        msg_type="task",
        payload={"description": "do something"},
    )

    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received) == 1
    assert received[0].payload["description"] == "do something"


@pytest.mark.asyncio
async def test_agent_message_bus_broadcast():
    from src.agents.message_bus import AgentMessageBus

    bus = AgentMessageBus()
    received_a, received_b = [], []

    async def handler_a(msg):
        received_a.append(msg)

    async def handler_b(msg):
        received_b.append(msg)

    bus.register_agent("a", handler_a)
    bus.register_agent("b", handler_b)
    await bus.start()

    await bus.broadcast(sender="coordinator", msg_type="heartbeat", payload={})

    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received_a) == 1
    assert len(received_b) == 1


@pytest.mark.asyncio
async def test_agent_message_bus_unregister():
    from src.agents.message_bus import AgentMessageBus

    bus = AgentMessageBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.register_agent("agent-1", handler)
    bus.unregister_agent("agent-1")
    await bus.start()

    await bus.send(sender="s", receiver="agent-1", msg_type="task", payload={})
    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents/test_message_bus.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement message bus**

```python
# src/agents/message_bus.py
import asyncio
import time
import uuid
from typing import Any, Callable, Awaitable

from src.core.events import EventBus, Message, MessageType


class AgentMessageBus:
    """Agent-specific message bus wrapping Core EventBus."""

    def __init__(self):
        self._event_bus = EventBus()
        self._agents: dict[str, Callable] = {}

    def register_agent(self, agent_id: str, handler: Callable[[Message], Awaitable[None]]):
        """Register an agent to receive messages."""
        self._agents[agent_id] = handler
        self._event_bus.subscribe(agent_id, handler)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        self._agents.pop(agent_id, None)
        self._event_bus.unsubscribe(agent_id)

    async def start(self):
        """Start the message bus."""
        await self._event_bus.start()

    async def stop(self):
        """Stop the message bus."""
        await self._event_bus.stop()

    async def send(
        self,
        sender: str,
        receiver: str,
        msg_type: str,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ):
        """Send a point-to-point message."""
        message = Message(
            sender=sender,
            receiver=receiver,
            type=MessageType(msg_type),
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id or str(uuid.uuid4()),
        )
        await self._event_bus.publish(message)

    async def broadcast(
        self,
        sender: str,
        msg_type: str,
        payload: dict[str, Any],
    ):
        """Broadcast a message to all agents."""
        message = Message(
            sender=sender,
            receiver="broadcast",
            type=MessageType(msg_type),
            payload=payload,
            timestamp=time.time(),
            correlation_id=str(uuid.uuid4()),
        )
        await self._event_bus.publish(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_message_bus.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/message_bus.py tests/test_agents/test_message_bus.py
git commit -m "feat: AgentMessageBus wrapping Core EventBus with agent-specific routing"
```

---

## Task 4: Base Agent + PlannerAgent

**Files:**
- Create: `src/agents/base.py`
- Create: `src/agents/planner.py`
- Create: `tests/test_agents/test_planner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_planner.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.llm.provider import LLMResponse, TokenUsage


@pytest.mark.asyncio
async def test_planner_agent_decomposes_task():
    """PlannerAgent decomposes a task into subtasks."""
    from src.agents.planner import PlannerAgent
    from src.agents.memory import ShortTermMemory

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Subtasks:\n1. Read the data file\n2. Analyze trends\n3. Generate report",
        tool_calls=None,
        usage=TokenUsage(input_tokens=20, output_tokens=30),
        model="test",
    ))

    agent = PlannerAgent(
        id="planner-1",
        llm=mock_llm,
        memory=ShortTermMemory(),
    )

    result = await agent.run(task_description="Analyze sales data and create report")
    assert result.success is True
    assert len(result.subtasks) == 3
    assert "Read" in result.subtasks[0]


@pytest.mark.asyncio
async def test_planner_agent_state_transitions():
    """PlannerAgent transitions through states correctly."""
    from src.agents.planner import PlannerAgent
    from src.agents.base import AgentState
    from src.agents.memory import ShortTermMemory

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="1. Step one\n2. Step two",
        tool_calls=None,
        usage=TokenUsage(input_tokens=5, output_tokens=10),
        model="test",
    ))

    agent = PlannerAgent(
        id="planner-1",
        llm=mock_llm,
        memory=ShortTermMemory(),
    )

    assert agent.state == AgentState.IDLE
    await agent.run(task_description="Test task")
    assert agent.state == AgentState.COMPLETED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents/test_planner.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement base agent and planner**

```python
# src/agents/base.py
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
    """Base class for all agents."""

    def __init__(
        self,
        id: str,
        role: str = "",
        goal: str = "",
        capabilities: list[str] | None = None,
        memory: ShortTermMemory | None = None,
    ):
        self.id = id
        self.role = role
        self.goal = goal
        self.capabilities = capabilities or []
        self.memory = memory or ShortTermMemory()
        self.state = AgentState.IDLE

    def transition(self, new_state: AgentState):
        """Transition to a new state."""
        self.state = new_state

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the agent's primary task."""
        ...
```

```python
# src/agents/planner.py
from typing import Any

from src.agents.base import BaseAgent, AgentState, PlannerResult
from src.agents.memory import ShortTermMemory
from src.llm.provider import LLMProvider


class PlannerAgent(BaseAgent):
    """Receives user tasks, decomposes into subtasks."""

    def __init__(
        self,
        id: str,
        llm: LLMProvider,
        memory: ShortTermMemory | None = None,
    ):
        super().__init__(id=id, role="planner", memory=memory)
        self._llm = llm

    async def run(self, task_description: str, **kwargs) -> PlannerResult:
        """Decompose a task into subtasks using LLM."""
        self.transition(AgentState.PLANNING)
        self.memory.add({"role": "user", "content": task_description})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task planning agent. Your job is to decompose complex tasks "
                    "into clear, actionable subtasks. Output each subtask on a numbered line."
                ),
            },
            {"role": "user", "content": task_description},
        ]

        response = await self._llm.complete(messages=messages)

        # Parse subtasks from numbered lines
        subtasks = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

        self.memory.add({"role": "assistant", "content": response.content})
        self.transition(AgentState.COMPLETED)

        return PlannerResult(
            success=True,
            subtasks=subtasks,
            content=response.content,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents/test_planner.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/base.py src/agents/planner.py tests/test_agents/test_planner.py
git commit -m "feat: BaseAgent with state machine and PlannerAgent with LLM decomposition"
```

---

## Task 5: ExecutorAgent + CriticAgent

**Files:**
- Create: `src/agents/executor.py`
- Create: `src/agents/critic.py`

- [ ] **Step 1: Write implementation**

```python
# src/agents/executor.py
from typing import Any

from src.agents.base import BaseAgent, AgentState, Result
from src.agents.memory import ShortTermMemory
from src.agents.strategy import ExecutionStrategy


class ExecutorAgent(BaseAgent):
    """Executes subtasks by calling Tools/Skills via a strategy."""

    def __init__(
        self,
        id: str,
        strategy: ExecutionStrategy,
        memory: ShortTermMemory | None = None,
    ):
        super().__init__(id=id, role="executor", memory=memory)
        self._strategy = strategy

    async def run(self, task_description: str, context: list[dict] | None = None, **kwargs) -> Result:
        """Execute a task using the configured strategy."""
        self.transition(AgentState.EXECUTING)
        self.memory.add({"role": "user", "content": task_description})

        result = await self._strategy.execute(
            task_description=task_description,
            context=context or self.memory.get_all(),
        )

        self.memory.add({"role": "assistant", "content": result.content})
        self.transition(AgentState.COMPLETED)

        return Result(
            task_id=self.id,
            success=True,
            data={"content": result.content, "tool_calls": result.tool_calls_made},
        )
```

```python
# src/agents/critic.py
from typing import Any

from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory
from src.llm.provider import LLMProvider


class CriticAgent(BaseAgent):
    """Reviews executor output and provides feedback."""

    def __init__(
        self,
        id: str,
        llm: LLMProvider,
        memory: ShortTermMemory | None = None,
    ):
        super().__init__(id=id, role="critic", memory=memory)
        self._llm = llm

    async def run(self, content_to_review: str, original_task: str = "", **kwargs) -> dict[str, Any]:
        """Review content and provide feedback."""
        self.transition(AgentState.REVIEWING)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code and content review agent. Review the work for: "
                    "1) Correctness 2) Performance 3) Security 4) Maintainability. "
                    "Output APPROVED if the work is good, or REJECTED with specific feedback."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original task: {original_task}\n\n"
                    f"Work to review:\n{content_to_review}"
                ),
            },
        ]

        response = await self._llm.complete(messages=messages)

        approved = "APPROVED" in response.content.upper()
        self.transition(AgentState.COMPLETED)

        return {
            "approved": approved,
            "feedback": response.content,
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/executor.py src/agents/critic.py
git commit -m "feat: ExecutorAgent (strategy-based) and CriticAgent (LLM review)"
```

---

## Task 6: RetrieverAgent + CoordinatorAgent

**Files:**
- Create: `src/agents/retriever.py`
- Create: `src/agents/coordinator.py`

- [ ] **Step 1: Write implementation**

```python
# src/agents/retriever.py
from typing import Any

from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory


class RetrieverAgent(BaseAgent):
    """Runs RAG retrieval and provides knowledge context."""

    def __init__(
        self,
        id: str,
        retrieval_pipeline: Any,
        memory: ShortTermMemory | None = None,
    ):
        super().__init__(id=id, role="retriever", memory=memory)
        self._retrieval = retrieval_pipeline

    async def run(self, query: str, top_k: int = 5, **kwargs) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge for a query."""
        self.transition(AgentState.EXECUTING)

        results = await self._retrieval.retrieve(query, top_k=top_k)

        knowledge = [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in results
        ]

        self.memory.add({"role": "user", "content": query})
        self.memory.add({"role": "assistant", "content": f"Found {len(knowledge)} results"})
        self.transition(AgentState.COMPLETED)

        return knowledge
```

```python
# src/agents/coordinator.py
from typing import Any

from src.agents.base import BaseAgent, AgentState
from src.agents.memory import ShortTermMemory
from src.agents.message_bus import AgentMessageBus


class CoordinatorAgent(BaseAgent):
    """Manages inter-agent communication and task coordination."""

    def __init__(
        self,
        id: str,
        message_bus: AgentMessageBus,
        memory: ShortTermMemory | None = None,
    ):
        super().__init__(id=id, role="coordinator", memory=memory)
        self._bus = message_bus
        self._task_status: dict[str, str] = {}

    async def run(self, tasks: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """Coordinate execution of multiple tasks across agents."""
        self.transition(AgentState.EXECUTING)

        results = {}
        for task in tasks:
            task_id = task.get("id", "")
            agent_id = task.get("assigned_to", "")
            description = task.get("description", "")

            self._task_status[task_id] = "pending"

            await self._bus.send(
                sender=self.id,
                receiver=agent_id,
                msg_type="task",
                payload={"task_id": task_id, "description": description},
                correlation_id=task_id,
            )

            self._task_status[task_id] = "dispatched"
            results[task_id] = {"status": "dispatched", "agent": agent_id}

        self.transition(AgentState.COMPLETED)
        return {"tasks_dispatched": len(results), "status": results}

    async def handle_result(self, message: Any):
        """Handle result messages from executor agents."""
        task_id = message.payload.get("task_id", message.correlation_id)
        self._task_status[task_id] = "completed"
        self.memory.add({"role": "system", "content": f"Task {task_id} completed"})
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/retriever.py src/agents/coordinator.py
git commit -m "feat: RetrieverAgent (RAG) and CoordinatorAgent (task dispatch)"
```

---

## Task 7: MCP Protocol

**Files:**
- Create: `src/mcp/__init__.py`
- Create: `src/mcp/protocol.py`
- Create: `tests/test_mcp/__init__.py`
- Create: `tests/test_mcp/test_protocol.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_protocol.py
import pytest


def test_mcp_request_creation():
    from src.mcp.protocol import MCPRequest

    req = MCPRequest(
        method="tools/call",
        params={"name": "read_file", "arguments": {"path": "/tmp"}},
        request_id="req-1",
    )
    assert req.method == "tools/call"
    assert req.request_id == "req-1"


def test_mcp_response_success():
    from src.mcp.protocol import MCPResponse

    resp = MCPResponse(
        request_id="req-1",
        result={"content": "file contents"},
        error=None,
    )
    assert resp.result is not None
    assert resp.error is None


def test_mcp_response_error():
    from src.mcp.protocol import MCPResponse, MCPError

    resp = MCPResponse(
        request_id="req-1",
        result=None,
        error=MCPError(code=-32600, message="Invalid request"),
    )
    assert resp.error is not None
    assert resp.error.code == -32600


def test_mcp_error_creation():
    from src.mcp.protocol import MCPError

    err = MCPError(code=-32601, message="Method not found", data={"method": "foo"})
    assert err.code == -32601
    assert err.data == {"method": "foo"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcp/test_protocol.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement protocol**

```python
# src/mcp/__init__.py
```

```python
# src/mcp/protocol.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPError:
    code: int
    message: str
    data: dict[str, Any] | None = None


@dataclass
class MCPRequest:
    method: str
    params: dict[str, Any]
    request_id: str


@dataclass
class MCPResponse:
    request_id: str
    result: dict[str, Any] | None = None
    error: MCPError | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp/test_protocol.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp/__init__.py src/mcp/protocol.py tests/test_mcp/__init__.py tests/test_mcp/test_protocol.py
git commit -m "feat: MCP protocol — MCPRequest, MCPResponse, MCPError dataclasses"
```

---

## Task 8: MCP Server + Decorators

**Files:**
- Create: `src/mcp/server.py`
- Create: `tests/test_mcp/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_server.py
import pytest


@pytest.mark.asyncio
async def test_mcp_server_handle_tools_list():
    from src.mcp.server import MCPServer, mcp_server, tool, resource
    from src.mcp.protocol import MCPRequest

    @mcp_server(name="test", version="1.0")
    class TestServer:
        @tool(description="Echo input")
        async def echo(self, text: str) -> str:
            return text

    server = TestServer()
    req = MCPRequest(method="tools/list", params={}, request_id="1")
    resp = await server.handle_request(req)

    assert resp.result is not None
    assert len(resp.result["tools"]) == 1
    assert resp.result["tools"][0]["name"] == "echo"


@pytest.mark.asyncio
async def test_mcp_server_handle_tool_call():
    from src.mcp.server import MCPServer, mcp_server, tool
    from src.mcp.protocol import MCPRequest

    @mcp_server(name="calc", version="1.0")
    class CalcServer:
        @tool(description="Add numbers")
        async def add(self, a: int, b: int) -> int:
            return a + b

    server = CalcServer()
    req = MCPRequest(
        method="tools/call",
        params={"name": "add", "arguments": {"a": 2, "b": 3}},
        request_id="2",
    )
    resp = await server.handle_request(req)

    assert resp.result is not None
    assert resp.result["content"] == 5


@pytest.mark.asyncio
async def test_mcp_server_unknown_method():
    from src.mcp.server import mcp_server
    from src.mcp.protocol import MCPRequest

    @mcp_server(name="test", version="1.0")
    class TestServer:
        pass

    server = TestServer()
    req = MCPRequest(method="unknown/method", params={}, request_id="3")
    resp = await server.handle_request(req)

    assert resp.error is not None
    assert resp.error.code == -32601
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcp/test_server.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement server**

```python
# src/mcp/server.py
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from src.mcp.protocol import MCPRequest, MCPResponse, MCPError


def tool(description: str = "", parameters: dict | None = None):
    """Decorator to mark a method as an MCP tool."""
    def decorator(func):
        func._mcp_tool = True
        func._mcp_tool_description = description
        func._mcp_tool_parameters = parameters or _infer_parameters(func)
        return func
    return decorator


def resource(uri_scheme: str = ""):
    """Decorator to mark a method as an MCP resource."""
    def decorator(func):
        func._mcp_resource = True
        func._mcp_uri_scheme = uri_scheme
        return func
    return decorator


def _infer_parameters(func: Callable) -> dict[str, Any]:
    """Infer JSON Schema parameters from function signature."""
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        prop: dict[str, Any] = {}
        if param.annotation == str:
            prop["type"] = "string"
        elif param.annotation == int:
            prop["type"] = "integer"
        elif param.annotation == float:
            prop["type"] = "number"
        elif param.annotation == bool:
            prop["type"] = "boolean"
        else:
            prop["type"] = "string"

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def mcp_server(name: str, version: str = "1.0"):
    """Class decorator to create an MCP server."""
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._mcp_name = name
            self._mcp_version = version

            self._tools = {}
            self._resources = {}

            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if attr and getattr(attr, "_mcp_tool", False):
                    self._tools[attr_name] = {
                        "handler": attr,
                        "description": attr._mcp_tool_description,
                        "parameters": attr._mcp_tool_parameters,
                    }
                if attr and getattr(attr, "_mcp_resource", False):
                    self._resources[attr_name] = {
                        "handler": attr,
                        "uri_scheme": attr._mcp_uri_scheme,
                    }

        cls.__init__ = new_init

        async def handle_request(self, request: MCPRequest) -> MCPResponse:
            if request.method == "tools/list":
                tools_list = [
                    {"name": name, "description": info["description"], "parameters": info["parameters"]}
                    for name, info in self._tools.items()
                ]
                return MCPResponse(request_id=request.request_id, result={"tools": tools_list})

            elif request.method == "tools/call":
                tool_name = request.params.get("name")
                arguments = request.params.get("arguments", {})

                if tool_name not in self._tools:
                    return MCPResponse(
                        request_id=request.request_id,
                        error=MCPError(code=-32601, message=f"Tool not found: {tool_name}"),
                    )

                try:
                    result = await self._tools[tool_name]["handler"](**arguments)
                    return MCPResponse(
                        request_id=request.request_id,
                        result={"content": result},
                    )
                except Exception as e:
                    return MCPResponse(
                        request_id=request.request_id,
                        error=MCPError(code=-32603, message=str(e)),
                    )

            elif request.method == "resources/list":
                resources_list = [
                    {"name": name, "uri_scheme": info["uri_scheme"]}
                    for name, info in self._resources.items()
                ]
                return MCPResponse(request_id=request.request_id, result={"resources": resources_list})

            else:
                return MCPResponse(
                    request_id=request.request_id,
                    error=MCPError(code=-32601, message=f"Method not found: {request.method}"),
                )

        cls.handle_request = handle_request
        return cls

    return decorator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp/test_server.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat: MCP server with @tool/@resource decorators and request handling"
```

---

## Task 9: MCP Built-in Servers + Client

**Files:**
- Create: `src/mcp/servers/__init__.py`
- Create: `src/mcp/servers/filesystem.py`
- Create: `src/mcp/servers/web.py`
- Create: `src/mcp/client.py`

- [ ] **Step 1: Write implementation**

```python
# src/mcp/servers/__init__.py
```

```python
# src/mcp/servers/filesystem.py
import os
from pathlib import Path

from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="filesystem", version="1.0")
class FileSystemServer:
    """MCP server for file system operations."""

    def __init__(self, root_dir: str = "."):
        self._root = Path(root_dir)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self._root / p
        return p.resolve()

    @tool(description="Read file contents")
    async def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    @tool(description="Write content to a file")
    async def write_file(self, path: str, content: str) -> bool:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True

    @tool(description="List directory contents")
    async def list_dir(self, path: str = ".") -> list[str]:
        p = self._resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return [str(item.name) for item in p.iterdir()]

    @tool(description="Search for files matching a pattern")
    async def search_files(self, pattern: str, directory: str = ".") -> list[str]:
        import fnmatch
        d = self._resolve(directory)
        results = []
        for root, dirs, files in os.walk(d):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    results.append(os.path.join(root, f))
        return sorted(results)

    @resource(uri_scheme="file")
    async def file_resource(self, uri: str) -> str:
        path = uri.replace("file://", "")
        return await self.read_file(path)
```

```python
# src/mcp/servers/web.py
import httpx

from src.mcp.server import mcp_server, tool, resource


@mcp_server(name="web", version="1.0")
class WebServer:
    """MCP server for web operations."""

    def __init__(self, allowed_domains: list[str] | None = None):
        self._allowed = allowed_domains

    def _check_domain(self, url: str):
        if self._allowed and "*" not in self._allowed:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if not any(domain.endswith(d) for d in self._allowed):
                raise PermissionError(f"Domain not allowed: {domain}")

    @tool(description="Fetch content from a URL")
    async def fetch_url(self, url: str, method: str = "GET") -> dict:
        self._check_domain(url)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url)
            return {"status": resp.status_code, "body": resp.text}

    @tool(description="Search the web (stub)")
    async def search_web(self, query: str) -> list[dict]:
        # Stub — real implementation would use a search API
        return [{"title": f"Result for: {query}", "url": "https://example.com"}]

    @resource(uri_scheme="https")
    async def https_resource(self, uri: str) -> str:
        result = await self.fetch_url(uri)
        return result["body"]
```

```python
# src/mcp/client.py
import uuid
from typing import Any

from src.mcp.protocol import MCPRequest, MCPResponse
from src.mcp.server import MCPServer


class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(self):
        self._servers: dict[str, Any] = {}

    def register_server(self, name: str, server: Any):
        """Register an MCP server."""
        self._servers[name] = server

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific server."""
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")

        request = MCPRequest(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=str(uuid.uuid4()),
        )

        response = await server.handle_request(request)

        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")

        return response.result.get("content") if response.result else None

    async def list_tools(self, server_name: str) -> list[dict]:
        """List tools available on a server."""
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")

        request = MCPRequest(method="tools/list", params={}, request_id=str(uuid.uuid4()))
        response = await server.handle_request(request)

        if response.error:
            raise RuntimeError(f"MCP error: {response.error.message}")

        return response.result.get("tools", []) if response.result else []
```

- [ ] **Step 2: Commit**

```bash
git add src/mcp/servers/__init__.py src/mcp/servers/filesystem.py src/mcp/servers/web.py src/mcp/client.py
git commit -m "feat: MCP built-in servers (filesystem, web) and MCPClient"
```

---

## Task 10: Skills Engine + Loader

**Files:**
- Create: `src/skills/__init__.py`
- Create: `src/skills/engine.py`
- Create: `src/skills/loader.py`
- Create: `tests/test_skills/__init__.py`
- Create: `tests/test_skills/test_engine.py`
- Create: `tests/test_skills/test_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skills/test_engine.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.llm.provider import LLMResponse, TokenUsage


@pytest.mark.asyncio
async def test_skill_engine_tool_call_step():
    """SkillEngine executes tool_call steps."""
    from src.skills.engine import SkillEngine, SkillContext

    mock_tool_engine = MagicMock()
    mock_tool_engine.execute = AsyncMock(return_value=MagicMock(success=True, data="file content"))
    mock_tool_engine.list_tools = MagicMock(return_value=[])

    mock_llm = MagicMock()

    engine = SkillEngine(tool_engine=mock_tool_engine, llm=mock_llm)

    skill = {
        "name": "test_skill",
        "steps": [
            {
                "id": "read",
                "type": "tool_call",
                "tool": "read_file",
                "params": {"path": "{{ input.file_path }}"},
                "output": "content",
            }
        ],
    }

    result = await engine.execute_skill(skill, {"file_path": "/tmp/test.txt"})
    assert result["content"] == "file content"


@pytest.mark.asyncio
async def test_skill_engine_llm_reasoning_step():
    """SkillEngine executes llm_reasoning steps."""
    from src.skills.engine import SkillEngine

    mock_tool_engine = MagicMock()
    mock_tool_engine.list_tools = MagicMock(return_value=[])

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Analysis: the code looks good",
        tool_calls=None,
        usage=TokenUsage(input_tokens=10, output_tokens=15),
        model="test",
    ))

    engine = SkillEngine(tool_engine=mock_tool_engine, llm=mock_llm)

    skill = {
        "name": "analyze",
        "steps": [
            {
                "id": "think",
                "type": "llm_reasoning",
                "prompt": "Analyze this: {{ input.text }}",
                "output": "analysis",
            }
        ],
    }

    result = await engine.execute_skill(skill, {"text": "some code"})
    assert "good" in result["analysis"]
```

```python
# tests/test_skills/test_loader.py
import pytest
import yaml
from pathlib import Path


def test_loader_loads_yaml(tmp_path):
    from src.skills.loader import SkillLoader

    skill_def = {
        "name": "test_skill",
        "description": "A test skill",
        "required_tools": ["read_file"],
        "steps": [
            {
                "id": "step1",
                "type": "tool_call",
                "tool": "read_file",
                "params": {"path": "{{ input.path }}"},
                "output": "content",
            }
        ],
    }

    skill_file = tmp_path / "test_skill.yaml"
    skill_file.write_text(yaml.dump(skill_def))

    loader = SkillLoader(str(tmp_path))
    loaded = loader.load("test_skill")

    assert loaded["name"] == "test_skill"
    assert len(loaded["steps"]) == 1


def test_loader_lists_available_skills(tmp_path):
    from src.skills.loader import SkillLoader

    for name in ["skill_a", "skill_b"]:
        skill_file = tmp_path / f"{name}.yaml"
        skill_file.write_text(yaml.dump({"name": name, "steps": []}))

    loader = SkillLoader(str(tmp_path))
    available = loader.list_skills()

    assert set(available) == {"skill_a", "skill_b"}


def test_loader_missing_skill(tmp_path):
    from src.skills.loader import SkillLoader

    loader = SkillLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_skills/ -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement engine and loader**

```python
# src/skills/__init__.py
```

```python
# src/skills/engine.py
import re
from typing import Any

from src.llm.provider import LLMProvider


class SkillContext:
    """Key-value context for skill execution."""

    def __init__(self, input_data: dict[str, Any]):
        self._data: dict[str, Any] = {"input": input_data}
        self.iteration = 0

    def set(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def resolve_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Resolve template variables in params."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_template(value)
            else:
                resolved[key] = value
        return resolved

    def _resolve_template(self, template: str) -> str:
        """Resolve {{ variable.path }} templates."""
        def replacer(match):
            path = match.group(1).strip()
            parts = path.split(".")
            value = self._data
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, match.group(0))
                else:
                    return match.group(0)
            return str(value)

        return re.sub(r"\{\{(.*?)\}\}", replacer, template)


class SkillEngine:
    """Executes skills step by step."""

    def __init__(self, tool_engine: Any, llm: LLMProvider):
        self._tool_engine = tool_engine
        self._llm = llm

    async def execute_skill(
        self,
        skill_def: dict[str, Any],
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a skill definition with given input."""
        context = SkillContext(input_data)

        for step in skill_def.get("steps", []):
            step_type = step["type"]

            if step_type == "tool_call":
                params = context.resolve_params(step.get("params", {}))
                result = await self._tool_engine.execute(step["tool"], params)
                context.set(step["output"], result.data if result.success else result.error)

            elif step_type == "llm_reasoning":
                prompt = context._resolve_template(step["prompt"])
                response = await self._llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                )
                context.set(step["output"], response.content)

            elif step_type == "conditional":
                condition = context._resolve_template(step["condition"])
                if condition.lower() in ("true", "1", "yes"):
                    for sub_step in step.get("then_steps", []):
                        await self._execute_step(sub_step, context)
                else:
                    for sub_step in step.get("else_steps", []):
                        await self._execute_step(sub_step, context)

            elif step_type == "loop":
                max_iter = step.get("max_iterations", 10)
                for _ in range(max_iter):
                    context.iteration += 1
                    for sub_step in step.get("body_steps", []):
                        await self._execute_step(sub_step, context)

        return context._data

    async def _execute_step(self, step: dict[str, Any], context: SkillContext):
        """Execute a single step."""
        step_type = step["type"]

        if step_type == "tool_call":
            params = context.resolve_params(step.get("params", {}))
            result = await self._tool_engine.execute(step["tool"], params)
            context.set(step["output"], result.data if result.success else result.error)

        elif step_type == "llm_reasoning":
            prompt = context._resolve_template(step["prompt"])
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
            )
            context.set(step["output"], response.content)
```

```python
# src/skills/loader.py
import yaml
from pathlib import Path
from typing import Any


class SkillLoader:
    """Loads skill definitions from YAML files."""

    def __init__(self, definitions_dir: str):
        self._dir = Path(definitions_dir)

    def load(self, name: str) -> dict[str, Any]:
        """Load a skill definition by name."""
        file_path = self._dir / f"{name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Skill definition not found: {name}")

        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def list_skills(self) -> list[str]:
        """List all available skill names."""
        return [
            f.stem for f in self._dir.glob("*.yaml")
            if f.is_file()
        ]

    def load_all(self) -> dict[str, dict[str, Any]]:
        """Load all skill definitions."""
        return {name: self.load(name) for name in self.list_skills()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_skills/ -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/skills/__init__.py src/skills/engine.py src/skills/loader.py tests/test_skills/__init__.py tests/test_skills/test_engine.py tests/test_skills/test_loader.py
git commit -m "feat: SkillEngine (tool_call, llm_reasoning, conditional, loop) and YAML SkillLoader"
```

---

## Task 11: Skill Definitions

**Files:**
- Create: `src/skills/definitions/code_review.yaml`
- Create: `src/skills/definitions/data_analysis.yaml`

- [ ] **Step 1: Create code_review.yaml**

```yaml
# src/skills/definitions/code_review.yaml
name: code_review
description: "审查代码质量和潜在问题"
required_tools:
  - read_file
  - lint_code
  - search_knowledge
steps:
  - id: read_code
    type: tool_call
    tool: read_file
    params:
      path: "{{ input.file_path }}"
    output: code_content

  - id: lint
    type: tool_call
    tool: lint_code
    params:
      code: "{{ code_content }}"
      language: "{{ input.language }}"
    output: diagnostics

  - id: analyze
    type: llm_reasoning
    prompt: |
      你是代码审查专家。以下是代码和静态分析结果：
      ```
      {{ code_content }}
      ```
      静态分析: {{ diagnostics }}

      请从以下维度审查：
      1. 代码正确性
      2. 性能问题
      3. 安全隐患
      4. 可维护性

      输出 JSON 格式的审查报告。
    output: review_report

  - id: check_best_practices
    type: tool_call
    tool: search_knowledge
    params:
      query: "{{ input.language }} best practices for the patterns used in this code"
      top_k: 3
    output: best_practices

  - id: final_report
    type: llm_reasoning
    prompt: |
      结合审查报告和最佳实践，生成最终审查意见：
      审查报告: {{ review_report }}
      最佳实践参考: {{ best_practices }}
    output: final_review
```

- [ ] **Step 2: Create data_analysis.yaml**

```yaml
# src/skills/definitions/data_analysis.yaml
name: data_analysis
description: "对数据集进行探索性分析，生成报告"
required_tools:
  - csv_to_dict
  - run_python
  - write_file
steps:
  - id: parse_data
    type: tool_call
    tool: csv_to_dict
    params:
      csv_text: "{{ input.csv_content }}"
    output: parsed_data

  - id: analyze
    type: llm_reasoning
    prompt: |
      你是一个数据分析师。请分析以下数据并生成 Python 代码来执行探索性分析：
      数据样本（前5行）: {{ parsed_data }}
      分析目标: {{ input.analysis_goal }}

      生成的代码应该：
      1. 计算基本统计量
      2. 识别趋势和异常
      3. 输出分析结果为 JSON 格式

      只输出 Python 代码，不要其他内容。
    output: analysis_code

  - id: execute_analysis
    type: tool_call
    tool: run_python
    params:
      code: "{{ analysis_code }}"
      timeout: 60
    output: analysis_result

  - id: generate_report
    type: llm_reasoning
    prompt: |
      基于数据分析结果，生成一份结构化的分析报告：
      分析结果: {{ analysis_result }}
      原始目标: {{ input.analysis_goal }}

      报告应包含：摘要、关键发现、数据趋势、建议。
    output: report

  - id: save_report
    type: tool_call
    tool: write_file
    params:
      path: "{{ input.output_path }}"
      content: "{{ report }}"
    output: save_result
```

- [ ] **Step 3: Commit**

```bash
git add src/skills/definitions/
git commit -m "feat: skill definitions — code_review.yaml and data_analysis.yaml"
```

---

## Verification Checklist

After completing all tasks, run:

```bash
# All tests pass
pytest tests/ -v

# No import errors
python -c "
from src.agents.base import BaseAgent, AgentState, Task, Result
from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.agents.retriever import RetrieverAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.memory import ShortTermMemory, LongTermMemory
from src.agents.strategy import ReActStrategy, CoTStrategy, PlanExecuteStrategy
from src.agents.message_bus import AgentMessageBus
from src.mcp.protocol import MCPRequest, MCPResponse, MCPError
from src.mcp.server import MCPServer, mcp_server, tool, resource
from src.mcp.client import MCPClient
from src.mcp.servers.filesystem import FileSystemServer
from src.mcp.servers.web import WebServer
from src.skills.engine import SkillEngine
from src.skills.loader import SkillLoader
print('All imports OK')
"

# Full system integration
python -c "
from src.core.config import load_config
from src.core.events import EventBus
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_all_builtin_tools
from src.rag.chunker import RecursiveChunker
from src.agents.memory import ShortTermMemory
from src.mcp.servers.filesystem import FileSystemServer
from src.skills.loader import SkillLoader

r = ToolRegistry()
register_all_builtin_tools(r)
print(f'Tools: {len(r.list_tools())}')

fs = FileSystemServer(root_dir='.')
print(f'MCP filesystem server ready')

print('System integration OK')
"
```
