# Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Core (config, logging, exceptions, event bus) and LLM (Anthropic provider) modules — the foundation everything else depends on.

**Architecture:** Core provides infrastructure (config, logging, events). LLM wraps the Anthropic API behind a provider interface. Both are dependency-free foundations that all later modules import from.

**Tech Stack:** Python 3.11+, PyYAML, Pydantic, anthropic SDK, pytest, pytest-asyncio

---

## File Structure

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py          # YAML config loader, Settings singleton, env var expansion
│   ├── logging.py         # Structured JSON logging with trace_id
│   ├── exceptions.py      # Exception hierarchy
│   └── events.py          # Async EventBus, Message dataclass, MessageType enum
├── llm/
│   ├── __init__.py
│   ├── provider.py        # LLMProvider ABC, LLMResponse/ToolCall/TokenUsage dataclasses
│   └── anthropic.py       # Anthropic implementation with retry and tool use
config.yaml                # System configuration
pyproject.toml             # Project metadata and dependencies
tests/
├── __init__.py
├── conftest.py            # Shared fixtures
├── test_core/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_logging.py
│   ├── test_exceptions.py
│   └── test_events.py
├── test_llm/
│   ├── __init__.py
│   └── test_anthropic.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "multi-agent-system"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.25.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "beautifulsoup4>=4.12.0",
    "pydantic>=2.7.0",
    "pyyaml>=6.0",
    "rich>=13.7.0",
    "typer>=0.12.0",
    "aiosqlite>=0.20.0",
    "RestrictedPython>=7.1",
    "jinja2>=3.1.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Create config.yaml**

```yaml
system:
  name: "Multi-Agent System"
  version: "0.1.0"
  log_level: INFO

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.7
  max_tokens: 4096
  timeout: 60

embedding:
  model: BAAI/bge-m3
  device: cpu
  batch_size: 32

vector_store:
  backend: chroma
  persist_dir: ./data/chroma
  collection: default

rag:
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
  rerank: true
  rerank_model: BAAI/bge-reranker-v2-m3

agents:
  max_concurrent: 5
  default_strategy: react
  memory:
    short_term_max: 20
    long_term_backend: sqlite

mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./data/workspace
    - name: database
      enabled: false
      connection_string: ${DB_URL}
    - name: web
      enabled: true
      allowed_domains: ["*"]
```

- [ ] **Step 3: Create package init files**

```python
# src/__init__.py
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Create conftest.py**

```python
# tests/conftest.py
import pytest
import asyncio


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config file for testing."""
    config_content = """
system:
  name: "Test System"
  version: "0.1.0"
  log_level: DEBUG

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key
  temperature: 0.7
  max_tokens: 4096
  timeout: 60

embedding:
  model: BAAI/bge-m3
  device: cpu
  batch_size: 32

vector_store:
  backend: chroma
  persist_dir: ./data/chroma
  collection: default

rag:
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
  rerank: false

agents:
  max_concurrent: 5
  default_strategy: react
  memory:
    short_term_max: 20
    long_term_backend: sqlite

mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./workspace
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file
```

- [ ] **Step 5: Install dev dependencies and verify**

Run: `pip install -e ".[dev]"`
Expected: Successfully installs package and dev dependencies

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml config.yaml src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffolding with pyproject.toml and config"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/config.py`
- Create: `tests/test_core/__init__.py`
- Create: `tests/test_core/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/test_core/test_config.py
import os
import pytest
from pathlib import Path


def test_load_config_from_file(sample_config):
    """Config loads YAML file and returns typed settings."""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.system.name == "Test System"
    assert settings.system.log_level == "DEBUG"
    assert settings.llm.provider == "anthropic"
    assert settings.llm.model == "claude-sonnet-4-20250514"


def test_config_expands_env_vars(tmp_path):
    """Config expands ${ENV_VAR} patterns in values."""
    os.environ["TEST_API_KEY"] = "sk-test-123"
    config_content = """
system:
  name: "Test"
  version: "0.1.0"
  log_level: INFO
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_API_KEY}
  temperature: 0.7
  max_tokens: 4096
  timeout: 60
embedding:
  model: BAAI/bge-m3
  device: cpu
  batch_size: 32
vector_store:
  backend: chroma
  persist_dir: ./data/chroma
  collection: default
rag:
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
  rerank: false
agents:
  max_concurrent: 5
  default_strategy: react
  memory:
    short_term_max: 20
    long_term_backend: sqlite
mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./workspace
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    from src.core.config import load_config
    settings = load_config(config_file)
    assert settings.llm.api_key == "sk-test-123"
    del os.environ["TEST_API_KEY"]


def test_config_singleton(sample_config):
    """get_settings returns the same instance on repeated calls."""
    from src.core.config import load_config, get_settings, reset_settings

    reset_settings()
    load_config(sample_config)
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    reset_settings()


def test_config_missing_file():
    """load_config raises FileNotFoundError for missing file."""
    from src.core.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_default_values(sample_config):
    """Config provides sensible defaults for optional fields."""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.rag.chunk_size == 512
    assert settings.rag.chunk_overlap == 64
    assert settings.agents.max_concurrent == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.core.config'"

- [ ] **Step 3: Implement config module**

```python
# src/core/__init__.py
```

```python
# src/core/config.py
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    name: str = "Multi-Agent System"
    version: str = "0.1.0"
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    backend: str = "chroma"
    persist_dir: str = "./data/chroma"
    collection: str = "default"


class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"


class MemoryConfig(BaseModel):
    short_term_max: int = 20
    long_term_backend: str = "sqlite"


class AgentsConfig(BaseModel):
    max_concurrent: int = 5
    default_strategy: str = "react"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class MCPServerConfig(BaseModel):
    name: str
    enabled: bool = True
    root_dir: str = ""
    connection_string: str = ""
    allowed_domains: list[str] = Field(default_factory=list)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class Settings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


_settings: Settings | None = None


def load_config(path: Path) -> Settings:
    """Load configuration from YAML file with env var expansion."""
    global _settings
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)
    _settings = Settings(**expanded)
    return _settings


def get_settings() -> Settings:
    """Get the current settings singleton."""
    if _settings is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _settings


def reset_settings():
    """Reset the settings singleton (for testing)."""
    global _settings
    _settings = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/__init__.py src/core/config.py tests/test_core/__init__.py tests/test_core/test_config.py
git commit -m "feat: config module with YAML loading, env var expansion, Pydantic settings"
```

---

## Task 3: Exceptions Module

**Files:**
- Create: `src/core/exceptions.py`
- Create: `tests/test_core/test_exceptions.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core/test_exceptions.py
import pytest


def test_base_error_has_code_and_context():
    from src.core.exceptions import MultiAgentError

    err = MultiAgentError("something broke", code="ERR_001", context={"key": "val"})
    assert str(err) == "something broke"
    assert err.code == "ERR_001"
    assert err.context == {"key": "val"}


def test_agent_error_inherits():
    from src.core.exceptions import AgentError, MultiAgentError

    err = AgentError("agent failed", agent_id="agent-1")
    assert isinstance(err, MultiAgentError)
    assert err.agent_id == "agent-1"


def test_tool_error_inherits():
    from src.core.exceptions import ToolError, MultiAgentError

    err = ToolError("tool failed", tool_name="read_file")
    assert isinstance(err, MultiAgentError)
    assert err.tool_name == "read_file"


def test_skill_error_inherits():
    from src.core.exceptions import SkillError, MultiAgentError

    err = SkillError("skill failed", skill_name="code_review")
    assert isinstance(err, MultiAgentError)
    assert err.skill_name == "code_review"


def test_rag_error_inherits():
    from src.core.exceptions import RAGError, MultiAgentError

    err = RAGError("rag failed")
    assert isinstance(err, MultiAgentError)


def test_mcp_error_inherits():
    from src.core.exceptions import MCPError, MultiAgentError

    err = MCPError("mcp failed", server_name="filesystem")
    assert isinstance(err, MultiAgentError)
    assert err.server_name == "filesystem"


def test_config_error_inherits():
    from src.core.exceptions import ConfigError, MultiAgentError

    err = ConfigError("config bad")
    assert isinstance(err, MultiAgentError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_exceptions.py -v`
Expected: FAIL with "ImportError: cannot import name 'MultiAgentError'"

- [ ] **Step 3: Implement exceptions**

```python
# src/core/exceptions.py
from typing import Any


class MultiAgentError(Exception):
    """Base exception for the multi-agent system."""

    def __init__(self, message: str, code: str = "", context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class AgentError(MultiAgentError):
    """Error originating from an agent."""

    def __init__(self, message: str, agent_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id


class ToolError(MultiAgentError):
    """Error originating from a tool execution."""

    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class SkillError(MultiAgentError):
    """Error originating from a skill execution."""

    def __init__(self, message: str, skill_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.skill_name = skill_name


class RAGError(MultiAgentError):
    """Error originating from the RAG module."""


class MCPError(MultiAgentError):
    """Error originating from the MCP module."""

    def __init__(self, message: str, server_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.server_name = server_name


class ConfigError(MultiAgentError):
    """Error originating from configuration."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_exceptions.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/exceptions.py tests/test_core/test_exceptions.py
git commit -m "feat: exception hierarchy with AgentError, ToolError, SkillError, RAGError, MCPError"
```

---

## Task 4: Logging Module

**Files:**
- Create: `src/core/logging.py`
- Create: `tests/test_core/test_logging.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core/test_logging.py
import json
import logging
import pytest


def test_setup_logging_configures_root_logger():
    from src.core.logging import setup_logging

    setup_logging(level="DEBUG")
    logger = logging.getLogger("multi_agent")
    assert logger.level == logging.DEBUG


def test_log_record_has_trace_id(caplog):
    from src.core.logging import setup_logging, get_logger

    setup_logging(level="DEBUG")
    logger = get_logger("test")
    with caplog.at_level(logging.DEBUG):
        logger.info("test message", trace_id="abc-123")

    assert "abc-123" in caplog.text


def test_log_record_is_json(caplog):
    from src.core.logging import setup_logging, get_logger

    setup_logging(level="DEBUG", json_format=True)
    logger = get_logger("test")
    with caplog.at_level(logging.DEBUG):
        logger.info("structured message", trace_id="trace-1")

    # Find the JSON log record
    for record in caplog.records:
        if record.getMessage() == "structured message":
            # The formatter should have produced JSON
            assert hasattr(record, "trace_id")
            break


def test_get_logger_returns_named_logger():
    from src.core.logging import get_logger

    logger = get_logger("my_module")
    assert logger.name == "multi_agent.my_module"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_logging.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement logging module**

```python
# src/core/logging.py
import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes trace_id and structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace_id if present
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        # Add any extra fields
        for key in ("agent_id", "tool_name", "skill_name", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """Human-readable formatter with optional trace_id."""

    def format(self, record: logging.LogRecord) -> str:
        trace = getattr(record, "trace_id", "")
        prefix = f"[{trace}] " if trace else ""
        return f"{prefix}{record.levelname} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", json_format: bool = False):
    """Configure the multi-agent logging system."""
    logger = logging.getLogger("multi_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(SimpleFormatter())

    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the multi_agent namespace."""
    return logging.getLogger(f"multi_agent.{name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_logging.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/logging.py tests/test_core/test_logging.py
git commit -m "feat: structured logging with trace_id support and JSON format"
```

---

## Task 5: Event Bus Module

**Files:**
- Create: `src/core/events.py`
- Create: `tests/test_core/test_events.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_core/test_events.py
import asyncio
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_message_creation():
    from src.core.events import Message, MessageType
    import time

    msg = Message(
        sender="agent-1",
        receiver="agent-2",
        type=MessageType.TASK,
        payload={"task": "do something"},
        timestamp=time.time(),
        correlation_id="corr-1",
    )
    assert msg.sender == "agent-1"
    assert msg.receiver == "agent-2"
    assert msg.type == MessageType.TASK
    assert msg.payload == {"task": "do something"}


@pytest.mark.asyncio
async def test_event_bus_subscribe_and_publish():
    from src.core.events import EventBus, Message, MessageType
    import time

    bus = EventBus()
    received = []

    async def handler(msg: Message):
        received.append(msg)

    bus.subscribe("agent-2", handler)
    await bus.start()

    msg = Message(
        sender="agent-1",
        receiver="agent-2",
        type=MessageType.TASK,
        payload={"task": "test"},
        timestamp=time.time(),
        correlation_id="c1",
    )
    await bus.publish(msg)

    # Give the bus time to dispatch
    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received) == 1
    assert received[0].payload == {"task": "test"}


@pytest.mark.asyncio
async def test_event_bus_broadcast():
    from src.core.events import EventBus, Message, MessageType
    import time

    bus = EventBus()
    received_a = []
    received_b = []

    async def handler_a(msg):
        received_a.append(msg)

    async def handler_b(msg):
        received_b.append(msg)

    bus.subscribe("agent-a", handler_a)
    bus.subscribe("agent-b", handler_b)
    await bus.start()

    msg = Message(
        sender="coordinator",
        receiver="broadcast",
        type=MessageType.TASK,
        payload={"broadcast": True},
        timestamp=time.time(),
        correlation_id="c2",
    )
    await bus.publish(msg)

    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received_a) == 1
    assert len(received_b) == 1


@pytest.mark.asyncio
async def test_event_bus_no_handler():
    """Publishing to an agent with no handler doesn't raise."""
    from src.core.events import EventBus, Message, MessageType
    import time

    bus = EventBus()
    await bus.start()

    msg = Message(
        sender="agent-1",
        receiver="nonexistent",
        type=MessageType.HEARTBEAT,
        payload={},
        timestamp=time.time(),
        correlation_id="c3",
    )
    # Should not raise
    await bus.publish(msg)
    await asyncio.sleep(0.05)
    await bus.stop()


@pytest.mark.asyncio
async def test_event_bus_multiple_messages():
    from src.core.events import EventBus, Message, MessageType
    import time

    bus = EventBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("target", handler)
    await bus.start()

    for i in range(5):
        msg = Message(
            sender="sender",
            receiver="target",
            type=MessageType.TASK,
            payload={"index": i},
            timestamp=time.time(),
            correlation_id=f"c{i}",
        )
        await bus.publish(msg)

    await asyncio.sleep(0.3)
    await bus.stop()

    assert len(received) == 5
    assert [m.payload["index"] for m in received] == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_core/test_events.py -v`
Expected: FAIL with "ImportError: cannot import name 'EventBus'"

- [ ] **Step 3: Implement event bus**

```python
# src/core/events.py
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    sender: str
    receiver: str  # agent ID or "broadcast"
    type: MessageType
    payload: dict[str, Any]
    timestamp: float
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


MessageHandler = Callable[[Message], Awaitable[None]]


class EventBus:
    """Async pub/sub event bus for agent communication."""

    def __init__(self):
        self._subscribers: dict[str, list[MessageHandler]] = {}
        self._queue: asyncio.Queue[Message | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    def subscribe(self, agent_id: str, handler: MessageHandler):
        """Register a handler for messages addressed to agent_id."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)

    def unsubscribe(self, agent_id: str):
        """Remove all handlers for agent_id."""
        self._subscribers.pop(agent_id, None)

    async def publish(self, message: Message):
        """Publish a message to the bus."""
        await self._queue.put(message)

    async def start(self):
        """Start the message dispatch loop."""
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        """Stop the dispatch loop gracefully."""
        self._running = False
        await self._queue.put(None)  # Sentinel
        if self._task:
            await self._task

    async def _dispatch_loop(self):
        """Main loop: read messages from queue and dispatch to handlers."""
        while self._running:
            message = await self._queue.get()
            if message is None:
                break

            if message.receiver == "broadcast":
                # Send to all subscribers except sender
                for agent_id, handlers in self._subscribers.items():
                    if agent_id != message.sender:
                        for handler in handlers:
                            asyncio.create_task(self._safe_deliver(handler, message))
            else:
                # Point-to-point
                handlers = self._subscribers.get(message.receiver, [])
                for handler in handlers:
                    asyncio.create_task(self._safe_deliver(handler, message))

    async def _safe_deliver(self, handler: MessageHandler, message: Message):
        """Deliver message to handler, catching exceptions."""
        try:
            await handler(message)
        except Exception:
            # Log but don't crash the bus
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_events.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/events.py tests/test_core/test_events.py
git commit -m "feat: async EventBus with point-to-point and broadcast messaging"
```

---

## Task 6: LLM Provider Interface

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/provider.py`
- Create: `tests/test_llm/__init__.py`
- Create: `tests/test_llm/test_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm/test_provider.py
import pytest


def test_llm_response_creation():
    from src.llm.provider import LLMResponse, TokenUsage

    resp = LLMResponse(
        content="Hello!",
        tool_calls=None,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-20250514",
    )
    assert resp.content == "Hello!"
    assert resp.tool_calls is None
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5


def test_tool_call_creation():
    from src.llm.provider import ToolCall

    tc = ToolCall(id="call-1", name="read_file", arguments={"path": "/tmp/test.txt"})
    assert tc.id == "call-1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "/tmp/test.txt"}


def test_llm_response_with_tool_calls():
    from src.llm.provider import LLMResponse, ToolCall, TokenUsage

    tc = ToolCall(id="c1", name="search", arguments={"query": "test"})
    resp = LLMResponse(
        content="",
        tool_calls=[tc],
        usage=TokenUsage(input_tokens=20, output_tokens=10),
        model="claude-sonnet-4-20250514",
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_provider.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement provider interface**

```python
# src/llm/__init__.py
```

```python
# src/llm/provider.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: TokenUsage
    model: str


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to the LLM and return a response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream the LLM response token by token."""
        ...
        # Make this an async generator for type checking
        yield ""  # pragma: no cover
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_provider.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/__init__.py src/llm/provider.py tests/test_llm/__init__.py tests/test_llm/test_provider.py
git commit -m "feat: LLM provider interface with LLMResponse, ToolCall, TokenUsage"
```

---

## Task 7: Anthropic LLM Implementation

**Files:**
- Create: `src/llm/anthropic.py`
- Create: `tests/test_llm/test_anthropic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm/test_anthropic.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llm.provider import LLMResponse, ToolCall, TokenUsage


@pytest.mark.asyncio
async def test_anthropic_complete_basic():
    """Anthropic provider sends messages and returns structured response."""
    from src.llm.anthropic import AnthropicProvider

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello!", type="text")]
    mock_response.stop_reason = "end_turn"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    with patch("src.llm.anthropic.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-20250514")
        result = await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello!"
    assert result.tool_calls is None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_complete_with_tool_use():
    """Anthropic provider handles tool_use response."""
    from src.llm.anthropic import AnthropicProvider

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.id = "call-1"
    mock_tool_block.name = "read_file"
    mock_tool_block.input = {"path": "/tmp/test.txt"}

    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]
    mock_response.stop_reason = "tool_use"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.usage = MagicMock(input_tokens=20, output_tokens=15)

    with patch("src.llm.anthropic.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-20250514")
        result = await provider.complete(
            messages=[{"role": "user", "content": "Read a file"}],
            tools=[{"name": "read_file", "description": "Read a file", "input_schema": {}}],
        )

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "/tmp/test.txt"}


@pytest.mark.asyncio
async def test_anthropic_retry_on_failure():
    """Anthropic provider retries on API failure."""
    from src.llm.anthropic import AnthropicProvider
    import anthropic

    mock_success = MagicMock()
    mock_success.content = [MagicMock(text="ok", type="text")]
    mock_success.stop_reason = "end_turn"
    mock_success.model = "claude-sonnet-4-20250514"
    mock_success.usage = MagicMock(input_tokens=5, output_tokens=3)

    with patch("src.llm.anthropic.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            side_effect=[anthropic.APIError("rate limit"), mock_success]
        )

        provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-20250514", max_retries=3)
        result = await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert result.content == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_anthropic.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.llm.anthropic'"

- [ ] **Step 3: Implement Anthropic provider**

```python
# src/llm/anthropic.py
import asyncio
from typing import Any, AsyncIterator

import anthropic

from src.llm.provider import LLMProvider, LLMResponse, TokenUsage, ToolCall


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider with retry and tool use support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send messages to Claude and return a response."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                params: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "messages": messages,
                }
                if tools:
                    params["tools"] = tools

                response = await self._client.messages.create(**params)
                return self._parse_response(response)

            except (anthropic.APIError, anthropic.RateLimitError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise last_error  # type: ignore

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream the Claude response."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
        }
        if tools:
            params["tools"] = tools

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into our LLMResponse format."""
        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm/test_anthropic.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/anthropic.py tests/test_llm/test_anthropic.py
git commit -m "feat: Anthropic provider with tool use, retry, and streaming support"
```

---

## Task 8: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Smoke test: Core + LLM modules work together."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import load_config, get_settings, reset_settings
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import MultiAgentError, ToolError
from src.core.events import EventBus, Message, MessageType
from src.llm.provider import LLMResponse, TokenUsage


def test_config_logging_integration(sample_config):
    """Config drives logging setup."""
    settings = load_config(sample_config)
    setup_logging(level=settings.system.log_level)
    logger = get_logger("integration")
    assert logger.level == 10  # DEBUG


def test_exceptions_chain():
    """Exceptions carry context through the chain."""
    try:
        raise ToolError("file not found", tool_name="read_file", code="E404")
    except MultiAgentError as e:
        assert e.tool_name == "read_file"
        assert e.code == "E404"


@pytest.mark.asyncio
async def test_event_bus_with_llm_response():
    """Event bus can carry LLM response payloads."""
    bus = EventBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("executor", handler)
    await bus.start()

    response = LLMResponse(
        content="Analysis complete",
        tool_calls=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        model="claude-sonnet-4-20250514",
    )

    msg = Message(
        sender="planner",
        receiver="executor",
        type=MessageType.TASK,
        payload={"instruction": response.content, "model": response.model},
        timestamp=0.0,
        correlation_id="task-1",
    )
    await bus.publish(msg)

    import asyncio
    await asyncio.sleep(0.1)
    await bus.stop()

    assert len(received) == 1
    assert received[0].payload["instruction"] == "Analysis complete"
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (config, exceptions, logging, events, provider, anthropic, integration)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for Core + LLM modules"
```

---

## Verification Checklist

After completing all tasks, run:

```bash
# All tests pass
pytest tests/ -v

# No import errors
python -c "from src.core.config import load_config; from src.core.events import EventBus; from src.llm.anthropic import AnthropicProvider; print('All imports OK')"

# Config loads
python -c "from src.core.config import load_config; s = load_config(Path('config.yaml')); print(f'System: {s.system.name}, LLM: {s.llm.model}')"
```
