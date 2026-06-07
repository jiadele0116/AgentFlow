# 第一阶段：基础层实施计划

> **致智能体工作者：** 必须使用子技能：superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框（`- [ ]`）语法进行追踪。

**目标：** 构建 Core（配置、日志、异常、事件总线）和 LLM（Anthropic 提供商）模块——所有后续模块的基础。

**架构：** Core 提供基础设施（配置、日志、事件）。LLM 将 Anthropic API 封装在提供商接口之后。两者都是无依赖的基础模块，所有后续模块都从它们导入。

**技术栈：** Python 3.11+、PyYAML、Pydantic、anthropic SDK、pytest、pytest-asyncio

---

## 文件结构

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py          # YAML 配置加载器、Settings 单例、环境变量展开
│   ├── logging.py         # 带 trace_id 的结构化 JSON 日志
│   ├── exceptions.py      # 异常层次结构
│   └── events.py          # 异步 EventBus、Message 数据类、MessageType 枚举
├── llm/
│   ├── __init__.py
│   ├── provider.py        # LLMProvider ABC、LLMResponse/ToolCall/TokenUsage 数据类
│   └── anthropic.py       # Anthropic 实现，支持重试和工具调用
config.yaml                # 系统配置
pyproject.toml             # 项目元数据和依赖
tests/
├── __init__.py
├── conftest.py            # 共享 fixtures
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

## 任务 1：项目脚手架

**文件：**
- 创建：`pyproject.toml`
- 创建：`config.yaml`
- 创建：`src/__init__.py`
- 创建：`tests/__init__.py`
- 创建：`tests/conftest.py`

- [ ] **步骤 1：创建 pyproject.toml**

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

- [ ] **步骤 2：创建 config.yaml**

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

- [ ] **步骤 3：创建包初始化文件**

```python
# src/__init__.py
```

```python
# tests/__init__.py
```

- [ ] **步骤 4：创建 conftest.py**

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
    """创建用于测试的最小配置文件。"""
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

- [ ] **步骤 5：安装开发依赖并验证**

运行：`pip install -e ".[dev]"`
预期：成功安装包和开发依赖

- [ ] **步骤 6：提交**

```bash
git init
git add pyproject.toml config.yaml src/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: project scaffolding with pyproject.toml and config"
```

---

## 任务 2：配置模块

**文件：**
- 创建：`src/core/__init__.py`
- 创建：`src/core/config.py`
- 创建：`tests/test_core/__init__.py`
- 创建：`tests/test_core/test_config.py`

- [ ] **步骤 1：编写失败的配置测试**

```python
# tests/test_core/test_config.py
import os
import pytest
from pathlib import Path


def test_load_config_from_file(sample_config):
    """配置加载 YAML 文件并返回类型化设置。"""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.system.name == "Test System"
    assert settings.system.log_level == "DEBUG"
    assert settings.llm.provider == "anthropic"
    assert settings.llm.model == "claude-sonnet-4-20250514"


def test_config_expands_env_vars(tmp_path):
    """配置展开值中的 ${ENV_VAR} 模式。"""
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
    """get_settings 在重复调用时返回同一实例。"""
    from src.core.config import load_config, get_settings, reset_settings

    reset_settings()
    load_config(sample_config)
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    reset_settings()


def test_config_missing_file():
    """load_config 对缺失文件抛出 FileNotFoundError。"""
    from src.core.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_config_default_values(sample_config):
    """配置为可选字段提供合理的默认值。"""
    from src.core.config import load_config

    settings = load_config(sample_config)
    assert settings.rag.chunk_size == 512
    assert settings.rag.chunk_overlap == 64
    assert settings.agents.max_concurrent == 5
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_core/test_config.py -v`
预期：失败，提示 "ModuleNotFoundError: No module named 'src.core.config'"

- [ ] **步骤 3：实现配置模块**

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
    """递归展开配置值中的 ${ENV_VAR} 模式。"""
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
    """从 YAML 文件加载配置并展开环境变量。"""
    global _settings
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)
    _settings = Settings(**expanded)
    return _settings


def get_settings() -> Settings:
    """获取当前设置单例。"""
    if _settings is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _settings


def reset_settings():
    """重置设置单例（用于测试）。"""
    global _settings
    _settings = None
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_core/test_config.py -v`
预期：全部 5 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/core/__init__.py src/core/config.py tests/test_core/__init__.py tests/test_core/test_config.py
git commit -m "feat: config module with YAML loading, env var expansion, Pydantic settings"
```

---

## 任务 3：异常模块

**文件：**
- 创建：`src/core/exceptions.py`
- 创建：`tests/test_core/test_exceptions.py`

- [ ] **步骤 1：编写失败的测试**

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

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_core/test_exceptions.py -v`
预期：失败，提示 "ImportError: cannot import name 'MultiAgentError'"

- [ ] **步骤 3：实现异常模块**

```python
# src/core/exceptions.py
from typing import Any


class MultiAgentError(Exception):
    """多智能体系统的基础异常。"""

    def __init__(self, message: str, code: str = "", context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class AgentError(MultiAgentError):
    """来自智能体的错误。"""

    def __init__(self, message: str, agent_id: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id


class ToolError(MultiAgentError):
    """来自工具执行的错误。"""

    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class SkillError(MultiAgentError):
    """来自技能执行的错误。"""

    def __init__(self, message: str, skill_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.skill_name = skill_name


class RAGError(MultiAgentError):
    """来自 RAG 模块的错误。"""


class MCPError(MultiAgentError):
    """来自 MCP 模块的错误。"""

    def __init__(self, message: str, server_name: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.server_name = server_name


class ConfigError(MultiAgentError):
    """来自配置的错误。"""
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_core/test_exceptions.py -v`
预期：全部 7 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/core/exceptions.py tests/test_core/test_exceptions.py
git commit -m "feat: exception hierarchy with AgentError, ToolError, SkillError, RAGError, MCPError"
```

---

## 任务 4：日志模块

**文件：**
- 创建：`src/core/logging.py`
- 创建：`tests/test_core/test_logging.py`

- [ ] **步骤 1：编写失败的测试**

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

    # 查找 JSON 日志记录
    for record in caplog.records:
        if record.getMessage() == "structured message":
            # 格式化器应该已生成 JSON
            assert hasattr(record, "trace_id")
            break


def test_get_logger_returns_named_logger():
    from src.core.logging import get_logger

    logger = get_logger("my_module")
    assert logger.name == "multi_agent.my_module"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_core/test_logging.py -v`
预期：失败，提示 "ModuleNotFoundError"

- [ ] **步骤 3：实现日志模块**

```python
# src/core/logging.py
import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """包含 trace_id 和结构化字段的 JSON 格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 如果存在 trace_id 则添加
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id

        # 添加其他额外字段
        for key in ("agent_id", "tool_name", "skill_name", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """人类可读的格式化器，带可选 trace_id。"""

    def format(self, record: logging.LogRecord) -> str:
        trace = getattr(record, "trace_id", "")
        prefix = f"[{trace}] " if trace else ""
        return f"{prefix}{record.levelname} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO", json_format: bool = False):
    """配置多智能体日志系统。"""
    logger = logging.getLogger("multi_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清除现有处理器
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(SimpleFormatter())

    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """获取 multi_agent 命名空间下的命名日志器。"""
    return logging.getLogger(f"multi_agent.{name}")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_core/test_logging.py -v`
预期：全部 4 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/core/logging.py tests/test_core/test_logging.py
git commit -m "feat: structured logging with trace_id support and JSON format"
```

---

## 任务 5：事件总线模块

**文件：**
- 创建：`src/core/events.py`
- 创建：`tests/test_core/test_events.py`

- [ ] **步骤 1：编写失败的测试**

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

    # 给总线分发消息的时间
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
    """向没有处理器的智能体发布消息不会抛出异常。"""
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
    # 不应抛出异常
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

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_core/test_events.py -v`
预期：失败，提示 "ImportError: cannot import name 'EventBus'"

- [ ] **步骤 3：实现事件总线**

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
    receiver: str  # 智能体 ID 或 "broadcast"
    type: MessageType
    payload: dict[str, Any]
    timestamp: float
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


MessageHandler = Callable[[Message], Awaitable[None]]


class EventBus:
    """用于智能体通信的异步发布/订阅事件总线。"""

    def __init__(self):
        self._subscribers: dict[str, list[MessageHandler]] = {}
        self._queue: asyncio.Queue[Message | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    def subscribe(self, agent_id: str, handler: MessageHandler):
        """注册用于处理发往 agent_id 消息的处理器。"""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)

    def unsubscribe(self, agent_id: str):
        """移除 agent_id 的所有处理器。"""
        self._subscribers.pop(agent_id, None)

    async def publish(self, message: Message):
        """向总线发布消息。"""
        await self._queue.put(message)

    async def start(self):
        """启动消息分发循环。"""
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        """优雅地停止分发循环。"""
        self._running = False
        await self._queue.put(None)  # 哨兵值
        if self._task:
            await self._task

    async def _dispatch_loop(self):
        """主循环：从队列读取消息并分发到处理器。"""
        while self._running:
            message = await self._queue.get()
            if message is None:
                break

            if message.receiver == "broadcast":
                # 发送给除发送者外的所有订阅者
                for agent_id, handlers in self._subscribers.items():
                    if agent_id != message.sender:
                        for handler in handlers:
                            asyncio.create_task(self._safe_deliver(handler, message))
            else:
                # 点对点
                handlers = self._subscribers.get(message.receiver, [])
                for handler in handlers:
                    asyncio.create_task(self._safe_deliver(handler, message))

    async def _safe_deliver(self, handler: MessageHandler, message: Message):
        """向处理器投递消息，捕获异常。"""
        try:
            await handler(message)
        except Exception:
            # 记录日志但不崩溃总线
            pass
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_core/test_events.py -v`
预期：全部 5 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/core/events.py tests/test_core/test_events.py
git commit -m "feat: async EventBus with point-to-point and broadcast messaging"
```

---

## 任务 6：LLM 提供商接口

**文件：**
- 创建：`src/llm/__init__.py`
- 创建：`src/llm/provider.py`
- 创建：`tests/test_llm/__init__.py`
- 创建：`tests/test_llm/test_provider.py`

- [ ] **步骤 1：编写失败的测试**

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

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_llm/test_provider.py -v`
预期：失败，提示 "ModuleNotFoundError"

- [ ] **步骤 3：实现提供商接口**

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
    """LLM 提供商的抽象基类。"""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """向 LLM 发送消息并返回响应。"""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """逐 token 流式传输 LLM 响应。"""
        ...
        # 使其成为异步生成器以进行类型检查
        yield ""  # pragma: no cover
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_llm/test_provider.py -v`
预期：全部 3 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/llm/__init__.py src/llm/provider.py tests/test_llm/__init__.py tests/test_llm/test_provider.py
git commit -m "feat: LLM provider interface with LLMResponse, ToolCall, TokenUsage"
```

---

## 任务 7：Anthropic LLM 实现

**文件：**
- 创建：`src/llm/anthropic.py`
- 创建：`tests/test_llm/test_anthropic.py`

- [ ] **步骤 1：编写失败的测试**

```python
# tests/test_llm/test_anthropic.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.llm.provider import LLMResponse, ToolCall, TokenUsage


@pytest.mark.asyncio
async def test_anthropic_complete_basic():
    """Anthropic 提供商发送消息并返回结构化响应。"""
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
    """Anthropic 提供商处理 tool_use 响应。"""
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
    """Anthropic 提供商在 API 失败时重试。"""
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

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_llm/test_anthropic.py -v`
预期：失败，提示 "ModuleNotFoundError: No module named 'src.llm.anthropic'"

- [ ] **步骤 3：实现 Anthropic 提供商**

```python
# src/llm/anthropic.py
import asyncio
from typing import Any, AsyncIterator

import anthropic

from src.llm.provider import LLMProvider, LLMResponse, TokenUsage, ToolCall


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM 提供商，支持重试和工具调用。"""

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
        """向 Claude 发送消息并返回响应。"""
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
        """流式传输 Claude 响应。"""
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
        """将 Anthropic 响应解析为我们的 LLMResponse 格式。"""
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

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_llm/test_anthropic.py -v`
预期：全部 3 个测试通过

- [ ] **步骤 5：提交**

```bash
git add src/llm/anthropic.py tests/test_llm/test_anthropic.py
git commit -m "feat: Anthropic provider with tool use, retry, and streaming support"
```

---

## 任务 8：集成冒烟测试

**文件：**
- 创建：`tests/test_integration.py`

- [ ] **步骤 1：编写集成测试**

```python
# tests/test_integration.py
"""冒烟测试：Core + LLM 模块协同工作。"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import load_config, get_settings, reset_settings
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import MultiAgentError, ToolError
from src.core.events import EventBus, Message, MessageType
from src.llm.provider import LLMResponse, TokenUsage


def test_config_logging_integration(sample_config):
    """配置驱动日志设置。"""
    settings = load_config(sample_config)
    setup_logging(level=settings.system.log_level)
    logger = get_logger("integration")
    assert logger.level == 10  # DEBUG


def test_exceptions_chain():
    """异常在调用链中携带上下文。"""
    try:
        raise ToolError("file not found", tool_name="read_file", code="E404")
    except MultiAgentError as e:
        assert e.tool_name == "read_file"
        assert e.code == "E404"


@pytest.mark.asyncio
async def test_event_bus_with_llm_response():
    """事件总线可以携带 LLM 响应载荷。"""
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

- [ ] **步骤 2：运行所有测试**

运行：`pytest tests/ -v`
预期：所有测试通过（config、exceptions、logging、events、provider、anthropic、integration）

- [ ] **步骤 3：提交**

```bash
git add tests/test_integration.py
git commit -m "test: integration smoke test for Core + LLM modules"
```

---

## 验证清单

完成所有任务后，运行：

```bash
# 所有测试通过
pytest tests/ -v

# 无导入错误
python -c "from src.core.config import load_config; from src.core.events import EventBus; from src.llm.anthropic import AnthropicProvider; print('All imports OK')"

# 配置加载正常
python -c "from src.core.config import load_config; s = load_config(Path('config.yaml')); print(f'System: {s.system.name}, LLM: {s.llm.model}')"
```
