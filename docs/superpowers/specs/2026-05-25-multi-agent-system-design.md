# Multi-Agent System Design Spec

## Overview

A Python-based multi-agent协作系统，集成 MCP、Tools、Skills 和 RAG 四大核心模块。基于事件驱动架构，多个 Agent 并行或串行协作，通过统一协议层调度工具与技能，结合 RAG 实现知识增强决策。

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Event-Driven (asyncio.Queue) | Matches PRD's message bus spec, loose coupling, natural parallelism |
| LLM Provider | Anthropic (Claude) | Strong tool-use support, well-documented SDK |
| Vector Store | ChromaDB | Persistent storage, good querying, production-ready |
| Embedding Model | BAAI/bge-m3 | 1024-dim, excellent Chinese support, sentence-transformers |
| Build Strategy | Incremental, module by module | Validate each module before stacking next |
| API Layer | Deferred | Focus on core logic first, add FastAPI later |

## Module Decomposition & Build Order

```
Phase 1: Foundation
  1. Core     (config, logging, exceptions, event bus)
  2. LLM      (Anthropic provider, unified interface)

Phase 2: Capabilities
  3. Tools    (registry, engine, sandbox, built-in tools)
  4. RAG      (processor, chunker, embedder, ChromaDB, retrieval)

Phase 3: Orchestration
  5. Agents   (base, types, memory, strategies, message bus)
  6. MCP      (protocol, client, servers)
  7. Skills   (engine, loader, YAML definitions)
```

Dependencies flow downward only. Each module is a Python package under `src/`.

---

## Phase 1: Foundation

### 1. Core Module (`src/core/`)

#### config.py
- Loads `config.yaml` using PyYAML
- Supports `${ENV_VAR}` expansion for secrets (api_key, connection_string)
- Singleton `Settings` object with typed access: `settings.llm.model`, `settings.rag.chunk_size`
- Pydantic models for config validation

#### logging.py
- Structured JSON logging via Python's `logging` module
- Every log entry includes `trace_id` for correlation
- Configurable log level from `settings.system.log_level`

#### exceptions.py
- Base `MultiAgentError` with error code and context dict
- Subclasses: `AgentError`, `ToolError`, `SkillError`, `RAGError`, `MCPError`, `ConfigError`
- Each carries relevant context (agent_id, tool_name, etc.)

#### events.py
- Async pub/sub event bus built on `asyncio.Queue`
- Point-to-point (agent A → agent B) and broadcast mode
- `Message` dataclass:

```python
class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    HEARTBEAT = "heartbeat"

@dataclass
class Message:
    sender: str          # Agent ID
    receiver: str        # Agent ID or "broadcast"
    type: MessageType
    payload: dict
    timestamp: float
    correlation_id: str  # tracks task chains
```

- Bus runs as background asyncio task, dispatches to subscriber callbacks
- API: `bus.subscribe(agent_id, callback)`, `bus.publish(message)`

### 2. LLM Module (`src/llm/`)

#### provider.py
```python
class LLMProvider(ABC):
    async def complete(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> LLMResponse: ...
    async def stream(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> AsyncIterator[str]: ...
```

#### anthropic.py
- Uses `anthropic` SDK
- Supports tool use (function calling) — critical for Agents and Skills
- Automatic retry with exponential backoff (3 retries)
- Maps config: `settings.llm.model`, `settings.llm.api_key`, `settings.llm.temperature`, `settings.llm.max_tokens`

```python
@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: TokenUsage
    model: str
```

Zero dependencies on other modules.

---

## Phase 2: Capabilities

### 3. Tools Module (`src/tools/`)

#### registry.py
- Central dict: tool name → Tool definition
- Decorator registration: `@register_tool(name, description, parameters, timeout, retry, requires_confirmation)`
- `list_tools() -> list[dict]` returns JSON Schema descriptions for LLM consumption

#### engine.py
- Execution pipeline: param validation → permission check → timeout → retry → result formatting → audit log
- All async, configurable timeout (default 30s)
- Returns `ToolResult(success, data, error, duration_ms)`

```python
class ToolEngine:
    def __init__(self, registry: ToolRegistry): ...
    async def execute(self, name: str, params: dict, context: ExecutionContext) -> ToolResult: ...
    def list_tools(self) -> list[dict]: ...
```

#### sandbox.py
- For code execution tools (run_python, run_shell)
- `RestrictedPython` for Python sandboxing
- File system restricted to workspace directory from config
- Memory/CPU limits where available

#### builtin/ — 4 files

**file_ops.py**: `read_file`, `write_file`, `search_files`
**code_exec.py**: `run_python`, `run_shell`, `lint_code`
**network.py**: `fetch_url`, `search_web`
**data_ops.py**: `parse_json`, `csv_to_dict`, `json_to_csv`
**rag_ops.py**: `search_knowledge(query, collection, top_k)`, `index_document(doc, collection)` — RAG-powered tools that bridge Tools and RAG modules

Each tool is a function decorated with `@register_tool`.

### 4. RAG Module (`src/rag/`)

#### processor.py
```python
class DocumentProcessor:
    async def ingest(self, source: str, metadata: dict = None) -> list[Chunk]:
        document = await self.loader.load(source)
        cleaned = self.cleaner.clean(document)
        chunks = self.chunker.chunk(cleaned)
        embeddings = await self.embedder.embed_batch([c.text for c in chunks])
        await self.vector_store.upsert(chunks, embeddings)
        return chunks
```

#### chunker.py
- Pluggable strategy interface
- Strategies: `fixed_size`, `sentence`, `semantic`, `code`, `recursive`
- Default: `recursive` — splits by Markdown headers, then by token count
- Config: `chunk_size=512`, `overlap=64`

#### embedder.py
- Wraps `sentence-transformers` with `BAAI/bge-m3`
- `embed(text) -> list[float]`, `embed_batch(texts) -> list[list[float]]`
- Device configurable (cpu/cuda) from config

#### stores/chroma_store.py
- Implements `VectorStore` ABC with `upsert`, `search`, `delete`
- Uses `chromadb` with persistent storage at `settings.vector_store.persist_dir`

#### vector_store.py
```python
class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]): ...
    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int, filters: dict = None) -> list[SearchResult]: ...
    @abstractmethod
    async def delete(self, ids: list[str]): ...
```

#### retrieval.py
- Hybrid search: vector similarity + BM25 keyword
- RRF (Reciprocal Rank Fusion) to combine results
- Optional reranking via `BAAI/bge-reranker-v2-m3`
- Configurable `top_k`, `rerank` flag

#### query_rewrite.py
- Uses LLM to rewrite queries for better retrieval
- Optional — controlled by `config.rag.rewrite`

---

## Phase 3: Orchestration

### 5. Agents Module (`src/agents/`)

#### base.py
```python
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
    input_data: dict
    assigned_to: str | None = None
    parent_id: str | None = None  # for subtask tracking

@dataclass
class Result:
    task_id: str
    success: bool
    data: dict
    error: str | None = None
    duration_ms: float = 0

class BaseAgent(ABC):
    id: str
    role: str
    goal: str
    capabilities: list[str]
    strategy: ExecutionStrategy
    memory: Memory
    state: AgentState

    async def run(self, task: Task) -> Result: ...
    async def send(self, message: Message): ...
    async def receive(self) -> Message: ...
    async def transition(self, new_state: AgentState): ...
```

#### 5 Agent Types

**planner.py** — `PlannerAgent`: Receives user task, decomposes into subtasks, assigns to executor. Uses LLM for task decomposition.

**executor.py** — `ExecutorAgent`: Executes subtasks by calling Tools/Skills. Reports results back to planner.

**critic.py** — `CriticAgent`: Reviews executor output for correctness, performance, security, maintainability. Provides feedback or approval.

**retriever.py** — `RetrieverAgent`: Runs RAG retrieval, provides knowledge context to other agents on request.

**coordinator.py** — `CoordinatorAgent`: Manages inter-agent communication, resolves conflicts, tracks overall task progress.

#### memory.py
- **Short-term**: In-memory list of recent messages (configurable max, default 20 turns)
- **Long-term**: SQLite-backed persistence via `aiosqlite`
- Per-agent isolation

#### strategy.py
```python
class ExecutionStrategy(ABC):
    async def execute(self, task: Task, agent: BaseAgent) -> Result: ...

class ReActStrategy(ExecutionStrategy):
    # Thought → Action → Observation loop

class PlanExecuteStrategy(ExecutionStrategy):
    # Plan first, then execute step by step

class CoTStrategy(ExecutionStrategy):
    # Chain-of-thought reasoning
```

#### message_bus.py
- Wraps Core `EventBus` with agent-specific routing
- Manages lifecycle transitions (INIT → IDLE → PLANNING → EXECUTING → REVIEWING → COMPLETED)
- REVISING loop back to EXECUTING when critic provides feedback
- Correlation ID tracking for task chains

### 6. MCP Module (`src/mcp/`)

#### protocol.py
```python
@dataclass
class MCPRequest:
    method: str           # "tools/call", "resources/read", etc.
    params: dict
    request_id: str

@dataclass
class MCPResponse:
    request_id: str
    result: dict | None
    error: MCPError | None
```

#### server.py
```python
class MCPServer(ABC):
    name: str
    version: str
    resources: list[Resource]
    tools: list[Tool]

    async def handle_request(self, request: MCPRequest) -> MCPResponse: ...
```

#### client.py
- Connects to MCP servers, sends requests, receives responses
- Used by Agents to access external tools through uniform interface
- Connection pooling, timeout management

#### servers/ — 4 built-in servers

| File | Server | Tools | Resources |
|------|--------|-------|-----------|
| `filesystem.py` | FileSystemServer | read_file, write_file, list_dir, search_files | `file://` |
| `database.py` | DatabaseServer | query, execute, describe_table | `db://` |
| `web.py` | WebServer | fetch_url, scrape, search_web | `https://` |
| `code.py` | CodeServer | run_python, lint, format | `code://` |

Custom server extension via decorators:
```python
@mcp_server(name="custom", version="1.0")
class CustomServer:
    @tool(description="...") async def my_tool(self, param): ...
    @resource(uri_scheme="custom") async def my_resource(self, uri): ...
```

### 7. Skills Module (`src/skills/`)

#### engine.py
```python
class SkillEngine:
    async def execute(self, skill_name: str, input_data: dict, agent: BaseAgent) -> SkillResult:
        skill = self.registry[skill_name]
        context = SkillContext(input=input_data, agent=agent)

        for step in skill.steps:
            if step.type == "tool_call":
                result = await self.tool_engine.execute(step.tool, step.resolve_params(context))
                context.set(step.output_key, result)
            elif step.type == "llm_reasoning":
                prompt = step.render_prompt(context)
                response = await self.llm.complete(prompt, tools=self.tool_engine.list_tools())
                context.set(step.output_key, response)
            elif step.type == "conditional":
                if step.evaluate(context):
                    await self._execute_substeps(step.then_steps, context)
                else:
                    await self._execute_substeps(step.else_steps, context)
            elif step.type == "loop":
                while step.condition(context) and context.iteration < step.max_iterations:
                    await self._execute_substeps(step.body_steps, context)
                    context.iteration += 1

        return skill.extract_output(context)
```

#### loader.py
- Loads skill definitions from YAML files in `definitions/`
- Validates against schema
- Supports hot-reload for development

#### definitions/ — 5 YAML files

| File | Skill | Tools |
|------|-------|-------|
| `code_review.yaml` | code_review | read_file, lint_code, search_knowledge |
| `bug_fix.yaml` | bug_fix | search_files, read_file, run_python, write_file |
| `data_analysis.yaml` | data_analysis | csv_to_dict, run_python, write_file |
| `doc_generator.yaml` | doc_generator | read_file, search_files, write_file |
| `research.yaml` | research | search_web, fetch_url, search_knowledge, index_document |

Each YAML uses Jinja2-style template resolution: `{{ input.file_path }}`, `{{ code_content }}`.

---

## Project Structure

```
multi_agent_system/
├── pyproject.toml
├── config.yaml
├── PRD.md
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── events.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py
│   │   └── anthropic.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── engine.py
│   │   ├── sandbox.py
│   │   └── builtin/
│   │       ├── __init__.py
│   │       ├── file_ops.py
│   │       ├── code_exec.py
│   │       ├── network.py
│   │       └── data_ops.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── stores/
│   │   │   └── chroma_store.py
│   │   ├── retrieval.py
│   │   └── query_rewrite.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   ├── retriever.py
│   │   ├── coordinator.py
│   │   ├── memory.py
│   │   ├── strategy.py
│   │   └── message_bus.py
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── server.py
│   │   ├── client.py
│   │   └── servers/
│   │       ├── __init__.py
│   │       ├── filesystem.py
│   │       ├── database.py
│   │       ├── web.py
│   │       └── code.py
│   │
│   └── skills/
│       ├── __init__.py
│       ├── engine.py
│       ├── loader.py
│       └── definitions/
│           ├── code_review.yaml
│           ├── bug_fix.yaml
│           ├── data_analysis.yaml
│           ├── doc_generator.yaml
│           └── research.yaml
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   ├── test_llm/
│   ├── test_tools/
│   ├── test_rag/
│   ├── test_agents/
│   ├── test_mcp/
│   └── test_skills/
│
├── data/
│   ├── knowledge/
│   └── workspace/
│
└── config.yaml
```

## Dependencies

```toml
[project]
name = "multi-agent-system"
version = "1.0.0"
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
```

## Configuration

`config.yaml` at project root:

```yaml
system:
  name: "Multi-Agent System"
  version: "1.0.0"
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

## Non-Functional Requirements

| Dimension | Requirement |
|-----------|-------------|
| Concurrency | Up to 5 agents in parallel |
| Timeout | Tool calls: 30s default, configurable |
| Retry | LLM calls: 3 retries, exponential backoff |
| Logging | Structured JSON, trace_id on every entry |
| Observability | Every Agent/Tool call has trace_id for tracing |
| Security | Tool permission control, sandboxed code execution, input validation |
| Extensibility | New Agent/Tool/Skill/MCP Server can be registered as plugins |

## Out of Scope (Deferred)

- FastAPI REST API layer
- WebSocket real-time communication
- CLI interface (typer)
- PostgreSQL backend for long-term memory
- FAISS vector store backend
- OpenAI/local LLM providers
