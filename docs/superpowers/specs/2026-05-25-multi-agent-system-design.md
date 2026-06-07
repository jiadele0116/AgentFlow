# 多智能体系统设计规格说明

## 概述

基于 Python 的多智能体协作系统，集成 MCP、Tools、Skills 和 RAG 四大核心模块。基于事件驱动架构，多个 Agent 并行或串行协作，通过统一协议层调度工具与技能，结合 RAG 实现知识增强决策。

## 关键决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 架构 | 事件驱动 (asyncio.Queue) | 符合 PRD 的消息总线规格，松耦合，天然支持并行 |
| LLM 提供商 | Anthropic (Claude) | 强大的工具调用支持，完善的 SDK 文档 |
| 向量存储 | ChromaDB | 持久化存储，良好的查询能力，可用于生产环境 |
| 嵌入模型 | BAAI/bge-m3 | 1024 维，优秀的中文支持，基于 sentence-transformers |
| 构建策略 | 增量式，逐模块推进 | 每个模块验证通过后再叠加下一个 |
| API 层 | 推迟实现 | 优先聚焦核心逻辑，后续再添加 FastAPI |

## 模块拆分与构建顺序

```
第一阶段：基础层
  1. Core     (配置、日志、异常、事件总线)
  2. LLM      (Anthropic 提供商，统一接口)

第二阶段：能力层
  3. Tools    (注册表、引擎、沙箱、内置工具)
  4. RAG      (处理器、分块器、嵌入器、ChromaDB、检索)

第三阶段：编排层
  5. Agents   (基类、类型、记忆、策略、消息总线)
  6. MCP      (协议、客户端、服务器)
  7. Skills   (引擎、加载器、YAML 定义)
```

依赖关系仅向下流动。每个模块都是 `src/` 下的一个 Python 包。

---

## 第一阶段：基础层

### 1. Core 模块 (`src/core/`)

#### config.py
- 使用 PyYAML 加载 `config.yaml`
- 支持 `${ENV_VAR}` 环境变量替换（用于密钥、连接字符串等敏感信息）
- 单例 `Settings` 对象，支持类型化访问：`settings.llm.model`、`settings.rag.chunk_size`
- 使用 Pydantic 模型进行配置验证

#### logging.py
- 基于 Python `logging` 模块的结构化 JSON 日志
- 每条日志包含 `trace_id` 用于关联追踪
- 日志级别可通过 `settings.system.log_level` 配置

#### exceptions.py
- 基础异常类 `MultiAgentError`，携带错误码和上下文字典
- 子类：`AgentError`、`ToolError`、`SkillError`、`RAGError`、`MCPError`、`ConfigError`
- 每个异常携带相关上下文（agent_id、tool_name 等）

#### events.py
- 基于 `asyncio.Queue` 的异步发布/订阅事件总线
- 支持点对点（Agent A → Agent B）和广播模式
- `Message` 数据类：

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
    receiver: str        # Agent ID 或 "broadcast"
    type: MessageType
    payload: dict
    timestamp: float
    correlation_id: str  # 用于追踪任务链
```

- 总线作为后台 asyncio 任务运行，分发消息到订阅者回调
- API：`bus.subscribe(agent_id, callback)`、`bus.publish(message)`

### 2. LLM 模块 (`src/llm/`)

#### provider.py
```python
class LLMProvider(ABC):
    async def complete(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> LLMResponse: ...
    async def stream(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> AsyncIterator[str]: ...
```

#### anthropic.py
- 使用 `anthropic` SDK
- 支持工具调用（function calling）—— 对 Agent 和 Skills 至关重要
- 自动重试，指数退避（最多 3 次）
- 配置映射：`settings.llm.model`、`settings.llm.api_key`、`settings.llm.temperature`、`settings.llm.max_tokens`

```python
@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] | None
    usage: TokenUsage
    model: str
```

不依赖其他模块。

---

## 第二阶段：能力层

### 3. Tools 模块 (`src/tools/`)

#### registry.py
- 中心字典：工具名 → Tool 定义
- 装饰器注册：`@register_tool(name, description, parameters, timeout, retry, requires_confirmation)`
- `list_tools() -> list[dict]` 返回 JSON Schema 描述，供 LLM 使用

#### engine.py
- 执行管线：参数校验 → 权限检查 → 超时 → 重试 → 结果格式化 → 审计日志
- 全异步，可配置超时（默认 30 秒）
- 返回 `ToolResult(success, data, error, duration_ms)`

```python
class ToolEngine:
    def __init__(self, registry: ToolRegistry): ...
    async def execute(self, name: str, params: dict, context: ExecutionContext) -> ToolResult: ...
    def list_tools(self) -> list[dict]: ...
```

#### sandbox.py
- 用于代码执行工具（run_python、run_shell）
- 使用 `RestrictedPython` 进行 Python 沙箱化
- 文件系统限制在配置的工作目录内
- 内存/CPU 使用限制（在可用的情况下）

#### builtin/ — 4 个文件

**file_ops.py**：`read_file`、`write_file`、`search_files`
**code_exec.py**：`run_python`、`run_shell`、`lint_code`
**network.py**：`fetch_url`、`search_web`
**data_ops.py**：`parse_json`、`csv_to_dict`、`json_to_csv`
**rag_ops.py**：`search_knowledge(query, collection, top_k)`、`index_document(doc, collection)` —— RAG 驱动的工具，桥接 Tools 和 RAG 模块

每个工具都是用 `@register_tool` 装饰的函数。

### 4. RAG 模块 (`src/rag/`)

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
- 可插拔的策略接口
- 策略：`fixed_size`、`sentence`、`semantic`、`code`、`recursive`
- 默认：`recursive` —— 先按 Markdown 标题分割，再按 token 数量分割
- 配置：`chunk_size=512`、`overlap=64`

#### embedder.py
- 封装 `sentence-transformers`，使用 `BAAI/bge-m3`
- `embed(text) -> list[float]`、`embed_batch(texts) -> list[list[float]]`
- 设备可通过配置指定（cpu/cuda）

#### stores/chroma_store.py
- 实现 `VectorStore` 抽象类的 `upsert`、`search`、`delete` 方法
- 使用 `chromadb`，持久化存储于 `settings.vector_store.persist_dir`

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
- 混合搜索：向量相似度 + BM25 关键词
- RRF（倒数排名融合）合并结果
- 可选重排序，使用 `BAAI/bge-reranker-v2-m3`
- 可配置 `top_k`、`rerank` 标志

#### query_rewrite.py
- 使用 LLM 改写查询以提高检索效果
- 可选功能 —— 由 `config.rag.rewrite` 控制

---

## 第三阶段：编排层

### 5. Agents 模块 (`src/agents/`)

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
    parent_id: str | None = None  # 用于子任务追踪

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

#### 5 种 Agent 类型

**planner.py** — `PlannerAgent`：接收用户任务，分解为子任务，分配给执行器。使用 LLM 进行任务分解。

**executor.py** — `ExecutorAgent`：通过调用 Tools/Skills 执行子任务。将结果报告给规划器。

**critic.py** — `CriticAgent`：审查执行器输出的正确性、性能、安全性、可维护性。提供反馈或批准。

**retriever.py** — `RetrieverAgent`：执行 RAG 检索，按需为其他 Agent 提供知识上下文。

**coordinator.py** — `CoordinatorAgent`：管理 Agent 间通信，解决冲突，追踪整体任务进度。

#### memory.py
- **短期记忆**：内存中的最近消息列表（可配置最大数量，默认 20 轮）
- **长期记忆**：基于 `aiosqlite` 的 SQLite 持久化存储
- 按 Agent 隔离

#### strategy.py
```python
class ExecutionStrategy(ABC):
    async def execute(self, task: Task, agent: BaseAgent) -> Result: ...

class ReActStrategy(ExecutionStrategy):
    # 思考 → 行动 → 观察 循环

class PlanExecuteStrategy(ExecutionStrategy):
    # 先规划，再逐步执行

class CoTStrategy(ExecutionStrategy):
    # 思维链推理
```

#### message_bus.py
- 封装 Core 层的 `EventBus`，提供 Agent 专属路由
- 管理生命周期转换（INIT → IDLE → PLANNING → EXECUTING → REVIEWING → COMPLETED）
- 当 Critic 提供反馈时，REVISING 循环回到 EXECUTING
- 关联 ID 追踪任务链

### 6. MCP 模块 (`src/mcp/`)

#### protocol.py
```python
@dataclass
class MCPRequest:
    method: str           # "tools/call"、"resources/read" 等
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
- 连接 MCP 服务器，发送请求，接收响应
- 供 Agent 通过统一接口访问外部工具
- 连接池、超时管理

#### servers/ — 4 个内置服务器

| 文件 | 服务器 | 工具 | 资源 |
|------|--------|------|------|
| `filesystem.py` | FileSystemServer | read_file、write_file、list_dir、search_files | `file://` |
| `database.py` | DatabaseServer | query、execute、describe_table | `db://` |
| `web.py` | WebServer | fetch_url、scrape、search_web | `https://` |
| `code.py` | CodeServer | run_python、lint、format | `code://` |

自定义服务器扩展通过装饰器实现：
```python
@mcp_server(name="custom", version="1.0")
class CustomServer:
    @tool(description="...") async def my_tool(self, param): ...
    @resource(uri_scheme="custom") async def my_resource(self, uri): ...
```

### 7. Skills 模块 (`src/skills/`)

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
- 从 `definitions/` 目录加载 YAML 格式的技能定义
- 按 schema 验证
- 支持开发时热重载

#### definitions/ — 5 个 YAML 文件

| 文件 | 技能 | 使用的工具 |
|------|------|-----------|
| `code_review.yaml` | code_review | read_file、lint_code、search_knowledge |
| `bug_fix.yaml` | bug_fix | search_files、read_file、run_python、write_file |
| `data_analysis.yaml` | data_analysis | csv_to_dict、run_python、write_file |
| `doc_generator.yaml` | doc_generator | read_file、search_files、write_file |
| `research.yaml` | research | search_web、fetch_url、search_knowledge、index_document |

每个 YAML 使用 Jinja2 风格的模板解析：`{{ input.file_path }}`、`{{ code_content }}`。

---

## 项目结构

```
multi_agent_system/
├── pyproject.toml
├── config.yaml
├── PRD.md
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── main.py                    # 入口点
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

## 依赖

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

## 配置

项目根目录下的 `config.yaml`：

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

## 非功能性需求

| 维度 | 要求 |
|------|------|
| 并发 | 最多支持 5 个 Agent 并行执行 |
| 超时 | 工具调用默认 30 秒，可配置 |
| 重试 | LLM 调用失败自动重试 3 次，指数退避 |
| 日志 | 结构化 JSON，每条日志包含 trace_id |
| 可观测性 | 每个 Agent/Tool 调用都有 trace_id 用于链路追踪 |
| 安全 | 工具权限控制、沙箱化代码执行、输入校验 |
| 可扩展性 | 新的 Agent/Tool/Skill/MCP Server 均可作为插件注册 |

## 暂不实现（推迟）

- FastAPI REST API 层
- WebSocket 实时通信
- CLI 接口（typer）
- PostgreSQL 长期记忆后端
- FAISS 向量存储后端
- OpenAI/本地 LLM 提供商
