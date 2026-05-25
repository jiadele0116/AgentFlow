# Multi-Agent System — 产品需求文档

## 1. 项目概述

构建一个基于 Python 的多 Agent 协作系统，集成 MCP（Model Context Protocol）、工具调用（Tools）、技能编排（Skills）和检索增强生成（RAG）四大核心模块。系统支持多个 Agent 并行或串行协作，通过统一的协议层调度工具与技能，结合 RAG 实现知识增强决策。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户接口层                          │
│              CLI / REST API / WebSocket                  │
├─────────────────────────────────────────────────────────┤
│                    Agent 编排层                           │
│         Orchestrator → Agent Router → Agent Pool         │
├──────────┬──────────┬──────────┬────────────────────────┤
│   MCP    │  Tools   │  Skills  │         RAG            │
│  Server  │  Engine  │  Engine  │    Retrieval + Index   │
├──────────┴──────────┴──────────┴────────────────────────┤
│                    基础设施层                             │
│         LLM Provider / Vector DB / Storage / Log         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Agent 模块

### 3.1 Agent 定义

每个 Agent 是一个独立的执行单元，拥有：
- **角色（Role）**：定义 Agent 的职责边界
- **目标（Goal）**：Agent 需要达成的任务目标
- **能力（Capabilities）**：可调用的 Tools 和 Skills 列表
- **记忆（Memory）**：短期上下文 + 长期持久化记忆
- **策略（Strategy）**：任务分解与执行策略（ReAct / Plan-and-Execute / Chain-of-Thought）

### 3.2 Agent 类型

| 类型 | 职责 | 示例 |
|------|------|------|
| `PlannerAgent` | 接收用户任务，分解为子任务，分配给其他 Agent | 项目经理 |
| `ExecutorAgent` | 执行具体子任务，调用 Tools/Skills | 开发者 |
| `CriticAgent` | 审查 Executor 的输出，提供反馈 | 代码审查员 |
| `RetrieverAgent` | 执行 RAG 检索，提供知识上下文 | 研究员 |
| `CoordinatorAgent` | 多 Agent 间的通信协调与冲突解决 | 调度员 |

### 3.3 Agent 生命周期

```
INIT → IDLE → PLANNING → EXECUTING → REVIEWING → COMPLETED
                                  ↑          │
                                  └── REVISING┘
```

### 3.4 Agent 通信协议

```python
@dataclass
class Message:
    sender: str          # Agent ID
    receiver: str        # Agent ID or "broadcast"
    type: MessageType    # TASK / RESULT / QUERY / FEEDBACK / HEARTBEAT
    payload: dict        # 消息内容
    timestamp: float
    correlation_id: str  # 用于追踪任务链
```

消息总线：基于 `asyncio.Queue` 的内存消息总线，支持点对点和广播模式。

---

## 4. MCP 模块（Model Context Protocol）

### 4.1 概述

MCP 是 Agent 与外部工具/数据源之间的标准化通信协议。每个 MCP Server 暴露一组 `Resources` 和 `Tools`，Agent 通过 MCP Client 统一访问。

### 4.2 MCP Server 定义

```python
class MCPServer:
    name: str
    version: str
    resources: list[Resource]   # 可读取的数据源
    tools: list[Tool]           # 可调用的工具

    async def handle_request(self, request: MCPRequest) -> MCPResponse: ...
```

### 4.3 内置 MCP Servers

| Server | 暴露的 Tools | 暴露的 Resources |
|--------|-------------|-----------------|
| `FileSystemServer` | read_file, write_file, list_dir, search_files | file:// URI |
| `DatabaseServer` | query, execute, describe_table | db:// URI |
| `WebServer` | fetch_url, scrape, search_web | https:// URI |
| `CodeServer` | run_python, lint, format | code:// URI |

### 4.4 MCP 协议消息格式

```python
@dataclass
class MCPRequest:
    method: str           # "tools/call", "resources/read", etc.
    params: dict          # 方法参数
    request_id: str

@dataclass
class MCPResponse:
    request_id: str
    result: dict | None
    error: MCPError | None
```

### 4.5 自定义 MCP Server 扩展

用户可通过装饰器快速注册新的 MCP Server：

```python
@mcp_server(name="custom", version="1.0")
class CustomServer:
    @tool(description="执行自定义操作")
    async def my_tool(self, param: str) -> str: ...

    @resource(uri_scheme="custom")
    async def my_resource(self, uri: str) -> bytes: ...
```

---

## 5. Tools 模块

### 5.1 Tool 定义

Tool 是系统中最小的可执行单元，具有明确的输入/输出 schema。

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema 格式
    handler: Callable         # 实际执行函数
    timeout: int = 30         # 秒
    retry: int = 0
    requires_confirmation: bool = False
```

### 5.2 内置 Tools

#### 文件操作
- `read_file(path) -> str`
- `write_file(path, content) -> bool`
- `search_files(pattern, directory) -> list[str]`

#### 代码执行
- `run_python(code, timeout) -> ExecutionResult`
- `run_shell(command, timeout) -> ExecutionResult`
- `lint_code(code, language) -> list[Diagnostic]`

#### 网络
- `fetch_url(url, method, headers, body) -> Response`
- `search_web(query) -> list[SearchResult]`

#### 数据处理
- `parse_json(text) -> dict`
- `csv_to_dict(csv_text) -> list[dict]`
- `json_to_csv(data) -> str`

#### RAG 相关
- `index_document(doc, collection) -> str`  # 返回 doc_id
- `search_knowledge(query, collection, top_k) -> list[Chunk]`

### 5.3 Tool 执行引擎

```python
class ToolEngine:
    registry: dict[str, Tool]

    def register(self, tool: Tool): ...
    def unregister(self, name: str): ...
    async def execute(self, name: str, params: dict, context: ExecutionContext) -> ToolResult: ...
    def list_tools(self) -> list[dict]:  # 返回 JSON Schema 描述，供 LLM 使用
```

执行流程：
1. 参数校验（基于 JSON Schema）
2. 权限检查（该 Agent 是否有权调用此 Tool）
3. 执行（带超时和重试）
4. 结果格式化
5. 审计日志记录

### 5.4 Tool 安全沙箱

代码执行类 Tool 必须在沙箱中运行：
- 禁止访问 `os.system`, `subprocess` 中的危险命令
- 文件系统限制在指定工作目录
- 网络访问白名单控制
- 内存和 CPU 使用限制

---

## 6. Skills 模块

### 6.1 概述

Skill 是 Tool 的高阶编排，由多个 Tool 调用 + 控制逻辑 + LLM 推理组合而成，实现完整的任务能力。

### 6.2 Skill 定义

```python
@dataclass
class Skill:
    name: str
    description: str
    required_tools: list[str]      # 依赖的 Tools
    required_capabilities: list[str]  # Agent 需要具备的能力
    steps: list[SkillStep]         # 执行步骤
    input_schema: dict
    output_schema: dict
```

### 6.3 内置 Skills

| Skill | 描述 | 使用的 Tools |
|-------|------|-------------|
| `code_review` | 对代码进行审查，输出问题列表和改进建议 | read_file, lint_code, search_knowledge |
| `bug_fix` | 接收 bug 描述，定位问题，生成修复方案 | search_files, read_file, run_python, write_file |
| `data_analysis` | 对数据集进行探索性分析，生成报告 | csv_to_dict, run_python, write_file |
| `doc_generator` | 根据代码生成文档 | read_file, search_files, write_file |
| `research` | 针对主题进行多源检索和总结 | search_web, fetch_url, search_knowledge, index_document |

### 6.4 Skill 执行模式

```python
class SkillEngine:
    async def execute(self, skill_name: str, input_data: dict, agent: Agent) -> SkillResult:
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

### 6.5 自定义 Skill 声明（YAML）

```yaml
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

---

## 7. RAG 模块

### 7.1 架构

```
文档输入 → 文档处理 → 分块 → 向量化 → 存储
                                          ↓
用户查询 → 查询改写 → 向量检索 → 重排序 → 上下文注入 → LLM 生成
```

### 7.2 文档处理管线

```python
class DocumentProcessor:
    """支持的文档格式：txt, md, pdf, docx, html, json, csv, code files"""

    async def ingest(self, source: str, metadata: dict = None) -> list[Chunk]:
        document = await self.loader.load(source)
        cleaned = self.cleaner.clean(document)
        chunks = self.chunker.chunk(cleaned)
        embeddings = await self.embedder.embed_batch([c.text for c in chunks])
        await self.vector_store.upsert(chunks, embeddings)
        return chunks
```

### 7.3 分块策略

| 策略 | 适用场景 | 参数 |
|------|---------|------|
| `fixed_size` | 通用文本 | chunk_size, overlap |
| `sentence` | 自然语言文本 | max_sentences |
| `semantic` | 主题边界明显的文档 | similarity_threshold |
| `code` | 源代码文件 | language, max_lines |
| `recursive` | 结构化文档（Markdown, HTML） | headers, max_size |

默认策略：`recursive`，chunk_size=512 tokens，overlap=64 tokens。

### 7.4 向量存储

支持两种后端：

| 后端 | 适用场景 | 依赖 |
|------|---------|------|
| `FAISS` | 本地开发、小规模数据 | faiss-cpu |
| `ChromaDB` | 持久化存储、中等规模 | chromadb |

抽象接口：

```python
class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]): ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int, filters: dict = None) -> list[SearchResult]: ...

    @abstractmethod
    async def delete(self, ids: list[str]): ...
```

### 7.5 检索策略

```python
class RetrievalPipeline:
    async def retrieve(self, query: str, config: RetrievalConfig) -> list[Chunk]:
        # Step 1: 查询改写（可选）
        rewritten = await self.query_rewriter.rewrite(query) if config.rewrite else query

        # Step 2: 混合检索
        vector_results = await self.vector_store.search(
            await self.embedder.embed(rewritten), top_k=config.top_k * 2
        )
        keyword_results = await self.bm25_search(rewritten, top_k=config.top_k * 2)

        # Step 3: 融合（RRF - Reciprocal Rank Fusion）
        fused = self.rrf_fusion(vector_results, keyword_results)

        # Step 4: 重排序
        if config.rerank:
            fused = await self.reranker.rerank(rewritten, fused)

        return fused[:config.top_k]
```

### 7.6 Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| `text-embedding-3-small` (OpenAI) | 1536 | 性价比高 |
| `bge-m3` (BAAI) | 1024 | 中文优秀，支持稠密+稀疏 |
| `jina-embeddings-v3` | 1024 | 多语言，Task LoRA |

默认使用 `bge-m3`，通过 sentence-transformers 加载。

---

## 8. 配置系统

### 8.1 全局配置 (`config.yaml`)

```yaml
system:
  name: "Multi-Agent System"
  version: "1.0.0"
  log_level: INFO

llm:
  provider: openai            # openai / anthropic / local
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 4096
  timeout: 60

embedding:
  model: BAAI/bge-m3
  device: cpu                 # cpu / cuda
  batch_size: 32

vector_store:
  backend: chroma             # faiss / chroma
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
  default_strategy: react     # react / plan_execute / cot
  memory:
    short_term_max: 20        # 消息轮数
    long_term_backend: sqlite  # sqlite / postgres

mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./workspace
    - name: database
      enabled: false
      connection_string: ${DB_URL}
    - name: web
      enabled: true
      allowed_domains: ["*"]
```

---

## 9. 项目结构

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
│   ├── core/                      # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py              # 配置加载
│   │   ├── logging.py             # 日志系统
│   │   ├── exceptions.py          # 自定义异常
│   │   └── events.py              # 事件总线
│   │
│   ├── agents/                    # Agent 模块
│   │   ├── __init__.py
│   │   ├── base.py                # BaseAgent 抽象类
│   │   ├── planner.py             # PlannerAgent
│   │   ├── executor.py            # ExecutorAgent
│   │   ├── critic.py              # CriticAgent
│   │   ├── retriever.py           # RetrieverAgent
│   │   ├── coordinator.py         # CoordinatorAgent
│   │   ├── memory.py              # 记忆管理
│   │   ├── strategy.py            # 执行策略 (ReAct, PlanExecute)
│   │   └── message_bus.py         # 消息总线
│   │
│   ├── mcp/                       # MCP 模块
│   │   ├── __init__.py
│   │   ├── protocol.py            # MCP 消息定义
│   │   ├── client.py              # MCP Client
│   │   ├── server.py              # MCP Server 基类
│   │   └── servers/               # 内置 MCP Servers
│   │       ├── __init__.py
│   │       ├── filesystem.py
│   │       ├── database.py
│   │       ├── web.py
│   │       └── code.py
│   │
│   ├── tools/                     # Tools 模块
│   │   ├── __init__.py
│   │   ├── registry.py            # Tool 注册中心
│   │   ├── engine.py              # Tool 执行引擎
│   │   ├── sandbox.py             # 代码执行沙箱
│   │   └── builtin/               # 内置 Tools
│   │       ├── __init__.py
│   │       ├── file_ops.py
│   │       ├── code_exec.py
│   │       ├── network.py
│   │       └── data_ops.py
│   │
│   ├── skills/                    # Skills 模块
│   │   ├── __init__.py
│   │   ├── engine.py              # Skill 执行引擎
│   │   ├── loader.py              # YAML Skill 加载器
│   │   └── definitions/           # Skill 定义文件
│   │       ├── code_review.yaml
│   │       ├── bug_fix.yaml
│   │       ├── data_analysis.yaml
│   │       ├── doc_generator.yaml
│   │       └── research.yaml
│   │
│   ├── rag/                       # RAG 模块
│   │   ├── __init__.py
│   │   ├── processor.py           # 文档处理管线
│   │   ├── chunker.py             # 分块策略
│   │   ├── embedder.py            # Embedding 封装
│   │   ├── vector_store.py        # 向量存储抽象
│   │   ├── stores/                # 向量存储实现
│   │   │   ├── faiss_store.py
│   │   │   └── chroma_store.py
│   │   ├── retrieval.py           # 检索管线
│   │   ├── reranker.py            # 重排序
│   │   └── query_rewrite.py       # 查询改写
│   │
│   ├── llm/                       # LLM 调用封装
│   │   ├── __init__.py
│   │   ├── provider.py            # 统一接口
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── local.py
│   │
│   └── api/                       # API 接口
│       ├── __init__.py
│       ├── server.py              # FastAPI 服务
│       ├── routes/
│       │   ├── agent.py
│       │   ├── chat.py
│       │   └── knowledge.py
│       └── websocket.py           # WebSocket 实时通信
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_mcp/
│   ├── test_tools/
│   ├── test_skills/
│   └── test_rag/
│
├── data/                          # 数据目录
│   ├── knowledge/                 # RAG 知识库文档
│   └── workspace/                 # Agent 工作目录
│
└── scripts/
    ├── init_db.py
    └── ingest_docs.py
```

---

## 10. 依赖

```toml
[project]
name = "multi-agent-system"
version = "1.0.0"
requires-python = ">=3.11"

dependencies = [
    # LLM
    "openai>=1.30.0",
    "anthropic>=0.25.0",

    # RAG
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",

    # 文档处理
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "beautifulsoup4>=4.12.0",

    # API
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
    "websockets>=12.0",

    # 工具
    "pydantic>=2.7.0",
    "pyyaml>=6.0",
    "rich>=13.7.0",            # CLI 美化
    "typer>=0.12.0",           # CLI 框架

    # 存储
    "aiosqlite>=0.20.0",

    # 安全
    "RestrictedPython>=7.1",   # 代码沙箱
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
]
```

---

## 11. 任务执行流程示例

**用户输入**："分析 `data.csv` 中的销售趋势，生成报告并存为 PDF"

```
1. PlannerAgent 接收任务
   ├── 分解为子任务：
   │   a. 读取并理解 data.csv
   │   b. 执行数据分析
   │   c. 生成可视化图表
   │   d. 撰写分析报告
   │   e. 转换为 PDF
   └── 分配给 ExecutorAgent

2. ExecutorAgent 执行
   ├── 调用 Tool: read_file("data.csv")
   ├── 调用 Skill: data_analysis
   │   ├── Tool: csv_to_dict
   │   ├── Tool: run_python (pandas 分析)
   │   └── LLM: 趋势解读
   ├── 调用 Tool: run_python (matplotlib 图表)
   ├── 调用 Skill: doc_generator
   │   └── LLM: 撰写报告
   └── 调用 Tool: run_python (markdown → PDF)

3. CriticAgent 审查
   ├── 检查数据准确性
   ├── 检查报告逻辑
   └── 输出反馈或通过

4. 返回最终结果给用户
```

---

## 12. 非功能性需求

| 维度 | 要求 |
|------|------|
| **并发** | 支持最多 5 个 Agent 并行执行 |
| **超时** | 单个 Tool 调用默认 30s，可配置 |
| **重试** | LLM 调用失败自动重试 3 次，指数退避 |
| **日志** | 结构化日志，支持 JSON 格式输出 |
| **可观测性** | 每个 Agent/Tool 调用记录 trace_id，支持链路追踪 |
| **安全** | Tool 权限控制，沙箱代码执行，输入校验 |
| **可扩展** | 新 Agent / Tool / Skill / MCP Server 均可插件化注册 |
