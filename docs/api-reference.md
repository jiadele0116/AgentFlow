# Multi-Agent System API 参考文档

> 基于 LangChain + LangGraph + FastAPI 构建的多 Agent 协作系统

---

## 目录

1. [技术栈总览](#1-技术栈总览)
2. [核心模块 (src/core/)](#2-核心模块)
3. [LLM 模块 (src/llm/)](#3-llm-模块)
4. [工具模块 (src/tools/)](#4-工具模块)
5. [Agent 模块 (src/agents/)](#5-agent-模块)
6. [API 模块 (src/api/)](#6-api-模块)
7. [RAG 模块 (src/rag/)](#7-rag-模块)
8. [MCP 模块 (src/mcp/)](#8-mcp-模块)
9. [启动流程](#9-启动流程)
10. [WebSocket 事件协议](#10-websocket-事件协议)

---

## 1. 技术栈总览

### LangChain

LangChain 是 LLM 应用开发框架，提供统一的接口来调用不同的大语言模型。本项目用它来：

- **统一 LLM 调用**：通过 `BaseChatModel` 接口，透明切换 Anthropic / Qwen 等模型
- **工具定义**：通过 `StructuredTool` 将 Python 函数包装为 LLM 可调用的工具
- **消息格式**：使用 `HumanMessage`、`SystemMessage`、`AIMessage` 等标准化消息类型

```python
# LangChain 调用 LLM 的核心方式
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

# 发送消息并获取响应
response = await llm.ainvoke([
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="你好"),
])
print(response.content)  # LLM 的文本回复
print(response.tool_calls)  # LLM 请求的工具调用（如果有）
```

### LangGraph

LangGraph 是 LangChain 生态的状态图框架，用于构建有状态的多步骤 Agent 工作流。核心概念：

- **StateGraph**：有向图，节点是处理函数，边定义流转逻辑
- **State**：TypedDict 定义的共享状态，在节点间传递
- **条件边**：根据状态决定下一步走向（如审查不通过则回退）

```python
from langgraph.graph import StateGraph, END

# 定义状态
class MyState(TypedDict):
    messages: list
    result: str

# 构建图
graph = StateGraph(MyState)
graph.add_node("step1", step1_function)
graph.add_node("step2", step2_function)
graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

# 编译并运行
compiled = graph.compile()
result = await compiled.ainvoke({"messages": [], "result": ""})
```

### FastAPI

FastAPI 是高性能异步 Web 框架，本项目用它提供：

- **HTTP 端点**：健康检查、静态文件服务
- **WebSocket 端点**：实时双向聊天通信
- **Pydantic 集成**：自动请求/响应验证

### 项目架构图

```
用户浏览器
    │
    ▼
FastAPI (WebSocket)
    │
    ▼
ChatSession.handle_message()
    │
    ▼
LangGraph StateGraph
    ├── Planner Node (LLM)
    ├── Executor Node (LLM + Tools)
    └── Critic Node (LLM)
           │
           ├── 通过 → 返回结果
           └── 不通过 → 回到 Executor
```

---

## 2. 核心模块

### 2.1 config.py — 配置管理

负责从 `config.yaml` 加载配置，支持 `${ENV_VAR}` 环境变量替换。

#### 类：`Settings`

系统顶层配置容器，聚合所有子模块配置。

```python
from src.core.config import Settings, load_config, get_settings
from pathlib import Path

# 加载配置
settings = load_config(Path("config.yaml"))

# 访问子配置
settings.llm.model           # "qwen3-rerank"
settings.llm.provider        # "qwen"
settings.llm.api_key         # 环境变量替换后的值
settings.rag.chunk_size      # 512
settings.rag.retrieval_top_k # 5
settings.agents.max_concurrent  # 5
```

#### 子配置类

| 类名 | 用途 | 关键字段 |
|------|------|---------|
| `SystemConfig` | 系统基本信息 | `name`, `version`, `log_level` |
| `LLMConfig` | LLM 模型配置 | `provider`, `model`, `api_key`, `temperature`, `max_tokens` |
| `EmbeddingConfig` | 嵌入模型配置 | `model`, `device`, `batch_size` |
| `VectorStoreConfig` | 向量数据库配置 | `backend`, `persist_dir`, `collection` |
| `RAGConfig` | RAG 检索配置 | `chunk_size`, `chunk_overlap`, `retrieval_top_k`, `rerank` |
| `AgentsConfig` | Agent 运行配置 | `max_concurrent`, `default_strategy`, `memory` |
| `MCPConfig` | MCP 服务器配置 | `servers` (列表) |

#### 函数：`load_config(path: Path) -> Settings`

从 YAML 文件加载配置。

```python
from pathlib import Path
from src.core.config import load_config

settings = load_config(Path("config.yaml"))
# 自动展开 ${ENV_VAR}，Pydantic 验证，存为全局单例
```

**流程**：
1. 读取 YAML 文件
2. `_expand_env_vars()` 递归替换 `${VAR}` 为环境变量值
3. `Settings(**expanded)` Pydantic 验证并构建对象
4. 存储为全局单例（`_settings`）

#### 函数：`get_settings() -> Settings`

获取已加载的全局配置。

```python
from src.core.config import get_settings

settings = get_settings()
print(settings.llm.model)
```

#### 函数：`_expand_env_vars(value: Any) -> Any`

递归展开配置值中的环境变量占位符。

```python
# 输入
"${ANTHROPIC_API_KEY}"
# 输出（假设环境变量 ANTHROPIC_API_KEY=sk-abc123）
"sk-abc123"

# 递归处理嵌套
{"api_key": "${KEY}", "nested": {"db": "${DB_URL}"}}
# → {"api_key": "sk-xxx", "nested": {"db": "postgresql://..."}}
```

---

### 2.2 logging.py — 日志系统

提供两种日志格式：JSON 结构化（生产环境）和人类可读（开发环境）。

#### 类：`StructuredFormatter`

JSON 格式化器，输出示例：
```json
{"timestamp":"2026-06-04T10:30:00+00:00","level":"INFO","logger":"multi_agent.llm","message":"Request completed","trace_id":"abc-123"}
```

支持的额外字段（通过 `extra` 注入）：`trace_id`, `agent_id`, `tool_name`, `skill_name`, `duration_ms`

#### 类：`SimpleFormatter`

人类可读格式：`LEVEL logger_name: message`

#### 函数：`setup_logging(level: str = "INFO", json_format: bool = False)`

初始化日志系统，在启动时调用一次。

```python
from src.core.logging import setup_logging, get_logger

setup_logging(level="INFO", json_format=False)
logger = get_logger("main")
logger.info("System started")  # → INFO multi_agent.main: System started
```

#### 函数：`get_logger(name: str) -> logging.Logger`

获取子日志器，自动在 `multi_agent.` 命名空间下。

```python
logger = get_logger("llm.langchain")  # → multi_agent.llm.langchain
logger.info("LLM created", extra={"trace_id": "abc-123"})
```

---

### 2.3 exceptions.py — 异常体系

统一的异常层级结构，所有异常继承自 `MultiAgentError`。

```
MultiAgentError (基类)
├── AgentError      — Agent 运行时错误（携带 agent_id）
├── ToolError       — 工具执行错误（携带 tool_name）
├── SkillError      — 技能执行错误（携带 skill_name）
├── RAGError        — RAG 检索/索引错误
├── MCPError        — MCP 协议通信错误（携带 server_name）
└── ConfigError     — 配置加载/验证错误
```

#### 类：`MultiAgentError(Exception)`

```python
class MultiAgentError(Exception):
    def __init__(self, message: str, code: str = "", context: dict | None = None):
        super().__init__(message)  # 调用 Exception.__init__，让 str(e) 能显示错误信息
        self.code = code           # 错误码，如 "AGENT_TIMEOUT"
        self.context = context     # 上下文信息，如 {"retry": 3}
```

#### 子类用法

```python
from src.core.exceptions import AgentError, ToolError

# AgentError：携带 agent_id
raise AgentError("LLM 调用失败", agent_id="executor-1", code="LLM_TIMEOUT")

# ToolError：携带 tool_name
raise ToolError("文件不存在", tool_name="read_file", context={"path": "/tmp/xxx"})

# 统一捕获
try:
    ...
except MultiAgentError as e:
    print(e.code, e.context)
```

---

## 3. LLM 模块

### 3.1 langchain_llm.py — LangChain LLM 工厂

根据配置创建对应的 LangChain ChatModel 实例。

#### 函数：`create_langchain_llm(settings: Settings) -> BaseChatModel`

```python
from src.core.config import load_config
from src.llm.langchain_llm import create_langchain_llm
from pathlib import Path

settings = load_config(Path("config.yaml"))
llm = create_langchain_llm(settings)
# → ChatTongyi(model="qwen3-rerank") 或 ChatAnthropic(model="claude-sonnet-4-20250514")
```

**支持的提供商**：

| provider 值 | LangChain 类 | 依赖包 |
|-------------|-------------|--------|
| `"anthropic"` | `ChatAnthropic` | `langchain-anthropic` |
| `"qwen"` | `ChatTongyi` | `langchain-community` |

**返回的 BaseChatModel 提供的核心方法**：

```python
# 完整响应
response = await llm.ainvoke([HumanMessage(content="你好")])
print(response.content)       # 文本内容
print(response.tool_calls)    # 工具调用请求列表

# 流式响应
async for chunk in llm.astream([HumanMessage(content="你好")]):
    print(chunk.content, end="")

# 绑定工具（让 LLM 知道可以调用哪些工具）
llm_with_tools = llm.bind_tools(tools)
response = await llm_with_tools.ainvoke([HumanMessage(content="读取 test.txt")])
```

---

## 4. 工具模块

### 4.1 registry.py — 工具注册表

提供装饰器风格的工具注册机制。

#### 数据类：`Tool`

```python
@dataclass
class Tool:
    name: str                          # 工具唯一标识符
    description: str                   # 功能描述（LLM 据此决定何时调用）
    parameters: dict[str, Any]         # JSON Schema 参数定义
    handler: Callable[..., Awaitable]  # 异步执行函数
    timeout: int = 30                  # 执行超时（秒）
    retry: int = 0                     # 失败重试次数
    requires_confirmation: bool = False # 是否需要用户确认
```

#### 类：`ToolRegistry`

```python
from src.tools.registry import ToolRegistry

registry = ToolRegistry()

# 注册工具（装饰器方式）
@registry.register(
    name="my_tool",
    description="做某件事",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数说明"}
        },
        "required": ["param1"]
    },
    timeout=30,
    retry=2,
)
async def my_tool(param1: str) -> str:
    return f"结果: {param1}"

# 查询工具
tool = registry.get("my_tool")         # 获取 Tool 对象
names = registry.list_names()           # ["my_tool", ...]
schema = registry.list_tools()          # [{name, description, input_schema}, ...]

# 注销
registry.unregister("my_tool")
```

**`list_tools()` 返回格式**（直接传给 LLM 的 tools 参数）：
```json
[
  {
    "name": "read_file",
    "description": "Read the contents of a file",
    "input_schema": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  }
]
```

---

### 4.2 langchain_tools.py — 工具转换桥

将 ToolRegistry 中的工具转换为 LangChain `StructuredTool`。

#### 函数：`convert_to_langchain_tools(registry: ToolRegistry) -> list[BaseTool]`

```python
from src.tools.registry import ToolRegistry
from src.tools.langchain_tools import convert_to_langchain_tools
from src.tools.builtin import register_all_builtin_tools

registry = ToolRegistry()
register_all_builtin_tools(registry, workspace_dir="./data/workspace")

# 转换为 LangChain 工具
lc_tools = convert_to_langchain_tools(registry)
# → [StructuredTool(name="read_file", ...), StructuredTool(name="write_file", ...), ...]

# 这些工具可以直接传给 LangGraph
from src.agents.langgraph_multi import create_multi_agent_graph
graph = create_multi_agent_graph(llm, lc_tools)
```

**转换原理**：
```python
# 每个 Tool 被包装为 StructuredTool
StructuredTool.from_function(
    coroutine=tool.handler,     # 异步执行函数
    name=tool.name,             # 工具名
    description=tool.description,  # 描述
    return_direct=False,
)
```

---

### 4.3 内置工具 (src/tools/builtin/)

#### file_ops.py — 文件操作

```python
from src.tools.builtin.file_ops import register_file_tools

registry = ToolRegistry()
register_file_tools(registry)
```

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `read_file` | `path: str` | `str` | 读取文件的 UTF-8 文本内容 |
| `write_file` | `path: str, content: str` | `bool` | 写入文件，自动创建父目录 |
| `search_files` | `pattern: str, directory: str` | `list[str]` | 按 glob 模式递归搜索文件 |

```python
# LLM 调用示例
result = await registry.get("read_file").handler(path="/tmp/test.txt")
# → "文件内容..."

result = await registry.get("search_files").handler(pattern="*.py", directory="./src")
# → ["src/main.py", "src/core/config.py", ...]
```

#### data_ops.py — 数据处理

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `parse_json` | `text: str` | `Any` | 解析 JSON 字符串 |
| `csv_to_dict` | `csv_text: str` | `list[dict]` | CSV 文本 → 字典列表 |
| `json_to_csv` | `data: list[dict]` | `str` | 字典列表 → CSV 文本 |

```python
# parse_json
result = await registry.get("parse_json").handler(text='{"name": "张三"}')
# → {"name": "张三"}

# csv_to_dict
result = await registry.get("csv_to_dict").handler(csv_text="name,age\n张三,25\n李四,30")
# → [{"name": "张三", "age": "25"}, {"name": "李四", "age": "30"}]
```

#### code_exec.py — 代码执行

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `run_python` | `code: str, timeout: int = 30` | `ExecutionResult` | 在受限沙箱中执行 Python |

```python
@dataclass
class ExecutionResult:
    success: bool      # 是否成功
    output: str = ""   # 标准输出
    error: str = ""    # 错误信息
    variables: dict    # 变量（预留）
```

**安全限制**：只允许安全的内置函数（`print`, `range`, `len`, `str`, `int`, `list`, `dict` 等），禁止 `os`, `subprocess`, `open` 等危险操作。

```python
result = await registry.get("run_python").handler(code="print(2 + 3)")
# → ExecutionResult(success=True, output="5\n")
```

#### network.py — 网络请求

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `fetch_url` | `url, method="GET", headers=None, body=None` | `dict` | HTTP 请求 |

返回值格式：
```json
{
  "status": 200,
  "body": "<!DOCTYPE html>...",
  "headers": {"content-type": "text/html", ...}
}
```

#### rag_ops.py — RAG 操作

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `search_knowledge` | `query, collection="default", top_k=5` | `list[dict]` | 知识库语义搜索 |
| `index_document` | `source, collection="default"` | `int` | 文档索引到知识库 |

> 这两个工具需要额外配置 `retrieval_pipeline` 和 `document_processor` 才能使用。

---

### 4.4 __init__.py — 统一注册入口

```python
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_all_builtin_tools

registry = ToolRegistry()
register_all_builtin_tools(
    registry,
    workspace_dir="./data/workspace",      # 代码执行的工作目录
    retrieval_pipeline=None,                # RAG 检索管道（可选）
    document_processor=None,                # 文档处理器（可选）
)
# 一次性注册所有 10 个内置工具
```

---

## 5. Agent 模块

### 5.1 langgraph_multi.py — 多 Agent 协作图

构建 **Planner → Executor → Critic** 的多 Agent 工作流。

#### 状态定义：`MultiAgentState`

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史（自动追加）
    task: str                       # 原始用户任务
    plan: str                       # Planner 生成的计划
    execution_result: str           # Executor 的执行结果
    review_status: str              # "pending" / "approved" / "rejected"
    iterations: int                 # 修订轮次计数
```

**字段说明**：
- `messages`：使用 `add_messages` reducer，新消息自动追加到列表末尾
- `task`：用户输入的原始任务文本
- `plan`：Planner 节点生成的执行计划
- `execution_result`：Executor 节点的执行结果
- `review_status`：Critic 节点的审查结论
- `iterations`：修订轮次计数，超过 2 轮强制结束

#### 节点函数

**`planner_node(state, llm) -> dict`**

Planner 节点：调用 LLM 分析任务，生成执行计划。

```python
async def planner_node(state: MultiAgentState, llm: BaseChatModel) -> dict:
    task = state["task"]
    messages = [
        SystemMessage(content="你是一个任务规划专家..."),
        HumanMessage(content=task),
    ]
    response = await llm.ainvoke(messages)
    return {
        "plan": response.content,
        "messages": [SystemMessage(content=f"[Planner 计划]\n{response.content}")],
    }
```

**`executor_node(state, llm, tools) -> dict`**

Executor 节点：根据计划执行任务，可调用工具。

```python
async def executor_node(state: MultiAgentState, llm: BaseChatModel, tools: list[BaseTool]) -> dict:
    task = state["task"]
    plan = state.get("plan", "")
    
    # 如果被 Critic 拒绝，加入修订指令
    revision_note = ""
    if state.get("review_status") == "rejected":
        revision_note = "\n\n注意：上一版结果未通过审查，请改进。"
    
    # 绑定工具，让 LLM 可以调用
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    response = await llm_with_tools.ainvoke(messages)
    
    # 如果 LLM 请求了工具调用，执行工具
    if response.tool_calls:
        for tc in response.tool_calls:
            tool = next(t for t in tools if t.name == tc["name"])
            result = await tool.ainvoke(tc["args"])
            # 把工具结果反馈给 LLM 生成最终答案
    
    return {"execution_result": result, "messages": [...]}
```

**`critic_node(state, llm) -> dict`**

Critic 节点：审查执行结果质量。

```python
async def critic_node(state: MultiAgentState, llm: BaseChatModel) -> dict:
    response = await llm.ainvoke([
        SystemMessage(content="审查结果是否合格，回复 APPROVED 或 REJECTED + 原因"),
        HumanMessage(content=f"任务：{task}\n结果：{result}"),
    ])
    
    if response.content.strip().upper().startswith("APPROVED"):
        return {"review_status": "approved", "iterations": ...}
    else:
        return {"review_status": "rejected", "iterations": ...}
```

#### 条件路由：`should_continue(state) -> str`

```python
def should_continue(state: MultiAgentState) -> str:
    if state.get("review_status") == "approved":
        return "end"           # 审查通过 → 结束
    if state.get("iterations", 0) >= 2:
        return "end"           # 超过 2 轮 → 强制结束
    if state.get("review_status") == "pending":
        return "critic"        # 未审查 → 去审查
    return "executor"          # 被拒绝 → 重新执行
```

#### 图构建：`create_multi_agent_graph(llm, tools) -> CompiledStateGraph`

```python
from src.agents.langgraph_multi import create_multi_agent_graph

graph = create_multi_agent_graph(llm, lc_tools)
```

**图结构**：
```
[planner] ──→ [executor] ──→ [critic] ──→ should_continue
                                 │              │
                                 │         ┌────┴────┐
                                 │      "end"    "executor"
                                 │         │        │
                                 ▼         ▼        │
                                END      END    [executor] ←┘
```

**使用方式**：

```python
# 同步调用
result = await graph.ainvoke({
    "task": "分析这段代码的性能问题",
    "messages": [],
    "review_status": "pending",
    "iterations": 0,
})

# 流式调用（获取中间事件）
async for event in graph.astream_events(input_state, version="v2"):
    if event["event"] == "on_chain_start":
        print(f"节点开始: {event['name']}")
    elif event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

### 5.2 langgraph_react.py — ReAct Agent

使用 LangGraph 内置的 `create_react_agent` 创建单 Agent 工具调用循环。

#### 函数：`create_react_agent_graph(llm, tools, system_prompt=None) -> CompiledStateGraph`

```python
from src.agents.langgraph_react import create_react_agent_graph

graph = create_react_agent_graph(
    llm=llm,
    tools=lc_tools,
    system_prompt="你是一个智能助手...",  # 可选
)

# 调用
result = await graph.ainvoke({
    "messages": [HumanMessage(content="帮我读取 config.yaml")]
})
```

**ReAct 循环**：
```
用户输入 → LLM 推理(Thought) → 决定调用工具(Action) → 执行工具 → 获取结果(Observation)
    → LLM 继续推理 → ... → 直到 LLM 给出最终答案（不再调用工具）
```

**与多 Agent 图的区别**：

| 特性 | ReAct Agent | 多 Agent 图 |
|------|-------------|------------|
| Agent 数量 | 1 个 | 3 个（Planner/Executor/Critic） |
| 适用场景 | 简单问答+工具调用 | 复杂任务分解+质量审查 |
| 审查机制 | 无 | Critic 审查，不通过则修订 |
| 图结构 | 内置循环 | 自定义节点+条件边 |

---

## 6. API 模块

### 6.1 server.py — FastAPI 应用

#### 函数：`create_app(graph) -> FastAPI`

创建 FastAPI 应用实例。

```python
from src.api.server import create_app

app = create_app(graph=compiled_graph)
```

**注册的路由**：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回聊天页面（index.html） |
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| WebSocket | `/ws/chat` | 聊天 WebSocket 端点 |

---

### 6.2 chat.py — WebSocket 聊天处理

#### 类：`ChatSession`

单个聊天会话，维护对话历史并通过 LangGraph 执行。

```python
class ChatSession:
    def __init__(self, session_id: str, graph):
        self.session_id = session_id    # 会话唯一 ID
        self._graph = graph             # LangGraph 编译好的图
        self._history = []              # 对话历史

    async def handle_message(self, websocket: WebSocket, user_message: str):
        """处理用户消息，流式返回 Agent 执行结果。"""
```

**`handle_message` 内部流程**：
1. 构造 `MultiAgentState` 输入
2. 调用 `self._graph.astream_events()` 流式执行
3. 根据事件类型发送 WebSocket 消息：
   - `on_chain_start` → `agent_status`（Agent 开始/完成）
   - `on_chat_model_stream` → `chunk`（LLM 流式文本）
   - `on_tool_start` → `tool_call`（工具调用开始）
   - `on_tool_end` → `tool_result`（工具调用结束）
4. 执行完成后发送 `done`

#### 类：`ChatManager`

管理多个聊天会话。

```python
class ChatManager:
    def __init__(self, graph):
        self._graph = graph
        self._sessions = {}

    def create_session(self) -> ChatSession:
        """创建新会话，返回 ChatSession。"""

    def get_session(self, session_id: str) -> ChatSession | None:
        """获取已有会话。"""

    def remove_session(self, session_id: str):
        """移除会话。"""
```

#### 函数：`handle_websocket_chat(websocket, chat_manager)`

WebSocket 端点处理函数。

```python
async def handle_websocket_chat(websocket: WebSocket, chat_manager: ChatManager):
    await websocket.accept()
    session = chat_manager.create_session()
    
    # 发送系统信息（Agent 列表等）
    await send_event(websocket, "system_info", agents=AGENTS, tools=[])
    
    # 消息循环
    while True:
        data = await websocket.receive_json()
        await session.handle_message(websocket, data["content"])
```

---

### 6.3 models.py — 数据模型

```python
from pydantic import BaseModel

class ChatMessage(BaseModel):
    content: str          # 用户消息内容

class ChatResponse(BaseModel):
    content: str          # Agent 回复内容
    role: str = "assistant"
```

---

## 7. RAG 模块

### 7.1 chunker.py — 文档分块

将长文档切分为小块，便于向量化和检索。

#### 类：`TextChunker`

```python
from src.rag.chunker import TextChunker

chunker = TextChunker(chunk_size=512, chunk_overlap=64)
chunks = chunker.chunk(document_text)
# → ["chunk1 text", "chunk2 text", ...]
```

**分块策略**：
- 按字符数切分（默认 512 字符）
- 相邻块重叠 64 字符，确保上下文连续性
- 支持按段落、句子边界切分

### 7.2 embedder.py — 文本向量化

将文本转换为向量表示。

#### 类：`Embedder`

```python
from src.rag.embedder import Embedder

embedder = Embedder(model="BAAI/bge-m3", device="cpu")
embeddings = await embedder.embed_batch(["text1", "text2"])
# → [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

### 7.3 vector_store.py — 向量存储抽象

定义向量数据库的抽象接口。

#### 类：`VectorStore(ABC)`

```python
class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, chunks, embeddings): ...
    
    @abstractmethod
    async def search(self, query_embedding, top_k, filters=None): ...
    
    @abstractmethod
    async def delete(self, ids): ...
```

### 7.4 retrieval.py — 检索管线

完整的 RAG 检索流程。

#### 类：`RetrievalPipeline`

```python
from src.rag.retrieval import RetrievalPipeline

pipeline = RetrievalPipeline(vector_store, embedder)
results = await pipeline.retrieve("什么是 RAG?", top_k=5)
# → [SearchResult(text="...", score=0.95, metadata={...}), ...]
```

---

## 8. MCP 模块

### 8.1 protocol.py — MCP 协议定义

定义 MCP 消息格式。

```python
@dataclass
class MCPRequest:
    method: str           # "tools/call", "resources/read"
    params: dict          # 方法参数
    request_id: str       # 请求 ID

@dataclass
class MCPResponse:
    request_id: str
    result: dict | None
    error: MCPError | None
```

### 8.2 server.py — MCP 服务器基类

```python
class MCPServer:
    name: str
    version: str
    resources: list[Resource]
    tools: list[Tool]
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse: ...
```

### 8.3 client.py — MCP 客户端

```python
class MCPClient:
    async def call_tool(self, server, tool_name, arguments) -> MCPResponse: ...
    async def read_resource(self, server, uri) -> MCPResponse: ...
```

---

## 9. 启动流程

`src/main.py` 的完整启动流程：

```python
async def main():
    # 1. 加载配置
    settings = load_config(Path("config.yaml"))
    
    # 2. 初始化日志
    setup_logging(level=settings.system.log_level)
    logger = get_logger("main")
    
    # 3. 创建 LangChain LLM
    langchain_llm = create_langchain_llm(settings)
    # → ChatTongyi 或 ChatAnthropic
    
    # 4. 注册内置工具并转换为 LangChain 格式
    registry = ToolRegistry()
    register_all_builtin_tools(registry, workspace_dir="./data/workspace")
    lc_tools = convert_to_langchain_tools(registry)
    # → [StructuredTool("read_file"), StructuredTool("write_file"), ...]
    
    # 5. 构建 LangGraph 多 Agent 协作图
    graph = create_multi_agent_graph(langchain_llm, lc_tools)
    # → CompiledStateGraph (Planner → Executor → Critic)
    
    # 6. 启动 Web 服务
    app = create_app(graph=graph)
    server = uvicorn.Config(app, host="0.0.0.0", port=8000)
    await server.serve()
```

**命令行启动**：
```bash
python -m src.main
# 输出：
# Starting Multi-Agent System v0.1.0
# LangChain LLM: qwen / qwen3-rerank
# Registered 10 tools
# Converted 10 tools to LangChain format
# Multi-agent graph compiled
# Web server starting at http://localhost:8000
```

---

## 10. WebSocket 事件协议

前端和后端通过 WebSocket JSON 消息通信。

### 前端 → 后端

```json
{"content": "用户输入的消息"}
```

### 后端 → 前端

| 事件类型 | 字段 | 说明 |
|---------|------|------|
| `system_info` | `agents`, `tools`, `session_id` | 连接时发送系统信息 |
| `agent_status` | `agent`, `status` | Agent 状态变化（executing/completed） |
| `tool_call` | `tool`, `args` | 工具调用开始 |
| `tool_result` | `tool`, `success` | 工具调用结束 |
| `chunk` | `content` | LLM 流式文本片段 |
| `done` | `content` | 完整响应完成 |
| `error` | `content` | 错误信息 |

**示例事件流**：
```json
{"type": "agent_status", "agent": "planner", "status": "executing"}
{"type": "chunk", "content": "分析"}
{"type": "chunk", "content": "任务"}
{"type": "agent_status", "agent": "planner", "status": "completed"}
{"type": "agent_status", "agent": "executor", "status": "executing"}
{"type": "tool_call", "tool": "read_file", "args": {"path": "test.py"}}
{"type": "tool_result", "tool": "read_file", "success": true}
{"type": "chunk", "content": "执行结果"}
{"type": "agent_status", "agent": "executor", "status": "completed"}
{"type": "agent_status", "agent": "critic", "status": "executing"}
{"type": "agent_status", "agent": "critic", "status": "completed"}
{"type": "done", "content": "完整回复内容"}
```
