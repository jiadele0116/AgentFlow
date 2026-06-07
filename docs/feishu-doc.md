# Multi-Agent System 技术文档

> 基于 LangChain + LangGraph + FastAPI 的多 Agent 协作系统

---

## 一、项目概述

### 1.1 这是什么

Multi-Agent System 是一个多智能体协作框架，多个 AI Agent 分工协作完成复杂任务。

**核心能力**：
- 🤖 多 Agent 协作：Planner 规划 → Executor 执行 → Critic 审查
- 🔧 工具调用：10 个内置工具（文件、代码、网络、数据、RAG）
- 🌐 Web 聊天界面：Stripe 风格前端，实时流式对话
- 📚 知识检索：RAG 检索增强生成

### 1.2 技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| **LangChain** | LLM 调用统一接口、工具定义 | ≥0.3.0 |
| **LangGraph** | 多 Agent 状态图工作流 | ≥0.2.0 |
| **FastAPI** | Web 服务 + WebSocket | ≥0.111.0 |
| **Pydantic** | 配置验证、数据模型 | ≥2.7.0 |
| **ChromaDB** | 向量数据库 | ≥0.5.0 |

### 1.3 系统架构

```
┌─────────────────────────────────────────────────┐
│                 用户浏览器                        │
│            Stripe 风格聊天界面                     │
└────────────────────┬────────────────────────────┘
                     │ WebSocket
                     ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Web 服务                     │
│         server.py + chat.py                      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           LangGraph 多 Agent 协作图               │
│                                                   │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│   │ Planner  │──→│ Executor │──→│  Critic  │    │
│   │  任务规划  │   │  任务执行  │   │  质量审查  │    │
│   └──────────┘   └────┬─────┘   └────┬─────┘    │
│                       │              │           │
│                       │    不通过 ←───┘           │
│                       └──────────────────→ 修订  │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌──────────┐ ┌────────┐ ┌─────────┐
   │ LangChain│ │ Tools  │ │   RAG   │
   │  LLM API │ │ 10个工具│ │ 知识检索  │
   └──────────┘ └────────┘ └─────────┘
```

---

## 二、快速开始

### 2.1 环境准备

```bash
# Python 3.11+
python --version

# 克隆项目
git clone <repo-url>
cd multi-agent-system
```

### 2.2 安装依赖

```bash
pip install -e .
```

### 2.3 配置环境变量

```bash
# 根据 config.yaml 中的 provider 设置对应的 API Key
export ANTHROPIC_API_KEY="your-api-key"    # Anthropic
# 或
export DASHSCOPE_API_KEY="your-api-key"    # Qwen/通义千问
```

### 2.4 启动服务

```bash
python -m src.main
```

输出：
```
Starting Multi-Agent System v0.1.0
LangChain LLM: qwen / qwen3-rerank
Registered 10 tools
Converted 10 tools to LangChain format
Multi-agent graph compiled
Web server starting at http://localhost:8000
```

### 2.5 访问

浏览器打开 `http://localhost:8000`，即可开始对话。

---

## 三、核心概念

在深入了解每个模块之前，先理解三个核心概念。

### 3.1 什么是 LangChain

LangChain 是 LLM 应用的统一接口层。不同厂商的 LLM（Anthropic、OpenAI、通义千问）API 各不相同，LangChain 提供统一的 `BaseChatModel` 接口，让你不用关心底层差异。

```python
# 不用 LangChain：每个厂商写一套
anthropic_client.messages.create(...)
dashscope.Generation.call(...)

# 用 LangChain：统一接口
from langchain_core.language_models import BaseChatModel

async def ask(llm: BaseChatModel, question: str):
    response = await llm.ainvoke([HumanMessage(content=question)])
    return response.content

# 传入 ChatAnthropic 或 ChatTongyi 都可以
answer = await ask(llm, "什么是 RAG?")
```

**本项目中 LangChain 负责**：
- 统一 LLM 调用（`src/llm/langchain_llm.py`）
- 工具定义（`src/tools/langchain_tools.py`）
- 消息格式（`HumanMessage`, `SystemMessage`, `AIMessage`）

### 3.2 什么是 LangGraph

LangGraph 是 LangChain 生态的状态图框架，用于构建**有状态的多步骤工作流**。

普通的 LLM 调用是一问一答。但多 Agent 协作需要：先规划 → 再执行 → 再审查 → 不通过还要回退。这就是 LangGraph 解决的问题。

```python
from langgraph.graph import StateGraph, END

# 1. 定义状态（在节点之间传递的数据）
class MyState(TypedDict):
    task: str
    result: str
    approved: bool

# 2. 定义节点（处理函数）
async def do_task(state):
    return {"result": "完成了"}

async def review(state):
    return {"approved": True}

# 3. 构建图（节点 + 边）
graph = StateGraph(MyState)
graph.add_node("worker", do_task)
graph.add_node("reviewer", review)
graph.set_entry_point("worker")
graph.add_edge("worker", "reviewer")
graph.add_conditional_edges("reviewer", lambda s: "end" if s["approved"] else "worker")

# 4. 编译并运行
app = graph.compile()
result = await app.ainvoke({"task": "分析代码", "result": "", "approved": False})
```

**本项目中 LangGraph 负责**：
- 多 Agent 协作图（`src/agents/langgraph_multi.py`）
- ReAct 单 Agent 循环（`src/agents/langgraph_react.py`）

### 3.3 什么是 FastAPI

FastAPI 是 Python 异步 Web 框架，本项目用它提供 HTTP 和 WebSocket 服务。

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        await ws.send_json({"reply": "收到"})
```

**本项目中 FastAPI 负责**：
- 提供聊天页面（`src/api/server.py`）
- WebSocket 实时通信（`src/api/chat.py`）

---

## 四、配置系统详解

### 4.1 配置文件结构

`config.yaml` 是系统的唯一配置入口：

```yaml
system:
  name: "Multi-Agent System"
  version: "0.1.0"
  log_level: INFO

llm:
  provider: qwen              # anthropic / qwen
  model: qwen3-rerank         # 模型 ID
  api_key: ${ANTHROPIC_API_KEY}  # 支持环境变量替换
  temperature: 0.7             # 生成温度 (0-1)
  max_tokens: 4096             # 最大 token 数
  timeout: 60                  # 请求超时（秒）

embedding:
  model: qwen3-rerank
  device: cpu                  # cpu / cuda
  batch_size: 32

vector_store:
  backend: chroma              # chroma / faiss
  persist_dir: ./data/chroma
  collection: default

rag:
  chunk_size: 512              # 分块大小（字符）
  chunk_overlap: 64            # 块重叠（字符）
  retrieval_top_k: 5           # 返回最相关的 K 个结果
  rerank: true                 # 是否重排序

agents:
  max_concurrent: 5            # 最大并发 Agent 数
  default_strategy: react      # 默认策略
  memory:
    short_term_max: 20         # 短期记忆轮数
    long_term_backend: sqlite  # 长期记忆后端

mcp:
  servers:
    - name: filesystem
      enabled: true
      root_dir: ./data/workspace
```

### 4.2 配置加载流程

```python
from pathlib import Path
from src.core.config import load_config, get_settings

# 步骤 1：读取 YAML
# 步骤 2：替换 ${ENV_VAR} 为环境变量
# 步骤 3：Pydantic 验证类型
# 步骤 4：存为全局单例
settings = load_config(Path("config.yaml"))

# 之后任何地方都可以获取
settings = get_settings()
print(settings.llm.model)  # "qwen3-rerank"
```

### 4.3 配置类层级

```
Settings（顶层容器）
├── system: SystemConfig          系统名称、版本、日志级别
├── llm: LLMConfig                LLM 提供商、模型、API Key
├── embedding: EmbeddingConfig    嵌入模型、设备、批次大小
├── vector_store: VectorStoreConfig  向量数据库后端、持久化目录
├── rag: RAGConfig                分块大小、top_k、是否重排序
├── agents: AgentsConfig          并发数、策略、记忆配置
└── mcp: MCPConfig                MCP 服务器列表
```

每个子配置都是 Pydantic BaseModel，自动做类型验证：

```python
class LLMConfig(BaseModel):
    provider: str = "qwen"
    model: str = "qwen3-rerank"
    api_key: str = ""
    temperature: float = 0.7      # 必须是浮点数
    max_tokens: int = 4096        # 必须是整数
    timeout: int = 60
```

### 4.4 环境变量替换

配置中 `${VAR_NAME}` 会被替换为环境变量值：

```yaml
llm:
  api_key: ${ANTHROPIC_API_KEY}
```

替换逻辑（`_expand_env_vars` 函数）：
- 字符串：正则匹配 `${...}` 并替换
- 字典：递归处理每个 value
- 列表：递归处理每个元素
- 其他类型：原样返回

---

## 五、LLM 模块详解

### 5.1 模块职责

将系统配置转换为 LangChain 可用的 ChatModel 实例。

### 5.2 工厂函数

```python
# src/llm/langchain_llm.py

def create_langchain_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm.provider.lower()
    
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm.model,
            anthropic_api_key=settings.llm.api_key,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
    
    elif provider == "qwen":
        from langchain_community.chat_models.tongyi import ChatTongyi
        return ChatTongyi(
            model=settings.llm.model,
            dashscope_api_key=settings.llm.api_key,
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )
    
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
```

### 5.3 支持的提供商

| provider | LangChain 类 | 适用场景 |
|----------|-------------|---------|
| `anthropic` | `ChatAnthropic` | Claude 系列模型 |
| `qwen` | `ChatTongyi` | 通义千问系列模型 |

### 5.4 BaseChatModel 核心方法

创建好的 LLM 实例提供以下方法：

```python
# 1. 完整调用 — 一次性返回全部内容
response = await llm.ainvoke([HumanMessage(content="你好")])
print(response.content)       # 文本内容
print(response.tool_calls)    # 工具调用请求（如有）

# 2. 流式调用 — 逐块返回（适合实时显示）
async for chunk in llm.astream([HumanMessage(content="你好")]):
    print(chunk.content, end="")

# 3. 绑定工具 — 让 LLM 知道可以调用哪些工具
llm_with_tools = llm.bind_tools([read_file_tool, write_file_tool])
response = await llm_with_tools.ainvoke([HumanMessage(content="读取 config.yaml")])
# response.tool_calls 可能包含 [{"name": "read_file", "args": {"path": "config.yaml"}}]
```

---

## 六、工具系统详解

### 6.1 工具是什么

工具是 Agent 能力的最小单元。LLM 本身只能生成文本，通过工具它可以：读写文件、执行代码、访问网络、搜索知识库。

### 6.2 工具注册机制

```python
# src/tools/registry.py

from src.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.register(
    name="read_file",                              # 工具名（唯一标识）
    description="Read the contents of a file",     # 描述（LLM 据此决定何时调用）
    parameters={                                   # JSON Schema 参数定义
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        },
        "required": ["path"]
    },
    timeout=30,           # 执行超时（秒）
    retry=0,              # 失败重试次数
)
async def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()
```

装饰器做了什么：
1. 创建 `Tool` 数据类（name + description + parameters + handler）
2. 存入 `registry._tools` 字典
3. 返回原函数（不影响直接调用）

### 6.3 Tool 数据类

```python
@dataclass
class Tool:
    name: str                          # 工具名
    description: str                   # 功能描述
    parameters: dict[str, Any]         # JSON Schema
    handler: Callable[..., Awaitable]  # 异步执行函数
    timeout: int = 30                  # 超时（秒）
    retry: int = 0                     # 重试次数
    requires_confirmation: bool = False # 危险操作标记
```

### 6.4 工具转换桥

LangChain 使用 `StructuredTool` 而不是原始的 `Tool`。`langchain_tools.py` 负责转换：

```python
# src/tools/langchain_tools.py

def convert_to_langchain_tools(registry: ToolRegistry) -> list[BaseTool]:
    tools = []
    for name in registry.list_names():
        tool = registry.get(name)
        lc_tool = StructuredTool.from_function(
            coroutine=tool.handler,     # 异步函数
            name=tool.name,
            description=tool.description,
        )
        tools.append(lc_tool)
    return tools
```

转换后，工具可以直接传给 LangGraph 的 Agent 使用。

### 6.5 10 个内置工具

#### 文件操作（file_ops.py）

| 工具 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `read_file` | `path: str` | `str` | 读取文件文本内容 |
| `write_file` | `path: str, content: str` | `bool` | 写入文件（自动创建目录） |
| `search_files` | `pattern: str, directory: str` | `list[str]` | glob 模式搜索文件 |

#### 数据处理（data_ops.py）

| 工具 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `parse_json` | `text: str` | `Any` | 解析 JSON 字符串 |
| `csv_to_dict` | `csv_text: str` | `list[dict]` | CSV → 字典列表 |
| `json_to_csv` | `data: list[dict]` | `str` | 字典列表 → CSV |

#### 代码执行（code_exec.py）

| 工具 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `run_python` | `code: str, timeout: int` | `ExecutionResult` | 沙箱中执行 Python |

安全限制：只允许安全的内置函数，禁止 `os`、`subprocess`、`open` 等。

```python
@dataclass
class ExecutionResult:
    success: bool      # 是否成功
    output: str = ""   # 标准输出
    error: str = ""    # 错误信息
```

#### 网络请求（network.py）

| 工具 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `fetch_url` | `url, method="GET", headers=None, body=None` | `dict` | HTTP 请求 |

返回格式：`{"status": 200, "body": "...", "headers": {...}}`

#### RAG 操作（rag_ops.py）

| 工具 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `search_knowledge` | `query, collection, top_k` | `list[dict]` | 知识库语义搜索 |
| `index_document` | `source, collection` | `int` | 文档索引到知识库 |

### 6.6 统一注册

```python
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_all_builtin_tools

registry = ToolRegistry()
register_all_builtin_tools(
    registry,
    workspace_dir="./data/workspace",
    retrieval_pipeline=None,      # RAG 检索管道（可选）
    document_processor=None,      # 文档处理器（可选）
)
# 一次性注册全部 10 个工具
```

---

## 七、Agent 系统详解

### 7.1 多 Agent 协作图

这是系统的核心——三个 Agent 分工协作完成任务。

#### 状态定义

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
    task: str                       # 用户任务
    plan: str                       # Planner 的计划
    execution_result: str           # Executor 的结果
    review_status: str              # "pending" / "approved" / "rejected"
    iterations: int                 # 修订轮次
```

`Annotated[list[BaseMessage], add_messages]` 的意思是：当节点返回 `{"messages": [new_msg]}` 时，`add_messages` 会自动把 `new_msg` 追加到现有列表末尾，而不是替换。

#### Planner 节点

职责：分析任务，生成执行计划。

```python
async def planner_node(state, llm):
    response = await llm.ainvoke([
        SystemMessage(content="你是任务规划专家，制定清晰的执行计划"),
        HumanMessage(content=state["task"]),
    ])
    return {
        "plan": response.content,
        "messages": [SystemMessage(content=f"[计划]\n{response.content}")],
    }
```

#### Executor 节点

职责：根据计划执行任务，可以调用工具。

```python
async def executor_node(state, llm, tools):
    # 绑定工具，让 LLM 可以调用
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(messages)
    
    # 如果 LLM 请求了工具调用
    if response.tool_calls:
        for tc in response.tool_calls:
            tool = next(t for t in tools if t.name == tc["name"])
            result = await tool.ainvoke(tc["args"])
            # 把工具结果反馈给 LLM 生成最终答案
    
    return {"execution_result": final_result}
```

#### Critic 节点

职责：审查执行结果质量。

```python
async def critic_node(state, llm):
    response = await llm.ainvoke([
        SystemMessage(content="审查结果，回复 APPROVED 或 REJECTED + 原因"),
        HumanMessage(content=f"任务：{task}\n结果：{result}"),
    ])
    
    if "APPROVED" in response.content.upper():
        return {"review_status": "approved"}
    else:
        return {"review_status": "rejected"}
```

#### 条件路由

```python
def should_continue(state):
    if state["review_status"] == "approved":
        return "end"            # 通过 → 结束
    if state["iterations"] >= 2:
        return "end"            # 超过 2 轮 → 强制结束
    return "executor"           # 不通过 → 重新执行
```

#### 图构建

```python
graph = StateGraph(MultiAgentState)

# 添加节点
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("critic", critic_node)

# 定义边
graph.set_entry_point("planner")          # 入口
graph.add_edge("planner", "executor")     # 规划完 → 执行
graph.add_edge("executor", "critic")      # 执行完 → 审查
graph.add_conditional_edges("critic", should_continue, {
    "end": END,                            # 通过 → 结束
    "executor": "executor",                # 不通过 → 重新执行
})

compiled = graph.compile()
```

#### 完整流程图

```
用户输入: "分析这段代码的性能问题"
         │
         ▼
    ┌─────────┐
    │ Planner │  LLM 分析任务，生成 3 步计划
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Executor│  调用 read_file 读取代码
    │         │  调用 run_python 执行分析
    │         │  生成分析报告
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Critic  │  审查报告质量
    └────┬────┘
         │
    ┌────┴────┐
    │         │
 APPROVED  REJECTED
    │         │
    ▼         ▼
  返回结果   回到 Executor 修订
```

### 7.2 ReAct Agent

单 Agent 模式，适合简单任务：LLM 推理 → 调用工具 → 继续推理 → 直到给出答案。

```python
from src.agents.langgraph_react import create_react_agent_graph

graph = create_react_agent_graph(llm, tools, system_prompt="你是一个助手")

result = await graph.ainvoke({
    "messages": [HumanMessage(content="帮我读取 config.yaml")]
})
```

#### ReAct 循环

```
用户: "帮我读取 config.yaml"
         │
    LLM 推理: "需要调用 read_file 工具"
         │
    调用工具: read_file(path="config.yaml")
         │
    获取结果: "system:\n  name: ..."
         │
    LLM 推理: "有了文件内容，可以回复用户了"
         │
    最终回答: "config.yaml 的内容如下: ..."
```

#### 两种 Agent 对比

| | ReAct Agent | 多 Agent 协作图 |
|--|-------------|----------------|
| Agent 数 | 1 个 | 3 个 |
| 适用场景 | 简单问答 + 工具调用 | 复杂任务分解 |
| 质量审查 | ❌ 无 | ✅ Critic 审查 |
| 任务规划 | ❌ 无（LLM 自行决定） | ✅ Planner 分解 |
| 修订机制 | ❌ 无 | ✅ 不通过可回退 |

---

## 八、Web 服务详解

### 8.1 FastAPI 应用

```python
# src/api/server.py

def create_app(graph) -> FastAPI:
    app = FastAPI(title="Multi-Agent System")
    chat_manager = ChatManager(graph)     # 聊天管理器

    @app.get("/")                         # 聊天页面
    async def index(): ...

    @app.get("/health")                   # 健康检查
    async def health():
        return {"status": "ok"}

    @app.websocket("/ws/chat")            # WebSocket 聊天
    async def websocket_chat(websocket):
        await handle_websocket_chat(websocket, chat_manager)

    return app
```

### 8.2 聊天会话管理

```python
# src/api/chat.py

class ChatManager:
    """管理多个并发聊天会话。"""
    def __init__(self, graph):
        self._graph = graph
        self._sessions = {}

    def create_session(self) -> ChatSession:
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, self._graph)
        self._sessions[session_id] = session
        return session

class ChatSession:
    """单个聊天会话。"""
    def __init__(self, session_id, graph):
        self.session_id = session_id
        self._graph = graph          # LangGraph 编译好的图
        self._history = []           # 对话历史

    async def handle_message(self, websocket, user_message):
        # 1. 构造 LangGraph 输入状态
        input_state = {
            "task": user_message,
            "messages": [HumanMessage(content=user_message)],
            "review_status": "pending",
            "iterations": 0,
        }

        # 2. 流式执行 LangGraph
        async for event in self._graph.astream_events(input_state, version="v2"):
            # 根据事件类型发送 WebSocket 消息
            if event["event"] == "on_chain_start":
                await send_event(websocket, "agent_status", ...)
            elif event["event"] == "on_chat_model_stream":
                await send_event(websocket, "chunk", ...)
            elif event["event"] == "on_tool_start":
                await send_event(websocket, "tool_call", ...)

        # 3. 发送完成信号
        await send_event(websocket, "done", content=full_response)
```

### 8.3 WebSocket 事件协议

#### 前端 → 后端

```json
{"content": "用户输入的消息"}
```

#### 后端 → 前端（7 种事件）

| 事件 | 字段 | 说明 |
|------|------|------|
| `system_info` | `agents`, `tools`, `session_id` | 连接时发送 |
| `agent_status` | `agent`, `status` | Agent 状态变化 |
| `tool_call` | `tool`, `args` | 工具调用开始 |
| `tool_result` | `tool`, `success` | 工具调用结束 |
| `chunk` | `content` | LLM 流式文本 |
| `done` | `content` | 响应完成 |
| `error` | `content` | 错误信息 |

#### 完整事件流示例

```json
→ {"type": "agent_status", "agent": "planner", "status": "executing"}
→ {"type": "chunk", "content": "分析"}
→ {"type": "chunk", "content": "任务"}
→ {"type": "agent_status", "agent": "planner", "status": "completed"}
→ {"type": "agent_status", "agent": "executor", "status": "executing"}
→ {"type": "tool_call", "tool": "read_file", "args": {"path": "test.py"}}
→ {"type": "tool_result", "tool": "read_file", "success": true}
→ {"type": "chunk", "content": "执行结果"}
→ {"type": "agent_status", "agent": "executor", "status": "completed"}
→ {"type": "agent_status", "agent": "critic", "status": "executing"}
→ {"type": "agent_status", "agent": "critic", "status": "completed"}
→ {"type": "done", "content": "完整回复"}
```

---

## 九、RAG 模块详解

### 9.1 RAG 是什么

RAG（Retrieval-Augmented Generation）= 检索增强生成。让 LLM 在回答时参考外部知识库，而不是只靠训练数据。

### 9.2 RAG 流程

```
文档入库：
  文档 → 分块(chunker) → 向量化(embedder) → 存储(vector_store)

查询检索：
  用户问题 → 向量化 → 相似度搜索 → 返回相关文档块 → 注入 LLM 上下文
```

### 9.3 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `TextChunker` | `chunker.py` | 文档分块（默认 512 字符，重叠 64 字符） |
| `Embedder` | `embedder.py` | 文本向量化（默认 bge-m3 模型） |
| `VectorStore` | `vector_store.py` | 向量存储抽象接口 |
| `ChromaStore` | `stores/chroma_store.py` | ChromaDB 实现 |
| `RetrievalPipeline` | `retrieval.py` | 完整检索管线 |

### 9.4 使用方式

```python
from src.rag.chunker import TextChunker
from src.rag.embedder import Embedder
from src.rag.retrieval import RetrievalPipeline

# 分块
chunker = TextChunker(chunk_size=512, chunk_overlap=64)
chunks = chunker.chunk(long_document)

# 向量化 + 存储
embedder = Embedder(model="BAAI/bge-m3")
embeddings = await embedder.embed_batch(chunks)
await vector_store.upsert(chunks, embeddings)

# 检索
pipeline = RetrievalPipeline(vector_store, embedder)
results = await pipeline.retrieve("什么是 RAG?", top_k=5)
```

---

## 十、异常体系详解

### 10.1 异常层级

```
MultiAgentError (基类)
├── AgentError      Agent 运行错误（agent_id）
├── ToolError       工具执行错误（tool_name）
├── SkillError      技能执行错误（skill_name）
├── RAGError        RAG 检索错误
├── MCPError        MCP 协议错误（server_name）
└── ConfigError     配置加载错误
```

### 10.2 基类设计

```python
class MultiAgentError(Exception):
    def __init__(self, message: str, code: str = "", context: dict | None = None):
        super().__init__(message)    # 让 str(e) 能显示错误信息
        self.code = code             # 错误码，如 "AGENT_TIMEOUT"
        self.context = context       # 上下文，如 {"retry": 3}
```

### 10.3 子类透传

子类通过 `**kwargs` 透传父类参数：

```python
class AgentError(MultiAgentError):
    def __init__(self, message: str, agent_id: str = "", **kwargs):
        super().__init__(message, **kwargs)  # code, context 透传给父类
        self.agent_id = agent_id

# 使用
raise AgentError("LLM 调用失败", agent_id="executor-1", code="LLM_TIMEOUT")
```

### 10.4 统一捕获

```python
from src.core.exceptions import MultiAgentError

try:
    await graph.ainvoke(input_state)
except MultiAgentError as e:
    logger.error(f"[{e.code}] {e}", extra=e.context)
except Exception as e:
    logger.error(f"Unknown error: {e}")
```

---

## 十一、日志系统详解

### 11.1 两种格式

```python
# 开发环境：人类可读
INFO multi_agent.main: System started

# 生产环境：JSON 结构化
{"timestamp":"2026-06-04T10:30:00+00:00","level":"INFO","logger":"multi_agent.main","message":"System started"}
```

### 11.2 日志命名空间

```
multi_agent（根）
├── multi_agent.core          核心模块
├── multi_agent.llm           LLM 调用
├── multi_agent.llm.langchain LangChain LLM
├── multi_agent.tools         工具执行
├── multi_agent.tools.langchain 工具转换
├── multi_agent.agents.multi  多 Agent 图
├── multi_agent.agents.react  ReAct Agent
├── multi_agent.api.server    Web 服务
└── multi_agent.api.chat      聊天处理
```

### 11.3 使用方式

```python
from src.core.logging import setup_logging, get_logger

# 初始化（启动时调用一次）
setup_logging(level="INFO", json_format=False)

# 获取日志器
logger = get_logger("agents.multi")
logger.info("Graph compiled")

# 带额外字段
logger.info("Tool executed", extra={
    "tool_name": "read_file",
    "duration_ms": 1523,
    "trace_id": "abc-123",
})
```

---

## 十二、项目结构总览

```
multi_agent_system/
├── config.yaml                    # 系统配置
├── pyproject.toml                 # 依赖管理
├── src/
│   ├── main.py                    # 入口（6 步启动流程）
│   │
│   ├── core/                      # 核心模块
│   │   ├── config.py              # 配置加载（Pydantic + YAML）
│   │   ├── logging.py             # 日志系统（JSON/文本双格式）
│   │   └── exceptions.py          # 异常层级
│   │
│   ├── llm/                       # LLM 模块
│   │   └── langchain_llm.py       # LangChain LLM 工厂
│   │
│   ├── agents/                    # Agent 模块
│   │   ├── langgraph_multi.py     # 多 Agent 协作图
│   │   └── langgraph_react.py     # ReAct 单 Agent
│   │
│   ├── tools/                     # 工具模块
│   │   ├── registry.py            # 工具注册表
│   │   ├── langchain_tools.py     # LangChain 工具转换
│   │   └── builtin/               # 10 个内置工具
│   │       ├── file_ops.py        # 文件操作
│   │       ├── data_ops.py        # 数据处理
│   │       ├── code_exec.py       # 代码执行
│   │       ├── network.py         # 网络请求
│   │       └── rag_ops.py         # RAG 操作
│   │
│   ├── api/                       # Web 服务
│   │   ├── server.py              # FastAPI 应用
│   │   ├── chat.py                # WebSocket 聊天
│   │   ├── models.py              # 数据模型
│   │   └── static/
│   │       └── index.html         # Stripe 风格前端
│   │
│   ├── rag/                       # RAG 模块
│   │   ├── chunker.py             # 文档分块
│   │   ├── embedder.py            # 文本向量化
│   │   ├── vector_store.py        # 向量存储抽象
│   │   ├── retrieval.py           # 检索管线
│   │   └── stores/
│   │       └── chroma_store.py    # ChromaDB 实现
│   │
│   └── mcp/                       # MCP 协议
│       ├── protocol.py            # 消息定义
│       ├── server.py              # 服务器基类
│       └── client.py              # 客户端
│
├── docs/
│   └── api-reference.md           # API 参考文档
│
└── tests/                         # 测试
```

---

## 十三、启动流程详解

`src/main.py` 的 6 步启动流程：

```python
async def main():
    # 步骤 1：加载配置
    # 读取 config.yaml → 替换环境变量 → Pydantic 验证
    settings = load_config(Path("config.yaml"))

    # 步骤 2：初始化日志
    # 配置根日志器 "multi_agent"
    setup_logging(level=settings.system.log_level)

    # 步骤 3：创建 LangChain LLM
    # 根据 provider 创建 ChatTongyi 或 ChatAnthropic
    langchain_llm = create_langchain_llm(settings)

    # 步骤 4：注册工具并转换
    # 注册 10 个内置工具 → 转换为 LangChain StructuredTool
    registry = ToolRegistry()
    register_all_builtin_tools(registry, workspace_dir="./data/workspace")
    lc_tools = convert_to_langchain_tools(registry)

    # 步骤 5：构建 LangGraph 图
    # Planner → Executor → Critic，条件边支持回退修订
    graph = create_multi_agent_graph(langchain_llm, lc_tools)

    # 步骤 6：启动 Web 服务
    # FastAPI + WebSocket，端口 8000
    app = create_app(graph=graph)
    await uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000)).serve()
```

---

## 十四、常见问题

### Q: 如何切换 LLM 提供商？

修改 `config.yaml` 中的 `llm.provider` 和 `llm.model`，并设置对应的环境变量。

### Q: 如何添加自定义工具？

```python
from src.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.register(
    name="my_tool",
    description="我的自定义工具",
    parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
)
async def my_tool(input: str) -> str:
    return f"处理结果: {input}"
```

然后在 `register_all_builtin_tools` 中调用你的注册函数。

### Q: 如何修改 Agent 行为？

编辑 `src/agents/langgraph_multi.py` 中的节点函数的 `SystemMessage` 内容。

### Q: 如何添加新的 Agent 节点？

1. 在 `MultiAgentState` 中添加新字段
2. 定义新的节点函数
3. `graph.add_node("new_agent", new_function)`
4. 添加边连接

### Q: 前端如何自定义？

编辑 `src/api/static/index.html`，它是自包含的 HTML（CSS + JS 内嵌），无需构建工具。
