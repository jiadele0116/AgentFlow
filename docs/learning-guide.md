# Multi-Agent System 学习指南

## 项目简介

这是一个基于 Python 的多 Agent 协作系统，集成四大核心模块：

- **Agent** — 多个 AI Agent 并行/串行协作
- **MCP** — Model Context Protocol，Agent 与外部工具的标准化通信
- **Tools** — 最小可执行单元，Agent 的具体能力
- **RAG** — 检索增强生成，为 Agent 提供外部知识

## 核心逻辑（先看全局）

```
用户任务
  │
  ▼
PlannerAgent（分解任务为子任务）
  │
  ▼
CoordinatorAgent（分配子任务给合适的 Agent）
  │
  ▼
ExecutorAgent（执行任务，调用 Tools / Skills / RAG）
  │
  ▼
CriticAgent（审查结果质量）
  │
  ├─ 通过 → 返回最终结果给用户
  │
  └─ 不通过 → ExecutorAgent 修订 → 再次审查
```

Agent 之间通过 MessageBus 异步通信，每个 Agent 是一个状态机，通过 LLM 推理决策，通过 Tools 执行操作，通过 RAG 获取知识。

## 学习路线：从底向上，逐层理解

---

### 第一层：基础设施

#### 1. `src/core/config.py` — 一切的起点

系统配置中心。用 Pydantic 定义所有子系统的配置模型（LLM、Embedding、RAG、Agent、MCP），从 `config.yaml` 加载，支持 `${ENV_VAR}` 环境变量替换。

**重点理解**：
- `Settings` 是顶层配置容器，聚合了所有子模块配置
- `load_config()` 流程：读 YAML → 展开环境变量 → Pydantic 验证 → 存储为全局单例
- `get_settings()` 全局访问模式

#### 2. `src/core/events.py` — 消息流转的底层

EventBus 是所有组件通信的基础设施，基于 `asyncio.Queue` 实现发布/订阅模式。

**重点理解**：
- `Message` 数据结构：sender、receiver、type、payload、correlation_id
- 发布/订阅机制：Agent 订阅自己的 topic，EventBus 负责分发
- 这是 Agent 间协作的地基

#### 3. `src/core/logging.py` / `src/core/exceptions.py`

结构化日志系统和异常层级定义。快速过一遍即可。

---

### 第二层：LLM 抽象层（理解"大脑"怎么接入）

#### 4. `src/llm/provider.py` — LLM 统一接口

定义 `LLMProvider` 抽象基类，只有两个核心方法：
- `complete()` — 发送消息，获取完整响应
- `stream()` — 发送消息，获取流式响应

**重点理解**：
- `LLMResponse`：包含文本内容 `content`、工具调用 `tool_calls`、token 用量 `usage`
- `ToolCall`：LLM 决定调用哪个工具时返回的数据结构（id、name、arguments）
- 所有 LLM 提供商都实现这个接口，上层 Agent 代码与具体提供商解耦

#### 5. `src/llm/anthropic.py` 或 `src/llm/qwen.py` — 具体实现

挑一个看即可，理解如何把 API 调用封装到统一接口中。

---

### 第三层：工具系统（理解 Agent 能"做什么"）

#### 6. `src/tools/registry.py` — 工具注册表

装饰器风格的工具注册机制。每个工具包含：
- `name` — 唯一标识符
- `description` — 功能描述（LLM 根据此决定何时调用）
- `parameters` — JSON Schema 格式的参数定义
- `handler` — 异步执行函数

**重点理解**：
- `list_tools()` 生成的格式直接传给 LLM 的 tools 参数，让 LLM 知道有哪些工具可用
- 注册表是全局的，系统启动时注册一次

#### 7. `src/tools/engine.py` — 工具执行引擎

在注册表之上封装执行逻辑：
- **超时控制**：`asyncio.wait_for` 包装，超时直接返回失败
- **重试机制**：指数退避重试（`0.1 * 2^attempt` 秒）
- **执行计时**：记录每次调用的耗时

**重点理解**：
- `ToolRegistry` 管理工具注册，`ToolEngine` 管理工具执行，职责分离
- `execute()` 是 Agent 调用工具的入口

#### 8. `src/tools/builtin/` — 内置工具

| 文件 | 功能 | 包含的工具 |
|------|------|-----------|
| `file_ops.py` | 文件操作 | read_file, write_file, search_files |
| `code_exec.py` | 代码执行 | run_python, run_shell, lint_code |
| `network.py` | 网络请求 | fetch_url, search_web |
| `data_ops.py` | 数据处理 | parse_json, csv_to_dict, json_to_csv |
| `rag_ops.py` | RAG 操作 | index_document, search_knowledge |

挑一两个看实现即可，理解工具的输入输出格式。

---

### 第四层：Agent 体系（理解"谁在干活"）

#### 9. `src/agents/base.py` — Agent 基类（最核心）

这是整个系统的**核心抽象**，必须深入理解。

**AgentState 状态机**：
```
INIT → IDLE → PLANNING → EXECUTING → REVIEWING → REVISING → COMPLETED
                    ↑                                  │
                    └──────────────────────────────────┘（修订循环）
```

**数据结构**：
- `Task` — 任务定义（id、description、input_data、assigned_to、parent_id）
- `Result` — 执行结果（task_id、success、data、error、duration_ms）
- `PlannerResult` — 规划结果（success、subtasks、content）

**BaseAgent 抽象类**：
- 属性：id、role、goal、capabilities、memory、state
- 方法：`transition()` 状态转换、`run()` 抽象执行方法（子类必须实现）

#### 10. `src/agents/memory.py` — 记忆系统

Agent 的记忆分两层：
- **ShortTermMemory**（短期）：基于 `deque`，自动淘汰旧消息，会话级
- **LongTermMemory**（长期）：基于 SQLite 持久化，支持跨会话

#### 11. `src/agents/message_bus.py` — Agent 间通信

封装 EventBus，提供 Agent 友好的接口：
- `send()` — 点对点消息
- `broadcast()` — 广播消息

#### 12. 五种 Agent 实现

按职责顺序阅读：

| 顺序 | 文件 | 角色 | 做什么 |
|------|------|------|--------|
| 1 | `planner.py` | 项目经理 | 接收任务，调用 LLM 分解为子任务列表 |
| 2 | `executor.py` | 开发者 | 执行具体子任务，调用 Tools/Skills |
| 3 | `critic.py` | 审查员 | 审查 Executor 的输出，给出反馈 |
| 4 | `retriever.py` | 研究员 | 执行 RAG 检索，提供知识上下文 |
| 5 | `coordinator.py` | 调度员 | 协调多 Agent 间的任务分发和冲突解决 |

**阅读建议**：先读 `planner.py`，它最简单，只有 60 多行，完整展示了 Agent 的工作模式：接收任务 → 状态转换 → 调用 LLM → 解析结果 → 返回。

---

### 第五层：高级能力

#### 13. `src/agents/strategy.py` — 执行策略

三种策略模式，决定了 Agent "怎么思考"：
- **ReAct** — 推理（Reasoning）+ 行动（Action）交替执行
- **Plan-and-Execute** — 先规划再执行
- **Chain-of-Thought** — 链式思维推理

#### 14. `src/skills/` — 技能编排

Skill 是 Tool 的高阶组合，由多个 Tool 调用 + 控制逻辑 + LLM 推理组成。
- `engine.py` — Skill 执行引擎，支持 tool_call、llm_reasoning、conditional、loop 四种步骤类型
- `loader.py` — 从 YAML 文件加载 Skill 定义

#### 15. `src/rag/` — RAG 检索增强

完整管线：
```
文档 → 分块(chunker) → 向量化(embedder) → 存储(vector_store)
                                              ↓
用户查询 → 查询改写 → 向量检索 → 重排序 → 上下文注入 → LLM 生成
```

| 文件 | 职责 |
|------|------|
| `chunker.py` | 文档分块策略（fixed_size / sentence / semantic / code / recursive） |
| `embedder.py` | 文本向量化封装 |
| `vector_store.py` | 向量存储抽象接口 |
| `stores/chroma_store.py` | ChromaDB 实现 |
| `retrieval.py` | 检索管线（混合检索 + RRF 融合 + 重排序） |
| `query_rewrite.py` | 查询改写 |

#### 16. `src/mcp/` — MCP 协议层

Agent 与外部工具/数据源的标准化通信协议：
- `protocol.py` — MCPRequest / MCPResponse 消息定义
- `server.py` — MCP Server 基类
- `client.py` — MCP Client
- `servers/` — 内置服务器（filesystem、web）

---

### 最后：入口点

#### `src/main.py` — 串联一切

启动流程：
1. 加载配置（`config.yaml`）
2. 初始化日志
3. 注册内置工具
4. 启动事件总线
5. 保持运行，等待退出

---

## 建议的学习顺序总结

```
第1步：config.py（理解系统参数）
  │
第2步：events.py（理解通信基础）
  │
第3步：llm/provider.py（理解 LLM 接口）
  │
第4步：tools/registry.py → tools/engine.py（理解工具系统）
  │
第5步：agents/base.py（理解 Agent 核心抽象）
  │
第6步：agents/planner.py（最简单的 Agent 实现）
  │
第7步：agents/executor.py → agents/critic.py（理解执行和审查）
  │
第8步：agents/coordinator.py（理解多 Agent 协调）
  │
第9步：agents/strategy.py（理解执行策略）
  │
第10步：rag/ 模块（理解知识检索）
  │
第11步：skills/ 模块（理解高级编排）
  │
第12步：mcp/ 模块（理解外部协议）
  │
最后：main.py（看全局如何串联）
```

## 三个核心抽象（抓住骨架）

如果时间有限，只需理解这三个文件就抓住了系统骨架：

1. **`src/agents/base.py`** — Agent 是什么（状态机 + 任务 + 结果）
2. **`src/tools/registry.py`** — Tool 是什么（name + description + JSON Schema + handler）
3. **`src/llm/provider.py`** — LLM 怎么接（complete/stream → LLMResponse）

一切扩展（新 Agent、新 Tool、新 LLM 提供商）都是在这三个抽象之上实现的。
