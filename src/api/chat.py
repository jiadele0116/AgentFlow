"""聊天 WebSocket 处理模块。

管理聊天会话，通过 LangGraph Agent 流式返回响应和 Agent 状态事件。
集成记忆系统：短期记忆保持上下文，长期记忆跨会话召回。
"""
import asyncio
import uuid
from typing import Any

from fastapi import WebSocket
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.core.logging import get_logger

logger = get_logger("api.chat")

# Agent 角色定义
AGENTS = [
    {"id": "planner", "name": "Planner", "icon": "📋", "role": "任务规划"},
    {"id": "executor", "name": "Executor", "icon": "⚡", "role": "任务执行"},
    {"id": "critic", "name": "Critic", "icon": "🔍", "role": "质量审查"},
    {"id": "retriever", "name": "Retriever", "icon": "📚", "role": "知识检索"},
]


async def send_event(ws: WebSocket, event_type: str, **data):
    """发送事件到前端。"""
    try:
        await ws.send_json({"type": event_type, **data})
    except Exception:
        pass


class ChatSession:
    """单个聊天会话，维护对话历史并通过 LangGraph 执行。

    Args:
        session_id: 会话唯一 ID
        graph: LangGraph 编译好的图
        memory: 记忆管理器实例
    """

    def __init__(self, session_id: str, graph, memory=None):
        self.session_id = session_id
        self._graph = graph
        self._memory = memory
        self._history: list[dict[str, Any]] = []

    async def handle_message(self, websocket: WebSocket, user_message: str, files: list[dict] | None = None):
        """处理用户消息，通过 LangGraph Agent 流式返回结果。

        Args:
            websocket: WebSocket 连接
            user_message: 用户消息文本
            files: 上传的文件信息列表 [{filename, path, size, type}, ...]
        """
        # 构建完整消息（包含文件信息）
        full_message = user_message
        if files:
            file_info = "\n\n[上传的文件]\n"
            for f in files:
                file_info += f"- {f['filename']} ({f['type']}, {f['size']} bytes): {f['path']}\n"
            full_message += file_info

        # 记录用户消息到短期记忆
        if self._memory:
            await self._memory.on_user_message(self.session_id, full_message)

        self._history.append({"role": "user", "content": full_message, "files": files or []})

        try:
            # 获取记忆上下文（短期历史 + 长期召回）
            memory_context = ""
            if self._memory:
                memory_context = await self._memory.get_context(
                    self.session_id, user_message
                )

            # 构造消息列表（只用最近 4 条历史，减少 token 消耗）
            messages = []
            for msg in self._history[-4:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(SystemMessage(content=f"[助手回复] {msg['content']}"))
            # 加入当前用户消息（如果历史里已经有了就跳过）
            if not self._history or self._history[-1]["content"] != user_message:
                messages.append(HumanMessage(content=user_message))

            # 构造输入状态
            input_state = {
                "task": user_message,
                "messages": messages,
                "review_status": "pending",
                "iterations": 0,
                "context": memory_context,
            }

            # 流式调用 LangGraph
            # 只输出 Executor 的结果给用户，Planner/Critic 的过程不输出
            full_response = ""
            current_agent = None

            async for event in self._graph.astream_events(input_state, version="v2"):
                kind = event.get("event", "")
                node_name = event.get("name", "")

                # 节点开始（更新侧边栏 Agent 状态）
                if kind == "on_chain_start" and node_name in ("planner", "executor", "critic"):
                    if current_agent and current_agent != node_name:
                        await send_event(websocket, "agent_status",
                                         agent=current_agent, status="completed")
                    current_agent = node_name
                    await send_event(websocket, "agent_status",
                                     agent=node_name, status="executing")

                # LLM 流式输出 — 只输出 Executor 的内容给用户
                elif kind == "on_chat_model_stream" and current_agent == "executor":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        text = chunk.content
                        full_response += text
                        await send_event(websocket, "chunk", content=text)

                # 工具调用开始（只展示 Executor 的工具调用）
                elif kind == "on_tool_start" and current_agent == "executor":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    if isinstance(tool_input, dict):
                        serializable_input = {k: str(v) for k, v in tool_input.items()}
                    else:
                        serializable_input = {"value": str(tool_input)}
                    await send_event(websocket, "tool_call",
                                     tool=tool_name, args=serializable_input)

                # 工具调用结束
                elif kind == "on_tool_end" and current_agent == "executor":
                    tool_name = event.get("name", "unknown")
                    await send_event(websocket, "tool_result",
                                     tool=tool_name, success=True)

            # 标记最后的 Agent 完成
            if current_agent:
                await send_event(websocket, "agent_status",
                                 agent=current_agent, status="completed")

            # 如果流式没有获取到内容，从最终状态提取 Executor 的结果
            if not full_response:
                try:
                    final_state = await self._graph.ainvoke(input_state)
                    # 优先取 execution_result（Executor 的输出）
                    full_response = final_state.get("execution_result", "")
                    # 兜底：从消息里找最后一条 AI 消息
                    if not full_response:
                        messages = final_state.get("messages", [])
                        for msg in reversed(messages):
                            if isinstance(msg, AIMessage) and msg.content:
                                full_response = msg.content
                                break
                    if not full_response:
                        full_response = "处理完成"
                    await send_event(websocket, "chunk", content=full_response)
                except Exception as e:
                    logger.error(f"备用方案执行失败: {e}")
                    full_response = "抱歉，处理过程中出现错误。"
                    await send_event(websocket, "chunk", content=full_response)

            # 记录助手回复到记忆系统
            if self._memory:
                await self._memory.on_assistant_message(self.session_id, full_response)

            # 发送完成信号
            self._history.append({"role": "assistant", "content": full_response})
            await send_event(websocket, "done", content=full_response)

        except Exception as e:
            logger.error(f"聊天处理出错: {e}", exc_info=True)
            await send_event(websocket, "error", content=str(e))


class ChatManager:
    """管理多个聊天会话。"""

    def __init__(self, graph, memory=None):
        self._graph = graph
        self._memory = memory
        self._sessions: dict[str, ChatSession] = {}

    def create_session(self) -> ChatSession:
        """创建新的聊天会话。"""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, self._graph, self._memory)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        """获取已有会话。"""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str):
        """移除会话。"""
        self._sessions.pop(session_id, None)


async def handle_websocket_chat(websocket: WebSocket, chat_manager: ChatManager):
    """WebSocket 聊天端点处理函数。"""
    await websocket.accept()
    session = chat_manager.create_session()
    logger.info(f"聊天会话已创建: {session.session_id}")

    # 发送系统信息
    await send_event(websocket, "system_info",
                     agents=AGENTS,
                     tools=[],
                     session_id=session.session_id)

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("content", "")
            files = data.get("files", [])
            if not message and not files:
                continue
            logger.info(f"[{session.session_id}] 用户消息: {message[:50]}... (文件: {len(files)} 个)")
            await session.handle_message(websocket, message, files=files)
    except Exception as e:
        if "disconnect" not in str(e).lower() and "1000" not in str(e):
            logger.info(f"会话 {session.session_id} 已断开连接: {e}")
    finally:
        chat_manager.remove_session(session.session_id)
        logger.info(f"会话 {session.session_id} 已清理")
