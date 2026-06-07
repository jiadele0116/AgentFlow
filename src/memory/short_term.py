"""短期记忆模块 (Short-Term Memory)

基于字典的会话级记忆，保存当前会话的对话历史。
每个 session_id 对应一个消息列表，支持最大轮数限制和自动淘汰。

存储格式：
  {
    "session-abc": [
      {"role": "user", "content": "你好", "timestamp": 1717500000.0},
      {"role": "assistant", "content": "你好！有什么可以帮你的？", "timestamp": 1717500001.0},
      ...
    ]
  }
"""
import time
from typing import Any

from src.core.logging import get_logger

logger = get_logger("memory.short_term")


class ShortTermMemory:
    """短期记忆：基于字典的会话级对话历史。

    Args:
        max_turns: 每个会话最大保留的消息轮数（超出后自动淘汰最早的）
    """

    def __init__(self, max_turns: int = 20):
        self._max_turns = max_turns
        self._store: dict[str, list[dict[str, Any]]] = {}

    def add(self, session_id: str, role: str, content: str):
        """添加一条消息到指定会话的历史中。

        Args:
            session_id: 会话 ID
            role: 角色 ("user" / "assistant" / "system")
            content: 消息内容
        """
        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

        # 超出最大轮数，淘汰最早的消息
        if len(self._store[session_id]) > self._max_turns * 2:
            self._store[session_id] = self._store[session_id][-(self._max_turns * 2):]
            logger.debug(f"Session {session_id}: trimmed to {self._max_turns * 2} messages")

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict[str, Any]]:
        """获取指定会话的对话历史。

        Args:
            session_id: 会话 ID
            last_n: 只返回最近 N 条消息（None 返回全部）

        Returns:
            消息列表 [{"role": ..., "content": ..., "timestamp": ...}, ...]
        """
        messages = self._store.get(session_id, [])
        if last_n is not None:
            return messages[-last_n:]
        return messages

    def get_context_prompt(self, session_id: str, last_n: int = 10) -> str:
        """将最近的对话历史拼成 LLM 可用的上下文字符串。

        Args:
            session_id: 会话 ID
            last_n: 取最近 N 条消息

        Returns:
            格式化的对话历史字符串
        """
        messages = self.get_history(session_id, last_n=last_n)
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        """清空指定会话的历史。"""
        self._store.pop(session_id, None)

    def get_all_sessions(self) -> list[str]:
        """获取所有会话 ID。"""
        return list(self._store.keys())
