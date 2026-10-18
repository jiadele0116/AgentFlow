"""记忆管理器模块 (Memory Manager)

统一管理短期记忆和长期记忆，提供简洁的接口给上层使用。
长期记忆通过 LLM 总结整段对话后存入向量数据库。
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.core.logging import get_logger

logger = get_logger("memory.manager")


class MemoryManager:
    """记忆管理器：统一管理短期和长期记忆。

    Args:
        short_term: 短期记忆实例
        long_term: 长期记忆实例（可选）
        llm: LLM 实例，用于生成对话摘要（可选）
    """

    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory | None = None,
                 llm: BaseChatModel | None = None):
        self.short_term = short_term
        self.long_term = long_term
        self._llm = llm

    async def _summarize_conversation(self, session_id: str) -> str | None:
        """用 LLM 总结整段对话，提取关键信息。

        Args:
            session_id: 会话 ID

        Returns:
            对话摘要文本，失败返回 None
        """
        if not self._llm:
            return None

        # 获取最近的对话历史
        history = self.short_term.get_history(session_id, last_n=20)
        if not history or len(history) < 2:
            return None

        # 拼成对话文本
        conversation = ""
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            conversation += f"{role}: {msg['content']}\n"

        try:
            from src.core.prompts import MEMORY_SUMMARIZE_PROMPT
            response = await self._llm.ainvoke([
                SystemMessage(content=MEMORY_SUMMARIZE_PROMPT),
                HumanMessage(content=conversation),
            ])
            summary = response.content.strip()
            logger.info(f"对话摘要: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.warning(f"对话摘要失败: {e}")
            return None

    async def on_user_message(self, session_id: str, content: str):
        """用户发消息时调用：记录到短期记忆。"""
        self.short_term.add(session_id, "user", content)

    async def on_assistant_message(self, session_id: str, content: str):
        """助手回复时调用：记录到短期记忆 + 总结对话存入长期记忆。"""
        self.short_term.add(session_id, "assistant", content)

        # 每 4 轮对话（8 条消息）总结一次存入长期记忆
        if self.long_term:
            history = self.short_term.get_history(session_id)
            if len(history) >= 8 and len(history) % 8 == 0:
                try:
                    summary = await self._summarize_conversation(session_id)
                    if summary:
                        await self.long_term.save(session_id, summary, memory_type="summary")
                        logger.info(f"长期记忆已保存: {session_id}")
                except Exception as e:
                    logger.warning(f"保存长期记忆失败: {e}")

    async def get_context(self, session_id: str, current_query: str) -> str:
        """获取完整的对话上下文（短期历史 + 长期记忆召回）。

        Args:
            session_id: 会话 ID
            current_query: 当前用户查询（用于语义召回）

        Returns:
            格式化的上下文字符串
        """
        parts = []

        # 短期记忆：最近的对话历史（精简为 4 条）
        history = self.short_term.get_context_prompt(session_id, last_n=4)
        if history:
            parts.append(f"最近的对话历史：\n{history}")

        # 长期记忆：语义召回相关记忆
        if self.long_term:
            recalled = await self.long_term.get_context_prompt(current_query, session_id=session_id)
            if recalled:
                parts.append(recalled)

        return "\n\n".join(parts) if parts else ""
