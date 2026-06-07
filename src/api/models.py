"""聊天消息的 Pydantic 模型。"""
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """用户发送的聊天消息。"""
    content: str


class ChatResponse(BaseModel):
    """Agent 返回的响应。"""
    content: str
    role: str = "assistant"
