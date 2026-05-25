"""Agent message bus wrapping Core EventBus."""
import time
import uuid
from typing import Any, Callable, Awaitable
from src.core.events import EventBus, Message, MessageType


class AgentMessageBus:
    def __init__(self):
        self._event_bus = EventBus()

    def register_agent(self, agent_id: str, handler: Callable[[Message], Awaitable[None]]):
        self._event_bus.subscribe(agent_id, handler)

    def unregister_agent(self, agent_id: str):
        self._event_bus.unsubscribe(agent_id)

    async def start(self):
        await self._event_bus.start()

    async def stop(self):
        await self._event_bus.stop()

    async def send(self, sender: str, receiver: str, msg_type: str, payload: dict[str, Any], correlation_id: str | None = None):
        await self._event_bus.publish(Message(
            sender=sender, receiver=receiver, type=MessageType(msg_type),
            payload=payload, timestamp=time.time(), correlation_id=correlation_id or str(uuid.uuid4())))

    async def broadcast(self, sender: str, msg_type: str, payload: dict[str, Any]):
        await self._event_bus.publish(Message(
            sender=sender, receiver="broadcast", type=MessageType(msg_type),
            payload=payload, timestamp=time.time(), correlation_id=str(uuid.uuid4())))
