"""Async event bus for agent communication."""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    sender: str
    receiver: str
    type: MessageType
    payload: dict[str, Any]
    timestamp: float
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


MessageHandler = Callable[[Message], Awaitable[None]]


class EventBus:
    """Async pub/sub event bus built on asyncio.Queue."""

    def __init__(self):
        self._subscribers: dict[str, list[MessageHandler]] = {}
        self._queue: asyncio.Queue[Message | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False

    def subscribe(self, agent_id: str, handler: MessageHandler):
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(handler)

    def unsubscribe(self, agent_id: str):
        self._subscribers.pop(agent_id, None)

    async def publish(self, message: Message):
        await self._queue.put(message)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        self._running = False
        await self._queue.put(None)
        if self._task:
            await self._task

    async def _dispatch_loop(self):
        while self._running:
            message = await self._queue.get()
            if message is None:
                break
            if message.receiver == "broadcast":
                for agent_id, handlers in self._subscribers.items():
                    if agent_id != message.sender:
                        for handler in handlers:
                            asyncio.create_task(self._safe_deliver(handler, message))
            else:
                for handler in self._subscribers.get(message.receiver, []):
                    asyncio.create_task(self._safe_deliver(handler, message))

    async def _safe_deliver(self, handler: MessageHandler, message: Message):
        try:
            await handler(message)
        except Exception:
            pass
