"""Tests for the event bus."""
import asyncio
import pytest


@pytest.mark.asyncio
async def test_event_bus_subscribe_and_publish():
    from src.core.events import EventBus, Message, MessageType
    import time
    bus = EventBus()
    received = []
    async def handler(msg): received.append(msg)
    bus.subscribe("agent-2", handler)
    await bus.start()
    await bus.publish(Message(sender="agent-1", receiver="agent-2", type=MessageType.TASK, payload={"task": "test"}, timestamp=time.time(), correlation_id="c1"))
    await asyncio.sleep(0.1)
    await bus.stop()
    assert len(received) == 1
    assert received[0].payload == {"task": "test"}


@pytest.mark.asyncio
async def test_event_bus_broadcast():
    from src.core.events import EventBus, Message, MessageType
    import time
    bus = EventBus()
    received_a, received_b = [], []
    async def ha(msg): received_a.append(msg)
    async def hb(msg): received_b.append(msg)
    bus.subscribe("a", ha)
    bus.subscribe("b", hb)
    await bus.start()
    await bus.publish(Message(sender="coordinator", receiver="broadcast", type=MessageType.TASK, payload={"b": True}, timestamp=time.time(), correlation_id="c2"))
    await asyncio.sleep(0.1)
    await bus.stop()
    assert len(received_a) == 1
    assert len(received_b) == 1
