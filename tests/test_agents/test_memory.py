"""Tests for agent memory."""
import pytest
from src.agents.memory import ShortTermMemory


def test_short_term_add_and_get():
    mem = ShortTermMemory(max_turns=5)
    mem.add({"role": "user", "content": "hello"})
    mem.add({"role": "assistant", "content": "hi"})
    assert len(mem.get_all()) == 2


def test_short_term_eviction():
    mem = ShortTermMemory(max_turns=3)
    for i in range(5):
        mem.add({"role": "user", "content": f"msg {i}"})
    messages = mem.get_all()
    assert len(messages) == 3
    assert messages[0]["content"] == "msg 2"


def test_short_term_clear():
    mem = ShortTermMemory(max_turns=5)
    mem.add({"role": "user", "content": "hello"})
    mem.clear()
    assert len(mem.get_all()) == 0


@pytest.mark.asyncio
async def test_long_term_store_and_retrieve(tmp_path):
    from src.agents.memory import LongTermMemory
    mem = LongTermMemory(db_path=str(tmp_path / "test.db"))
    await mem.store("agent-1", "fact1", "Paris is the capital of France")
    results = await mem.retrieve("agent-1", "capital")
    assert len(results) >= 1
    assert any("Paris" in r["value"] for r in results)
    await mem.close()
