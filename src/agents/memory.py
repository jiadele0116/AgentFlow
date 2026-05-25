"""Agent memory: short-term (deque) and long-term (SQLite)."""
import aiosqlite
from collections import deque
from typing import Any


class ShortTermMemory:
    def __init__(self, max_turns: int = 20):
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_turns)

    def add(self, message: dict[str, Any]):
        self._messages.append(message)

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    @property
    def count(self) -> int:
        return len(self._messages)


class LongTermMemory:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self):
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id)")
            await self._db.commit()

    async def store(self, agent_id: str, key: str, value: str):
        await self._ensure_db()
        await self._db.execute("INSERT INTO memories (agent_id, key, value) VALUES (?, ?, ?)", (agent_id, key, value))
        await self._db.commit()

    async def retrieve(self, agent_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT key, value, created_at FROM memories WHERE agent_id = ? AND value LIKE ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, f"%{query}%", limit))
        rows = await cursor.fetchall()
        return [{"key": r[0], "value": r[1], "created_at": r[2]} for r in rows]

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None
