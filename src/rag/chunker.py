"""Text chunking strategies for RAG."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    source: str = ""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]: ...


class FixedSizeChunker(Chunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not text.strip():
            return []
        meta = metadata or {}
        chunks = []
        start, index = 0, 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(text=chunk_text, metadata=meta.copy(), index=index, source=meta.get("source", "")))
                index += 1
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not text.strip():
            return []
        meta = metadata or {}
        sections = self._split_by_headers(text)
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                if section.strip():
                    chunks.append(Chunk(text=section, metadata=meta.copy(), index=len(chunks), source=meta.get("source", "")))
            else:
                sub = FixedSizeChunker(self.chunk_size, self.overlap).chunk(section, meta)
                for sc in sub:
                    sc.index = len(chunks)
                    chunks.append(sc)
        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        lines = text.split("\n")
        sections, current = [], []
        for line in lines:
            if line.startswith("#") and current:
                sections.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current))
        return sections
