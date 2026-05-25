"""Tests for RAG chunker."""
import pytest
from src.rag.chunker import FixedSizeChunker, RecursiveChunker, Chunk


def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk("a" * 25)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 10


def test_recursive_chunker():
    text = "# Section 1\nContent here.\n# Section 2\nMore content."
    chunker = RecursiveChunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1


def test_empty_text():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)
    assert chunker.chunk("") == []


def test_preserves_metadata():
    chunker = FixedSizeChunker(chunk_size=10, overlap=0)
    chunks = chunker.chunk("hello world test", metadata={"source": "test.txt"})
    for c in chunks:
        assert c.metadata["source"] == "test.txt"
