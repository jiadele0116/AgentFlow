"""Tests for RAG chunker."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.rag.chunker import FixedSizeChunker, RecursiveChunker, SemanticChunker, Chunk


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


# --- SemanticChunker Tests ---


def _make_mock_embedder(similarities: list[float] | None = None):
    """创建 mock embedder，返回预设的相似度对应的向量。

    Args:
        similarities: 预设的相邻句子相似度列表。
                      如果为 None，返回随机向量（真实计算相似度）。
    """
    import random
    mock = AsyncMock()
    if similarities is not None:
        # 构造向量使得相邻句子的余弦相似度等于预设值
        # 简单方案：用 2 维向量，通过角度控制相似度
        import math
        embeddings = []
        angle = 0.0
        embeddings.append([math.cos(angle), math.sin(angle)])
        for sim in similarities:
            # sim = cos(delta_angle)，反算 delta_angle
            delta = math.acos(max(-1, min(1, sim)))
            angle += delta
            embeddings.append([math.cos(angle), math.sin(angle)])
        mock.embed_batch = AsyncMock(return_value=embeddings)
    else:
        # 返回随机向量
        def random_embeddings(texts):
            return [[random.random() for _ in range(10)] for _ in texts]
        mock.embed_batch = AsyncMock(side_effect=random_embeddings)
    return mock


@pytest.mark.asyncio
async def test_semantic_chunker_basic():
    """语义分块器基本功能：多句子文本被切分为至少一个块。"""
    mock_embedder = _make_mock_embedder([0.9, 0.9, 0.2, 0.9])
    chunker = SemanticChunker(embedder=mock_embedder, max_chunk_size=500)
    text = "句子一。句子二。句子三。句子四。句子五。"
    chunks = await chunker.chunk_async(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    # 所有文本都被覆盖
    full_text = "\n".join(c.text for c in chunks)
    assert "句子一" in full_text
    assert "句子五" in full_text


@pytest.mark.asyncio
async def test_semantic_chunker_breakpoint_detection():
    """语义分块器在相似度骤降处切分。"""
    # 前 3 个句子相似度高（0.9），第 4 个骤降（0.1），后面又高（0.9）
    mock_embedder = _make_mock_embedder([0.9, 0.9, 0.1, 0.9])
    chunker = SemanticChunker(
        embedder=mock_embedder,
        max_chunk_size=1000,
        min_chunk_size=1,
        breakpoint_threshold_amount=0.5,
    )
    text = "A。B。C。D。E。"
    chunks = await chunker.chunk_async(text)

    # 应该在相似度 0.1 处切分，产生多个块
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_semantic_chunker_empty_text():
    """空文本返回空列表。"""
    mock_embedder = _make_mock_embedder()
    chunker = SemanticChunker(embedder=mock_embedder)
    chunks = await chunker.chunk_async("")
    assert chunks == []


@pytest.mark.asyncio
async def test_semantic_chunker_single_sentence():
    """单句文本返回单个块。"""
    mock_embedder = _make_mock_embedder()
    chunker = SemanticChunker(embedder=mock_embedder)
    chunks = await chunker.chunk_async("只有一句话。")
    assert len(chunks) == 1
    assert chunks[0].text == "只有一句话。"


@pytest.mark.asyncio
async def test_semantic_chunker_preserves_metadata():
    """语义分块器保留元数据。"""
    mock_embedder = _make_mock_embedder([0.5])
    chunker = SemanticChunker(embedder=mock_embedder, min_chunk_size=1)
    chunks = await chunker.chunk_async("句子一。句子二。", metadata={"source": "test.txt"})
    for c in chunks:
        assert c.metadata["source"] == "test.txt"


@pytest.mark.asyncio
async def test_semantic_chunker_max_size():
    """超长块会被拆分。"""
    mock_embedder = _make_mock_embedder([0.9] * 19)
    chunker = SemanticChunker(
        embedder=mock_embedder,
        max_chunk_size=20,
        min_chunk_size=1,
    )
    # 20 个短句，每句约 4 字符，总 ~80 字符，max=20 应切多块
    text = "。".join([f"句子{i}" for i in range(20)]) + "。"
    chunks = await chunker.chunk_async(text)
    assert len(chunks) >= 2


def test_semantic_chunker_no_embedder_sync():
    """无 embedder 时，同步调用使用降级方案（按句子分组）。"""
    chunker = SemanticChunker(max_chunk_size=100)
    text = "句子一。句子二。句子三。"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_semantic_chunker_cosine_similarity():
    """余弦相似度计算正确。"""
    assert SemanticChunker._cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert SemanticChunker._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert SemanticChunker._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert SemanticChunker._cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


def test_recursive_chunker_with_embedder():
    """RecursiveChunker 传入 embedder 时，长段落使用语义切分。"""
    mock_embedder = _make_mock_embedder([0.9, 0.2, 0.9])
    chunker = RecursiveChunker(chunk_size=20, embedder=mock_embedder)
    text = "# 标题\n这是一个很长的段落内容。包含多个句子。还有更多句子。最后句子。"
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1


def test_recursive_chunker_without_embedder():
    """RecursiveChunker 无 embedder 时，长段落使用固定大小切分（原有行为）。"""
    chunker = RecursiveChunker(chunk_size=20, overlap=5)
    text = "# 标题\n" + "很长的内容" * 20
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
