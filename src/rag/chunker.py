"""
文档分块模块 (Text Chunking Strategies)

将长文档切分为适合嵌入和检索的小块（Chunk）。分块质量直接影响
检索精度：块太大则语义模糊，块太小则丢失上下文。

分块策略：
1. FixedSizeChunker（固定大小分块）：
   - 按固定字符数切分，相邻块有重叠区域
   - 优点：简单高效，块大小一致
   - 缺点：可能在句子中间切断

2. RecursiveChunker（递归分块）：
   - 先按 Markdown 标题（#）切分大段落
   - 超长段落再用 FixedSizeChunker 二次切分
   - 优点：保留文档结构，标题段落完整
   - 缺点：块大小不均匀

分块参数权衡：
  - chunk_size: 512 字符（默认）→ 平衡精度和上下文
  - overlap: 64 字符（默认）→ 约 12.5% 重叠，确保跨块信息连续
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """文档块数据类：表示分块后的一个文本单元。

    每个 Chunk 包含文本内容和丰富的元信息，用于后续的嵌入、存储和检索。

    Attributes:
        text: 块的文本内容
        metadata: 元数据字典（来源、标题、作者等）
        index: 块在原文档中的序号（从 0 开始）
        source: 块的来源标识（文件路径或文档ID）
    """
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    source: str = ""


class Chunker(ABC):
    """分块器抽象基类。

    所有分块策略必须实现 chunk() 方法，接收原始文本返回 Chunk 列表。
    """

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """将文本切分为块列表。

        Args:
            text: 原始文本
            metadata: 可选的元数据，会复制到每个 Chunk

        Returns:
            Chunk 列表
        """
        ...


class FixedSizeChunker(Chunker):
    """固定大小分块器：按固定字符数切分文本。

    切分逻辑：
    1. 从位置 0 开始，取 chunk_size 个字符作为一个块
    2. 下一个块的起始位置 = 当前起始 + (chunk_size - overlap)
    3. 重叠区域确保相邻块有连续的上下文
    4. 跳过空白块

    示例（chunk_size=10, overlap=3）：
      text = "0123456789ABCDEFGHIJ"
      块1: "0123456789" (位置 0-9)
      块2: "789ABCDEF"  (位置 7-15, 重叠 "789")
      块3: "EFGHIJ"     (位置 14-19, 重叠 "EF")

    Args:
        chunk_size: 每个块的最大字符数
        overlap: 相邻块的重叠字符数
    """

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
            # 步进 = chunk_size - overlap，实现重叠
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveChunker(Chunker):
    """递归分块器：先按文档结构切分，再对超长段落二次切分。

    切分策略：
    1. 按 Markdown 标题（# 开头的行）切分为段落
    2. 短段落（≤ chunk_size）直接作为块
    3. 长段落使用 FixedSizeChunker 二次切分

    适用于 Markdown、文档等有明确结构的文本。

    Args:
        chunk_size: 每个块的最大字符数
        overlap: 相邻块的重叠字符数
    """

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
                # 短段落直接作为块
                if section.strip():
                    chunks.append(Chunk(text=section, metadata=meta.copy(), index=len(chunks), source=meta.get("source", "")))
            else:
                # 长段落使用固定大小分块器二次切分
                sub = FixedSizeChunker(self.chunk_size, self.overlap).chunk(section, meta)
                for sc in sub:
                    sc.index = len(chunks)
                    chunks.append(sc)
        return chunks

    def _split_by_headers(self, text: str) -> list[str]:
        """按 Markdown 标题切分文本。

        遇到 # 开头的行时，将之前的内容作为一个段落。
        """
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
