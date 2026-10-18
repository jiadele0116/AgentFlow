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
   - 超长段落再用 SemanticChunker 二次切分
   - 优点：保留文档结构，标题段落完整
   - 缺点：块大小不均匀

3. SemanticChunker（语义分块）：
   - 先将文本切为句子，计算相邻句子的 embedding 余弦相似度
   - 在相似度骤降处（低于阈值）切分，语义断点即分块边界
   - 优点：语义最连贯，检索精度最高
   - 缺点：需要 embedding 模型，计算成本较高

分块参数权衡：
  - chunk_size: 512 字符（默认）→ 平衡精度和上下文
  - overlap: 64 字符（默认）→ 约 12.5% 重叠，确保跨块信息连续
"""
import re
import asyncio
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
    3. 长段落使用 SemanticChunker 二次切分（有 embedder 时）
       或回退到 FixedSizeChunker（无 embedder 时）

    适用于 Markdown、文档等有明确结构的文本。

    Args:
        chunk_size: 每个块的最大字符数
        overlap: 相邻块的重叠字符数
        embedder: 文本嵌入器实例（可选，启用语义二次切分）
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, embedder: Any | None = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._embedder = embedder

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
                # 长段落二次切分：有 embedder 用语义切分，否则用固定大小
                if self._embedder:
                    sub = SemanticChunker(
                        embedder=self._embedder,
                        max_chunk_size=self.chunk_size,
                    ).chunk(section, meta)
                else:
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


class SemanticChunker(Chunker):
    """语义分块器：基于 embedding 相似度在语义断点处切分文本。

    核心思想：
      语义相近的句子应该在同一个块中，语义差异大的地方就是天然的切分边界。
      通过计算相邻句子的 embedding 余弦相似度，找到相似度骤降的位置（语义断点），
      在这些位置切分文本，就能得到语义连贯的块。

    完整流程（chunk_async 是核心，chunk 是对外的同步包装）：
      ┌─────────────────────────────────────────────────────────────────┐
      │  原始文本                                                       │
      │     │                                                           │
      │     ▼  _split_sentences()                                       │
      │  ["句子A。", "句子B。", "句子C。", "句子D。", "句子E。"]           │
      │     │                                                           │
      │     ▼  embedder.embed_batch()  ← 异步，需要调用 API 或本地模型    │
      │  [[0.1, 0.9, ...], [0.2, 0.8, ...], ...]  每句一个向量           │
      │     │                                                           │
      │     ▼  _cosine_similarity()  逐对计算相邻句子的相似度              │
      │  [0.92, 0.88, 0.31, 0.85]  句子A-B:0.92, B-C:0.88, C-D:0.31... │
      │     │                                                           │
      │     ▼  _find_breakpoints()  统计方法找相似度骤降的位置             │
      │  [3]  → 第3和第4个句子之间相似度骤降（0.31），在此切分             │
      │     │                                                           │
      │     ▼  在断点处切分                                              │
      │  ["句子A。句子B。句子C。", "句子D。句子E。"]                       │
      │     │                                                           │
      │     ▼  _merge_small_chunks() + _split_large_chunks()            │
      │  合并过短的块（避免碎片化），拆分过长的块（避免超出限制）           │
      │     │                                                           │
      │     ▼                                                           │
      │  [Chunk(text="句子A。句子B。句子C。", ...), Chunk(...)]            │
      └─────────────────────────────────────────────────────────────────┘

    为什么需要三个方法（chunk / chunk_async / _find_breakpoints）：
      - _find_breakpoints(): 纯数学计算，从相似度列表中找切分位置。
        这是算法核心，独立出来方便测试和复用。
      - chunk_async(): 完整的异步分块流程。必须是异步的，因为 Embedder
        需要调用远程 API（如 DashScope）或本地模型推理，这些都是 I/O 密集操作，
        用 async 可以不阻塞事件循环。
      - chunk(): 同步兼容入口。基类 Chunker 定义的接口是同步的，有些调用方
        不在异步上下文中（比如脚本、测试），chunk() 自动处理 async/sync 桥接。

    Args:
        embedder: 文本嵌入器实例（需有 embed_batch 方法）。
                  传 None 时 chunk() 会使用降级方案（纯按句子分组，不调模型）。
        max_chunk_size: 单个块的最大字符数（防止块过大，超出 embedding 模型限制）。
        min_chunk_size: 单个块的最小字符数（合并过短的块，避免碎片化）。
        breakpoint_threshold_amount: 断点阈值系数。
            阈值 = mean(sim) - std(sim) * 此系数。
            值越大 → 阈值越低 → 更少的断点 → 更大的块。
            值越小 → 阈值越高 → 更多的断点 → 更小的块。
            默认 1.0，通常在 0.5 ~ 1.5 之间调整。
        sentence_split_regex: 句子切分正则表达式，默认按中英文句号等切分。
    """

    # 中英文句子切分正则：使用正向后顾（lookbehind）在标点后切分
    # (?<=[。！？；.!?;])  →  匹配标点后面的位置
    # \s*                  →  标点后的空白也作为分隔符
    # 效果："Hello world. How are you? Fine!" → ["Hello world.", "How are you?", "Fine!"]
    DEFAULT_SENTENCE_REGEX = r'(?<=[。！？；.!?;])\s*'

    def __init__(
        self,
        embedder: Any | None = None,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 50,
        breakpoint_threshold_amount: float = 1.0,
        sentence_split_regex: str | None = None,
    ):
        # Embedder 实例，用于将句子转换为向量。
        # 通常是 src.rag.embedder.Embedder 的实例，内部使用 sentence-transformers。
        # 传 None 表示不使用语义分块，chunk() 会降级为纯句子分组。
        self._embedder = embedder

        # 块大小约束：防止块太小（碎片化）或太大（超出模型限制）
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size

        # 断点阈值系数：控制切分的激进程度。
        # 具体阈值 = mean(相似度) - std(相似度) * 此系数。
        # 系数越大 → 阈值越低 → 保留更多相似度低的边界不切 → 块更大。
        self._breakpoint_threshold_amount = breakpoint_threshold_amount

        # 句子切分正则，可自定义（比如按换行切分代码）
        self._sentence_split_regex = sentence_split_regex or self.DEFAULT_SENTENCE_REGEX

    # ──────────────────────────────────────────────────────────────
    # 第一步：文本预处理 — 将长文本切为句子列表
    # ──────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """将文本按标点符号切分为句子列表。

        这是整个语义分块流程的起点。切分粒度是"句子"而非"词"或"段落"，
        因为句子是语义完整性的最小单位 — 太小（词）会丢失语境，太大（段落）
        会掩盖语义边界。

        Args:
            text: 原始文本

        Returns:
            非空句子列表，已去除首尾空白

        示例：
            >>> chunker = SemanticChunker()
            >>> chunker._split_sentences("你好。世界！Hello.")
            ["你好。", "世界！", "Hello."]
        """
        sentences = re.split(self._sentence_split_regex, text)
        # 去除切分产生的空白字符串（比如连续标点之间）
        return [s.strip() for s in sentences if s.strip()]

    # ──────────────────────────────────────────────────────────────
    # 第二步：向量相似度计算 — 判断相邻句子的语义接近程度
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。

        余弦相似度衡量两个向量的方向一致性，不关心长度：
          - 1.0 → 方向完全一致（语义相同）
          - 0.0 → 方向正交（语义无关）
          - -1.0 → 方向完全相反（语义对立）

        公式：cos(θ) = (A·B) / (|A| × |B|)

        为什么用余弦而不是欧氏距离：
          - Embedding 向量的长度（模）不代表语义信息
          - 余弦相似度只看方向，更适合衡量语义接近程度
          - 值域固定 [-1, 1]，容易设定阈值

        Args:
            a: 第一个向量（句子 A 的 embedding）
            b: 第二个向量（句子 B 的 embedding）

        Returns:
            余弦相似度，范围 [-1, 1]
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        # 防止零向量除零
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    # ──────────────────────────────────────────────────────────────
    # 第三步：断点检测 — 找到语义切换的位置（算法核心）
    # ──────────────────────────────────────────────────────────────

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """根据相似度分布，用统计方法找到语义断点位置。

        这是语义分块的核心算法。思路是：
          1. 计算所有相邻句子相似度的均值和标准差
          2. 设定阈值 = 均值 - 标准差 × 系数
          3. 相似度低于阈值的位置 = 语义发生了明显切换 = 切分点

        为什么用统计方法而不是固定阈值：
          - 不同文档的相似度分布不同（技术文档普遍相似度高，对话体普遍低）
          - 固定阈值（比如 0.5）对某些文档太敏感，对另一些又太迟钝
          - 用 mean - std 自适应每篇文档的分布，只捕捉"异常低"的相似度

        参数系数（breakpoint_threshold_amount）的影响：
          - 系数 = 0 → 阈值 = 均值 → 凡是低于平均相似度的地方都切 → 块很碎
          - 系数 = 1 → 阈值 = 均值 - 标准差 → 只切明显异常低的地方 → 适中
          - 系数 = 2 → 阈值 = 均值 - 2×标准差 → 只切极端异常的地方 → 块很大

        Args:
            similarities: 相邻句子的余弦相似度列表。
                          例如 [0.92, 0.88, 0.31, 0.85] 表示：
                            句子0-1: 0.92（很相似）
                            句子1-2: 0.88（很相似）
                            句子2-3: 0.31（语义骤降！）
                            句子3-4: 0.85（很相似）

        Returns:
            断点索引列表。每个索引表示"在此句子之前切分"。
            对于上面的例子，返回 [3]（在第3个句子之前切分）

        示例：
            >>> chunker = SemanticChunker(breakpoint_threshold_amount=1.0)
            >>> chunker._find_breakpoints([0.92, 0.88, 0.31, 0.85])
            [3]  # 第3个句子（索引2→3）的相似度0.31远低于阈值，在此切分
        """
        if not similarities:
            return []

        # 计算相似度的均值
        n = len(similarities)
        mean_sim = sum(similarities) / n

        # 计算标准差（衡量相似度的波动程度）
        # 标准差大 → 句子间相似度差异大 → 有明显的语义切换
        # 标准差小 → 句子间相似度差不多 → 语义均匀
        variance = sum((s - mean_sim) ** 2 for s in similarities) / n if n > 1 else 0
        std_sim = variance ** 0.5

        # 阈值 = 均值 - 标准差 × 系数
        # 低于此阈值的相似度被视为"异常低"，即语义断点
        threshold = mean_sim - std_sim * self._breakpoint_threshold_amount

        # 遍历所有相似度，找到低于阈值的位置
        # similarities[i] 表示句子 i 和句子 i+1 的相似度
        # 如果 similarities[i] < threshold，说明句子 i 和 i+1 语义差异大
        # 切分点是 i+1（在句子 i+1 之前切分）
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # 在第 i+1 个句子之前切分

        return breakpoints

    # ──────────────────────────────────────────────────────────────
    # 第四步：后处理 — 合并小块、拆分大块
    # ──────────────────────────────────────────────────────────────

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """合并过短的相邻块，避免碎片化。

        语义断点可能很密集，导致产生大量只有几个字的碎片块。
        这些碎片块对检索没有价值（语义信息太少），需要合并到相邻块中。

        策略：遍历块列表，如果当前块太短，就和下一个块合并。

        Args:
            chunks: 原始切分后的文本块列表

        Returns:
            合并后的文本块列表（所有块 >= min_chunk_size，除最后一个可能略短）

        示例（min_chunk_size=20）：
            输入: ["短。", "也很短。", "这是一个足够长的句子内容。"]
            输出: ["短。\n也很短。", "这是一个足够长的句子内容。"]
        """
        if not chunks:
            return chunks

        merged = [chunks[0]]
        for chunk in chunks[1:]:
            # 如果前一个块太短，把当前块合并进去
            if len(merged[-1]) < self._min_chunk_size:
                merged[-1] += "\n" + chunk
            else:
                merged.append(chunk)

        # 最后一个块如果太短，合并到倒数第二个
        if len(merged) > 1 and len(merged[-1]) < self._min_chunk_size:
            merged[-2] += "\n" + merged[-1]
            merged.pop()

        return merged

    def _split_large_chunks(self, chunks: list[str]) -> list[str]:
        """拆分过大的块，确保不超过 max_chunk_size。

        某些情况下，语义断点之间可能包含大量连续文本（比如一段长论述），
        导致单个块超出 embedding 模型的输入长度限制。
        这种情况用固定大小切分作为兜底。

        Args:
            chunks: 文本块列表

        Returns:
            拆分后的文本块列表（所有块 <= max_chunk_size）
        """
        result = []
        for chunk in chunks:
            if len(chunk) <= self._max_chunk_size:
                result.append(chunk)
            else:
                # 超大块用固定大小切分兜底，overlap=0 避免重复
                sub = FixedSizeChunker(self._max_chunk_size, overlap=0).chunk(chunk)
                result.extend([c.text for c in sub])
        return result

    # ──────────────────────────────────────────────────────────────
    # 主入口方法：chunk_async（核心）和 chunk（同步包装）
    # ──────────────────────────────────────────────────────────────

    async def chunk_async(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """异步语义分块 — 这是语义分块的真正执行入口。

        为什么是异步的：
          - embedder.embed_batch() 内部调用 embedding 模型（如 bge-m3）
          - 如果是远程 API（DashScope、OpenAI），这是网络 I/O
          - 如果是本地模型，这是 CPU/GPU 密集计算
          - 无论哪种，用 async 都能让调用方的事件循环继续处理其他任务

        调用链：
          chunk_async()
            ├─ _split_sentences()          纯 CPU，同步
            ├─ embedder.embed_batch()      ← 异步！这一步决定了整个方法必须是 async
            ├─ _cosine_similarity()        纯 CPU，同步
            ├─ _find_breakpoints()         纯 CPU，同步
            ├─ _merge_small_chunks()       纯 CPU，同步
            └─ _split_large_chunks()       纯 CPU，同步

        Args:
            text: 原始文本
            metadata: 可选的元数据（会复制到每个 Chunk）

        Returns:
            Chunk 列表，每个 Chunk 包含文本、元数据和序号
        """
        if not text.strip():
            return []
        if not self._embedder:
            raise ValueError("SemanticChunker 需要 embedder 实例才能进行语义分块")

        meta = metadata or {}

        # ── 第一步：切句子 ──
        # 将长文本按标点切分为句子列表，作为后续处理的最小单元
        sentences = self._split_sentences(text)

        # 如果只有 1 个或 0 个句子，无需分块，直接返回
        if len(sentences) <= 1:
            return [Chunk(text=text, metadata=meta.copy(), index=0, source=meta.get("source", ""))]

        # ── 第二步：计算句子向量 ──
        # 这是最耗时的一步：将每个句子通过 embedding 模型转换为高维向量
        # 例如 bge-m3 输出 1024 维向量
        # embed_batch 比逐条 embed 更高效（利用了 GPU 批量推理）
        embeddings = await self._embedder.embed_batch(sentences)

        # ── 第三步：计算相邻句子的余弦相似度 ──
        # similarities[i] = 句子 i 和句子 i+1 的语义相似度
        # 相似度高 → 语义相近 → 应该在同一个块中
        # 相似度低 → 语义切换 → 可能是切分点
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # ── 第四步：找到语义断点 ──
        # 用统计方法（mean - std * coefficient）找出相似度异常低的位置
        # 这些位置就是语义发生切换的边界
        breakpoints = self._find_breakpoints(similarities)

        # ── 第五步：在断点处切分句子 ──
        # 把句子列表按照断点分成若干组，每组合并为一个原始块
        # 例如 sentences = ["A。", "B。", "C。", "D。", "E。"]
        #      breakpoints = [3]
        #      → raw_chunks = ["A。\nB。\nC。", "D。\nE。"]
        raw_chunks = []
        start = 0
        for bp in breakpoints:
            raw_chunks.append("\n".join(sentences[start:bp]))
            start = bp
        raw_chunks.append("\n".join(sentences[start:]))  # 最后一段

        # ── 第六步：后处理 ──
        # 合并过短的块（避免碎片化）、拆分过长的块（避免超出限制）
        merged = self._merge_small_chunks(raw_chunks)
        final = self._split_large_chunks(merged)

        # ── 第七步：构造 Chunk 对象 ──
        # 将纯文本块包装为 Chunk 数据类，附带元数据和序号
        chunks = []
        for i, chunk_text in enumerate(final):
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=meta.copy(),
                    index=i,
                    source=meta.get("source", ""),
                ))
        return chunks

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """同步语义分块入口 — 对外暴露的统一接口。

        这个方法是基类 Chunker 定义的抽象方法的实现，目的是让调用方
        不需要关心底层是异步还是同步。

        为什么需要这个包装：
          - 基类 Chunker.chunk() 是同步方法（def，不是 async def）
          - 有些调用方不在异步上下文中（脚本、测试、同步函数）
          - chunk() 自动处理 async → sync 的桥接

        内部逻辑：
          1. 有 embedder → 调用 chunk_async()（真正的语义分块）
             - 如果当前在事件循环中（比如 FastAPI 里），用线程池避免死锁
             - 如果不在事件循环中，直接 asyncio.run()
          2. 无 embedder → 调用 _chunk_by_sentences()（降级方案，不调模型）

        Args:
            text: 原始文本
            metadata: 可选的元数据

        Returns:
            Chunk 列表
        """
        if self._embedder:
            try:
                # 尝试获取当前线程的事件循环
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 当前已经在异步上下文中（比如 FastAPI 路由里）
                    # 不能在运行中的事件循环里再调 asyncio.run()，会死锁
                    # 解决方案：开一个新线程，在新线程里运行 asyncio.run()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self.chunk_async(text, metadata))
                        return future.result()
                # 当前没有运行中的事件循环，直接用 run_until_complete
                return loop.run_until_complete(self.chunk_async(text, metadata))
            except RuntimeError:
                # 没有事件循环，创建一个新的来运行
                return asyncio.run(self.chunk_async(text, metadata))
        else:
            # 降级方案：没有 embedder，不调模型，纯按句子分组
            # 适合：测试环境、不想加载模型、或者文本量很小的场景
            return self._chunk_by_sentences(text, metadata)

    def _chunk_by_sentences(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """降级分块方案：纯按句子分组，不调用 embedding 模型。

        当没有 embedder 时（测试、轻量场景），退化为简单的句子分组：
        贪心地累积句子，直到当前组的长度接近 max_chunk_size 时换组。

        这不是语义分块，只是按长度分组，但比固定大小切分好：
          - 不会在句子中间切断
          - 保持句子的完整性

        Args:
            text: 原始文本
            metadata: 可选的元数据

        Returns:
            Chunk 列表
        """
        if not text.strip():
            return []
        meta = metadata or {}
        sentences = self._split_sentences(text)

        # 贪心分组：累积句子直到接近 max_chunk_size
        chunks_text = []
        current = []        # 当前组的句子列表
        current_len = 0     # 当前组的总字符数
        for sent in sentences:
            # 加入当前句子会超出限制 → 开始新的一组
            if current_len + len(sent) > self._max_chunk_size and current:
                chunks_text.append("\n".join(current))
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)
        # 别忘了最后一组
        if current:
            chunks_text.append("\n".join(current))

        # 构造 Chunk 对象
        chunks = []
        for i, ct in enumerate(chunks_text):
            if ct.strip():
                chunks.append(Chunk(text=ct, metadata=meta.copy(), index=i, source=meta.get("source", "")))
        return chunks
