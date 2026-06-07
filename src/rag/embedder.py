"""
文本嵌入模块 (Text Embedding)

使用 sentence-transformers 将文本转换为稠密向量表示。
默认使用 BAAI/bge-m3 多语言嵌入模型，支持中英文等多语言。

嵌入流程：
  文本 → SentenceTransformer.encode() → 归一化向量 → list[float]

bge-m3 模型特点：
  - 维度：1024
  - 支持多语言（中、英、日、韩等）
  - 最大输入长度：8192 tokens
  - 支持 dense 和 sparse 两种表示（此处仅用 dense）

性能考虑：
  - 单条文本使用 embed()，批量使用 embed_batch()
  - batch_size 控制同时处理的文本数，影响内存和速度
  - CPU 模式下较慢，GPU 可显著加速
"""
from sentence_transformers import SentenceTransformer


class Embedder:
    """文本嵌入器：将文本转换为向量表示。

    封装 SentenceTransformer 模型，提供异步的嵌入接口。

    Args:
        model_name: HuggingFace 模型名称或本地路径
        device: 计算设备 ("cpu", "cuda", "mps")
        batch_size: 批量编码的批次大小
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 32):
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        """返回嵌入向量的维度。bge-m3 为 1024 维。"""
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        """将单条文本转换为向量。

        Args:
            text: 待嵌入的文本

        Returns:
            浮点数列表表示的向量
        """
        return self._model.encode([text], batch_size=1)[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。

        使用配置的 batch_size 进行批量编码，比逐条调用 embed() 更高效。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            向量列表，与输入文本一一对应
        """
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
