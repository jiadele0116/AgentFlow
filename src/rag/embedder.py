"""
文本嵌入模块 (Text Embedding)

将文本转换为向量表示，用于语义分块和向量检索。

支持两种模式：
  1. API 模式（device="api"）：调用 DashScope API
     - 模型：text-embedding-v3（默认）
     - 维度：1024（可配置 64/128/256/512/768/1024）
     - 优点：无需本地 GPU，开箱即用
     - 缺点：需要网络，有 API 调用成本

  2. 本地模式（device="cpu"/"cuda"）：加载 sentence-transformers 模型
     - 模型：BAAI/bge-m3（默认）
     - 维度：1024
     - 优点：离线可用，无调用成本
     - 缺点：需要下载模型（~2GB），首次加载慢

使用示例：
    # API 模式（推荐）
    embedder = Embedder(model="text-embedding-v3", device="api", dimensions=1024)
    vector = await embedder.embed("你好世界")
    vectors = await embedder.embed_batch(["句子1", "句子2"])

    # 本地模式
    embedder = Embedder(model="BAAI/bge-m3", device="cpu")
    vector = await embedder.embed("你好世界")
"""
import asyncio
import httpx

from src.core.logging import get_logger

logger = get_logger("rag.embedder")


class Embedder:
    """文本嵌入器：将文本转换为向量表示。

    根据 device 参数自动选择 API 模式或本地模式。

    Args:
        model: 模型名称
               - API 模式：DashScope 模型 ID（如 "text-embedding-v3"）
               - 本地模式：HuggingFace 模型 ID（如 "BAAI/bge-m3"）
        device: 计算设备（"api" / "cpu" / "cuda" / "mps"）
        batch_size: 批量编码的批次大小
        dimensions: 输出向量维度（仅 API 模式生效）
        api_key: DashScope API Key（仅 API 模式，不传则从环境变量读取）
        base_url: API 基础 URL（仅 API 模式）
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        device: str = "api",
        batch_size: int = 32,
        dimensions: int = 1024,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._api_key = api_key
        self._base_url = base_url

        # 本地模式：延迟加载 sentence-transformers（首次调用时才加载模型）
        self._local_model = None

    @property
    def dimension(self) -> int:
        """返回嵌入向量的维度。"""
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """将单条文本转换为向量。

        Args:
            text: 待嵌入的文本

        Returns:
            浮点数列表表示的向量
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。

        根据 device 自动选择 API 调用或本地模型推理。
        API 模式下会自动分批处理，避免单次请求过大。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            向量列表，与输入文本一一对应
        """
        if not texts:
            return []

        if self._device == "api":
            return await self._embed_via_api(texts)
        else:
            return await self._embed_via_local(texts)

    # ──────────────────────────────────────────────────────────────
    # API 模式：调用 DashScope 的 OpenAI 兼容接口
    # ──────────────────────────────────────────────────────────────

    async def _embed_via_api(self, texts: list[str]) -> list[list[float]]:
        """通过 DashScope API 批量获取文本向量。

        使用 OpenAI 兼容的 /v1/embeddings 接口。
        自动将大批次拆分为多个小批次，避免超出 API 限制。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        all_embeddings = []

        # 按 batch_size 分批处理
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            batch_embeddings = await self._call_embedding_api(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """调用一次 DashScope embedding API。

        Args:
            texts: 一批文本（不超过 batch_size）

        Returns:
            这批文本的向量列表

        Raises:
            RuntimeError: API 调用失败
        """
        # 构建请求头
        headers = {
            "Content-Type": "application/json",
        }
        # API Key 优先用构造参数，其次从环境变量读取
        api_key = self._api_key
        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 构建请求体
        # DashScope 的 OpenAI 兼容接口格式
        payload = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimensions,
        }

        url = f"{self._base_url}/embeddings"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            # 解析响应：按 index 排序确保顺序正确
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API 调用失败: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Embedding API 调用失败: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Embedding API 调用失败: {e}")
            raise RuntimeError(f"Embedding API 调用失败: {e}") from e

    # ──────────────────────────────────────────────────────────────
    # 本地模式：使用 sentence-transformers 加载模型
    # ──────────────────────────────────────────────────────────────

    def _ensure_local_model(self):
        """确保本地模型已加载（延迟加载，首次调用时才加载）。"""
        if self._local_model is None:
            logger.info(f"正在加载本地嵌入模型: {self._model} (设备: {self._device})")
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._model, device=self._device)
            # 更新维度为模型实际输出维度
            self._dimensions = self._local_model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成，向量维度: {self._dimensions}")

    async def _embed_via_local(self, texts: list[str]) -> list[list[float]]:
        """通过本地 sentence-transformers 模型批量获取文本向量。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        # 模型加载是 CPU 密集操作，放到线程池避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """同步编码（在线程池中执行）。"""
        self._ensure_local_model()
        embeddings = self._local_model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]
