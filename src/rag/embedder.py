"""Text embedding using sentence-transformers."""
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 32):
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        return self._model.encode([text], batch_size=1)[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
