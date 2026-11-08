"""GPU embedding with sentence-transformers, used locally for ingestion.

The ONNX adapter serves the same weights on CPU in production, so the model is
small enough to run in both places.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from myscholar.adapters.embedding import QUERY_INSTRUCTION
from myscholar.ports.embedding import Embedder, Vector


class TorchEmbedder(Embedder):
    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        *,
        device: str = "cuda",
        batch_size: int = 32,
        query_instruction: str | None = QUERY_INSTRUCTION,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._query_instruction = query_instruction or ""
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Loaded on first use, so constructing a container stays cheap."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id, device=self._device)
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> list[Vector]:
        # L2-normalised, matching the ONNX adapter's output.
        vectors = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        # Encoding blocks even on a GPU, so it runs in a thread off the event loop.
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> Vector:
        prefixed = f"{self._query_instruction}{text}"
        vectors = await asyncio.to_thread(self._encode, [prefixed])
        return vectors[0]
