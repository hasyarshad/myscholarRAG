"""GPU embedding with sentence-transformers.

Used at ingest, where there is a GPU and thousands of chunks to process. The
ONNX adapter (Phase 7) implements the same port for CPU-only production, with
the *same weights* — which is why the model is deliberately small enough to run
in both places.

``model_id`` and ``dim`` are read from the loaded model rather than from
configuration, so the identity stored with every vector is necessarily the
identity of the thing that produced it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from myscholar.ports.embedding import Embedder, Vector

#: BGE models were trained with an instruction prefix on queries but not on
#: passages. Omitting it costs a few points of retrieval accuracy; applying it
#: to passages as well is worse. The asymmetry lives here, inside one class,
#: rather than at call sites where the two could drift apart.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class TorchEmbedder(Embedder):
    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        *,
        device: str = "cuda",
        batch_size: int = 32,
        query_instruction: str | None = _QUERY_INSTRUCTION,
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
        # Normalised so cosine similarity reduces to a dot product, which is
        # what pgvector's inner-product operator computes.
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
        # Encoding is CPU-blocking even when the maths runs on the GPU, so keep
        # it off the event loop; ingestion runs other work concurrently.
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> Vector:
        prefixed = f"{self._query_instruction}{text}"
        vectors = await asyncio.to_thread(self._encode, [prefixed])
        return vectors[0]
