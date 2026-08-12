"""Cross-encoder reranking on the GPU.

A bi-encoder embeds the question and each chunk separately, so a chunk's vector
must summarise it for every possible query and cannot emphasise the part
actually asked about. A cross-encoder scores the pair jointly and can — at the
cost of one forward pass per candidate, which is why it reranks a shortlist
rather than searching the corpus.

Measured on this corpus: 50 pairs in roughly 60ms on a 5090. Scoring all 2,896
chunks would take about 3.5 seconds per query, before generation begins.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from myscholar.ports.reranker import Reranker


class TorchReranker(Reranker):
    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-base",
        *,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        # Question plus passage must fit. Chunks are ~350 tokens and questions
        # are short, so 512 covers the pair without truncating the passage —
        # truncation would silently score only the opening of each chunk.
        self._max_length = max_length
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_id, device=self._device, max_length=self._max_length
            )
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_id

    def _score(self, query: str, passages: Sequence[str]) -> list[float]:
        scores = self.model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not passages:
            return []

        scores = await asyncio.to_thread(self._score, query, passages)
        ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])
        return ranked[:top_n]
