"""The reranking interface.

A cross-encoder scores each (query, passage) pair jointly, which is far more
accurate than comparing independently-computed embeddings — and far too slow to
run over the whole corpus. So it reranks the shortlist that search returns.

A no-op implementation exists deliberately: if the reranker is unavailable in
production, the pipeline degrades to fusion-ranked results and says so, rather
than failing the request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Reranker(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        """Score passages against the query.

        Returns ``(original_index, score)`` pairs, best first, at most ``top_n``.
        Indices refer to positions in ``passages`` so callers can map results
        back to their own objects without this port knowing about them.
        """
