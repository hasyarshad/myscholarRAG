"""A reranker that preserves the incoming order."""

from __future__ import annotations

from collections.abc import Sequence

from myscholar.ports.reranker import Reranker


class NoopReranker(Reranker):
    """Returns the first ``top_n`` passages unchanged.

    Used in tests, and in production when the real reranker is unavailable. In
    that second case the pipeline marks the answer as degraded rather than
    failing the request — reduced precision beats an error page.
    """

    @property
    def model_id(self) -> str:
        return "noop"

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        # Descending scores preserve input order under a stable sort downstream.
        return [(i, 1.0 - i * 1e-6) for i in range(min(top_n, len(passages)))]
