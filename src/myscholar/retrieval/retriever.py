"""Hybrid retrieval: search, fuse, rerank, expand.

    dense (2896 chunks)  ─┐
                          ├─ RRF ─→ 50 ─→ cross-encoder ─→ 8 ─→ parent spans
    lexical (2896 chunks) ┘

Cheap methods narrow the field; the expensive method fixes the order of what
survives; expansion restores the prose the chunking deliberately broke up.

``RetrievalMode`` exists so the three configurations can be compared on the
same code path rather than through three implementations — an ablation where
the variants differ in more than the thing under test measures nothing.

See docs/learning/02-hybrid-search-and-reranking.md.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from myscholar.adapters.persistence.repository import PostgresStore
from myscholar.domain.documents import ContextBlock
from myscholar.ports.embedding import Embedder
from myscholar.ports.reranker import Reranker
from myscholar.ports.store import SearchFilters, SearchHit
from myscholar.retrieval.fusion import weighted_branches
from myscholar.retrieval.merge import SeedHit, Span, merge_windows


class RetrievalMode(StrEnum):
    """Which parts of the pipeline are active.

    DENSE reproduces the Phase 1 baseline, HYBRID adds the lexical branch, and
    FULL adds the cross-encoder. Each is the previous one plus one component,
    so a difference in the results is attributable.
    """

    DENSE = "dense"
    HYBRID = "hybrid"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    blocks: tuple[ContextBlock, ...]

    ranked: tuple[SearchHit, ...]
    """The final ordering, after reranking. What the evaluation measures.

    Measuring the pre-rerank candidates instead makes the reranker invisible by
    construction — every mode scores identically, which is not a null result but
    a broken experiment.
    """

    candidates: tuple[SearchHit, ...]
    """Post-fusion, pre-rerank. Kept for debugging the branches."""

    mode: RetrievalMode
    timings_ms: dict[str, int] = field(default_factory=dict)
    degraded: bool = False
    """True when reranking was requested but unavailable."""

    @property
    def document_ids(self) -> list[UUID]:
        """Documents in final rank order, deduplicated.

        Document level, because "did it find the right paper" is the question a
        visitor is really asking, and one paper contributing several chunks
        should not count as several results.
        """
        seen: list[UUID] = []
        for hit in self.ranked:
            if hit.document_id not in seen:
                seen.append(hit.document_id)
        return seen


class HybridRetriever:
    def __init__(
        self,
        *,
        store: PostgresStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        top_k: int = 50,
        rerank_k: int = 8,
        window: int = 2,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
        max_span_chunks: int = 8,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_k = rerank_k
        self._window = window
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        self._rrf_k = rrf_k
        self._max_span_chunks = max_span_chunks

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.FULL,
        filters: SearchFilters | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        timings: dict[str, int] = {}
        limit = limit or self._rerank_k

        started = time.monotonic()
        candidates = await self._search(query, mode=mode, filters=filters)
        timings["search"] = _elapsed(started)

        if not candidates:
            return RetrievalResult(
                blocks=(), ranked=(), candidates=(), mode=mode, timings_ms=timings
            )

        degraded = False
        if mode is RetrievalMode.FULL:
            started = time.monotonic()
            reranked, degraded = await self._rerank(query, candidates, limit)
            timings["rerank"] = _elapsed(started)
        else:
            reranked = candidates[:limit]

        started = time.monotonic()
        blocks = await self._expand(reranked)
        timings["expand"] = _elapsed(started)

        return RetrievalResult(
            blocks=tuple(blocks),
            ranked=tuple(reranked),
            candidates=tuple(candidates),
            mode=mode,
            timings_ms=timings,
            degraded=degraded,
        )

    async def _search(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        filters: SearchFilters | None,
    ) -> list[SearchHit]:
        """Run the branches concurrently and fuse them."""
        dense_task = self._store.search_dense(
            await self._embedder.embed_query(query), limit=self._top_k, filters=filters
        )
        if mode is RetrievalMode.DENSE:
            return await dense_task

        dense, lexical = await asyncio.gather(
            dense_task,
            self._store.search_lexical(query, limit=self._top_k, filters=filters),
        )

        by_id = {hit.chunk_id: hit for hit in [*dense, *lexical]}
        fused = weighted_branches(
            [hit.chunk_id for hit in dense],
            [hit.chunk_id for hit in lexical],
            dense_weight=self._dense_weight,
            lexical_weight=self._lexical_weight,
            k=self._rrf_k,
            limit=self._top_k,
        )
        # Carry the fusion score onto the hit, so downstream ordering and the
        # span scores reflect the fused ranking rather than one branch's score.
        return [
            SearchHit(
                chunk_id=by_id[chunk_id].chunk_id,
                document_id=by_id[chunk_id].document_id,
                section_id=by_id[chunk_id].section_id,
                ordinal_in_section=by_id[chunk_id].ordinal_in_section,
                text=by_id[chunk_id].text,
                score=score,
            )
            for chunk_id, score in fused
        ]

    async def _rerank(
        self, query: str, candidates: list[SearchHit], limit: int
    ) -> tuple[list[SearchHit], bool]:
        """Reorder candidates with the cross-encoder.

        A reranker failure degrades to fusion order and is reported, rather
        than failing the request: reduced precision beats an error page.
        """
        if self._reranker is None:
            return candidates[:limit], True

        try:
            ranked = await self._reranker.rerank(
                query, [hit.text for hit in candidates], top_n=limit
            )
        except Exception:  # noqa: BLE001 - any reranker failure degrades the same way
            return candidates[:limit], True

        return [
            SearchHit(
                chunk_id=candidates[index].chunk_id,
                document_id=candidates[index].document_id,
                section_id=candidates[index].section_id,
                ordinal_in_section=candidates[index].ordinal_in_section,
                text=candidates[index].text,
                score=score,
            )
            for index, score in ranked
        ], False

    async def _expand(self, hits: list[SearchHit]) -> list[ContextBlock]:
        """Merge neighbouring hits into contiguous spans and fetch their text."""
        section_lengths = await self._store.section_chunk_counts({hit.section_id for hit in hits})
        spans = merge_windows(
            [
                SeedHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    section_id=hit.section_id,
                    ordinal=hit.ordinal_in_section,
                    score=hit.score,
                )
                for hit in hits
            ],
            window=self._window,
            max_chunks=self._max_span_chunks,
            section_lengths=section_lengths,
        )

        blocks = await asyncio.gather(
            *(self._block_for(span, i) for i, span in enumerate(spans, 1))
        )
        return [block for block in blocks if block is not None]

    async def _block_for(self, span: Span, label_index: int) -> ContextBlock | None:
        rows = await self._store.fetch_span(span.section_id, span.start, span.end)
        if not rows:
            return None

        context = await self._store.span_context(span.section_id)
        if context is None:
            return None
        document_title, section_heading, page_start, page_end = context

        return ContextBlock(
            label=f"C{label_index}",
            document_id=span.document_id,
            section_id=span.section_id,
            document_title=document_title,
            section_heading=section_heading,
            text="\n\n".join(row.text for row in rows),
            page_start=page_start,
            page_end=page_end,
            score=span.max_score,
        )


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
