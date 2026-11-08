"""Hybrid retrieval: dense and lexical search fused by reciprocal rank, then expanded.

The top ``top_k`` chunks from each branch are fused, the best ``context_blocks`` kept,
and neighbouring chunks merged back into spans of prose.
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
from myscholar.ports.store import SearchFilters, SearchHit
from myscholar.retrieval.fusion import weighted_branches
from myscholar.retrieval.merge import SeedHit, Span, merge_windows


class RetrievalMode(StrEnum):
    """Which retrieval branches run: DENSE alone, or HYBRID with the lexical branch.

    Both share one code path, so ablations differ only in the lexical branch. A
    cross-encoder reranker lowered routed MRR from 0.865 to 0.771 and hit@1 from 0.84
    to 0.65 (results/report.md), so ranking is RRF only.
    """

    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    blocks: tuple[ContextBlock, ...]

    ranked: tuple[SearchHit, ...]
    """The final ordering handed to expansion, which the evaluation measures."""

    candidates: tuple[SearchHit, ...]
    """Everything fusion produced, before the limit."""

    mode: RetrievalMode
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def document_ids(self) -> list[UUID]:
        """Documents in final rank order, deduplicated."""
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
        top_k: int = 50,
        context_blocks: int = 8,
        window: int = 2,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
        max_span_chunks: int = 8,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._context_blocks = context_blocks
        self._window = window
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight
        self._rrf_k = rrf_k
        self._max_span_chunks = max_span_chunks

    async def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        filters: SearchFilters | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        timings: dict[str, int] = {}
        limit = limit or self._context_blocks

        started = time.monotonic()
        candidates = await self._search(query, mode=mode, filters=filters)
        timings["search"] = _elapsed(started)

        if not candidates:
            return RetrievalResult(
                blocks=(), ranked=(), candidates=(), mode=mode, timings_ms=timings
            )

        selected = candidates[:limit]

        started = time.monotonic()
        blocks = await self._expand(selected)
        timings["expand"] = _elapsed(started)

        return RetrievalResult(
            blocks=tuple(blocks),
            ranked=tuple(selected),
            candidates=tuple(candidates),
            mode=mode,
            timings_ms=timings,
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
        # Each hit carries its fused score, which orders the spans downstream.
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

        return ContextBlock(
            label=f"C{label_index}",
            document_id=span.document_id,
            section_id=span.section_id,
            document_title=context.document_title,
            section_heading=context.section_heading,
            text="\n\n".join(row.text for row in rows),
            page_start=context.page_start,
            page_end=context.page_end,
            score=span.max_score,
            document_kind=context.document_kind,
            is_review=context.is_review,
            authors=tuple(context.authors or ()),
            source_url=context.source_url,
        )


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
