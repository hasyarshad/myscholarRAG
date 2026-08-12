"""Classify, plan, then execute — the routed entry point to retrieval.

This is what the API and the CLI call. It owns the decision of whether to search
at all, which is the decision that removes most of a public bot's hallucination
surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import UUID

from myscholar.adapters.persistence.repository import PostgresStore
from myscholar.domain.documents import ContextBlock
from myscholar.ports.store import SearchFilters
from myscholar.query.bibliographic import BibliographicAnswer, BibliographicQuery
from myscholar.query.classifier import Classification, QueryClassifier
from myscholar.query.planner import Plan, QueryPlanner
from myscholar.query.transforms import QueryTransformer, SearchQuery
from myscholar.retrieval.fusion import reciprocal_rank_fusion
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode


@dataclass(frozen=True, slots=True)
class RoutedResult:
    """What routing produced, and how it got there."""

    classification: Classification
    plan: Plan
    blocks: tuple[ContextBlock, ...] = ()
    bibliographic: BibliographicAnswer | None = None
    queries: tuple[SearchQuery, ...] = ()
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def answered_without_retrieval(self) -> bool:
        return self.plan.is_terminal or self.plan.use_sql


class QueryRouter:
    def __init__(
        self,
        *,
        classifier: QueryClassifier,
        planner: QueryPlanner,
        transformer: QueryTransformer,
        retriever: HybridRetriever,
        store: PostgresStore,
    ) -> None:
        self._classifier = classifier
        self._planner = planner
        self._transformer = transformer
        self._retriever = retriever
        self._bibliographic = BibliographicQuery(store)
        self._store = store
        self._titles_loaded = False

    async def _ensure_titles(self) -> dict[str, UUID]:
        """Corpus titles, for entity linking and for resolving filters."""
        documents = await self._store.list_documents()
        if not self._titles_loaded:
            self._classifier.set_titles([d.title for d in documents])
            self._titles_loaded = True
        return {d.title: d.id for d in documents}

    async def route(self, question: str) -> RoutedResult:
        timings: dict[str, int] = {}
        title_to_id = await self._ensure_titles()

        started = time.monotonic()
        classification = await self._classifier.classify(question)
        timings["classify"] = _elapsed(started)

        plan = self._planner.plan(classification)

        if plan.is_terminal:
            return RoutedResult(classification=classification, plan=plan, timings_ms=timings)

        if plan.use_sql:
            started = time.monotonic()
            answer = await self._bibliographic.answer(question)
            timings["sql"] = _elapsed(started)
            return RoutedResult(
                classification=classification,
                plan=plan,
                bibliographic=answer,
                timings_ms=timings,
            )

        started = time.monotonic()
        queries = await self._transformer.apply(question, plan.transforms)
        timings["transform"] = _elapsed(started)

        started = time.monotonic()
        blocks = await self._search(queries, plan, title_to_id)
        timings["retrieve"] = _elapsed(started)

        return RoutedResult(
            classification=classification,
            plan=plan,
            blocks=tuple(blocks),
            queries=tuple(queries),
            timings_ms=timings,
        )

    async def _search(
        self,
        queries: list[SearchQuery],
        plan: Plan,
        title_to_id: dict[str, UUID],
    ) -> list[ContextBlock]:
        """Retrieve for every query and fuse the results.

        A filtered branch is always accompanied by an unfiltered one. An
        over-confident entity link would otherwise return nothing at all, and
        an imperfect result beats an empty one.
        """
        document_ids = frozenset(
            title_to_id[title] for title in plan.filter_document_titles if title in title_to_id
        )

        # The baseline search: document filter (if the entity linker resolved
        # one) but no section restriction.
        #
        # Section kinds are a *boost*, not a filter. Passing them as a filter
        # restricted methodological questions to methods and results sections,
        # excluding abstracts, introductions and conclusions — where answers
        # frequently are. Measured, that cost 9 points of MRR. So the boosted
        # search runs as an extra branch alongside the unrestricted one, and
        # fusion decides.
        base = SearchFilters(
            document_ids=document_ids,
            exclude_section_kinds=plan.exclude_sections,
        )
        boosted = (
            SearchFilters(
                document_ids=document_ids,
                section_kinds=plan.boost_sections,
                exclude_section_kinds=plan.exclude_sections,
            )
            if plan.boost_sections
            else None
        )

        branches: dict[str, list[str]] = {}
        block_by_key: dict[str, ContextBlock] = {}
        weights: dict[str, float] = {}

        async def run(name: str, text: str, filters: SearchFilters, weight: float) -> None:
            result = await self._retriever.retrieve(
                text,
                mode=RetrievalMode.FULL,
                filters=filters if not filters.is_empty else None,
                limit=plan.rerank_k,
            )
            # An empty filtered result means the filter was wrong, not that the
            # corpus lacks an answer.
            if not result.blocks and not filters.is_empty:
                result = await self._retriever.retrieve(
                    text, mode=RetrievalMode.FULL, limit=plan.rerank_k
                )
            keys: list[str] = []
            for block in result.blocks:
                # Keyed on the text, not the page range. Two distinct spans in
                # one section frequently share a page range, and keying on that
                # made them collide — the second was silently dropped, losing
                # real evidence from the fusion.
                key = f"{block.section_id}:{hash(block.text)}"
                block_by_key.setdefault(key, block)
                keys.append(key)
            branches[name] = keys
            weights[name] = weight

        for index, query in enumerate(queries):
            await run(f"{query.origin.value}:{index}", query.text, base, query.weight)
            if boosted is not None:
                await run(
                    f"{query.origin.value}:{index}:sections",
                    query.text,
                    boosted,
                    query.weight * 0.5,
                )

        if not branches:
            return []

        fused = reciprocal_rank_fusion(branches, weights=weights, limit=plan.rerank_k * 2)
        blocks = [block_by_key[key] for key, _ in fused if key in block_by_key]

        if plan.max_chunks_per_document is not None:
            blocks = _cap_per_document(blocks, plan.max_chunks_per_document)

        # Relabel after fusion so citation markers run C1..Cn in the order the
        # generator will see them.
        return [
            ContextBlock(
                label=f"C{i}",
                document_id=b.document_id,
                section_id=b.section_id,
                document_title=b.document_title,
                section_heading=b.section_heading,
                text=b.text,
                page_start=b.page_start,
                page_end=b.page_end,
                score=b.score,
            )
            for i, b in enumerate(blocks[: plan.rerank_k], start=1)
        ]


def _cap_per_document(blocks: list[ContextBlock], cap: int) -> list[ContextBlock]:
    """Limit how many blocks any one document contributes.

    Without this a verbose paper monopolises a synthesis question and the answer
    quietly becomes about that paper rather than about the corpus.
    """
    counts: dict[UUID, int] = {}
    kept: list[ContextBlock] = []
    for block in blocks:
        seen = counts.get(block.document_id, 0)
        if seen < cap:
            counts[block.document_id] = seen + 1
            kept.append(block)
    return kept


def _elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
