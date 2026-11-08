"""The routed entry point to retrieval: classify, plan, then search or answer from metadata.

Bibliographic, out-of-scope and ambiguous questions are answered here without a search.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field, replace
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
from myscholar.retrieval.selection import DocumentSelector


@dataclass(frozen=True, slots=True)
class RoutedResult:
    """What routing produced, and how it got there."""

    classification: Classification
    plan: Plan
    blocks: tuple[ContextBlock, ...] = ()
    bibliographic: BibliographicAnswer | None = None
    queries: tuple[SearchQuery, ...] = ()
    degraded: bool = False
    """Set when the answer is known to be lower quality than usual."""

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
        selector: DocumentSelector,
        mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> None:
        self._mode = mode
        self._selector = selector
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
            titles = [d.title for d in documents]
            self._classifier.set_titles(titles)
            # HyDE uses them to write in the corpus's vocabulary.
            self._transformer.set_titles(titles)
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
            if answer is not None:
                return RoutedResult(
                    classification=classification,
                    plan=plan,
                    bibliographic=answer,
                    timings_ms=timings,
                )
            # Nothing asked for a count, date, venue or list, so it is a content question.
            plan = replace(plan, use_sql=False)

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
        """Search with every query concurrently and fuse the results by reciprocal rank.

        A filtered search that returns nothing is retried without filters: an empty
        result more likely means a wrong filter than a missing answer.
        """
        # Document scope comes from the selector alone (see retrieval/selection.py).
        selection = await self._selector.select(
            queries[0].text if queries else "",
            titles=(),
            title_to_id=title_to_id,
            limit=plan.select_documents,
        )
        document_ids = selection.document_ids

        # Section kinds boost through an extra branch instead of filtering the base
        # search. As a filter they excluded abstracts and conclusions from
        # methodological questions and cost 9 points of MRR.
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

        async def run(text: str, filters: SearchFilters) -> tuple[ContextBlock, ...]:
            result = await self._retriever.retrieve(
                text,
                mode=self._mode,
                filters=filters if not filters.is_empty else None,
                limit=plan.context_blocks,
            )
            if not result.blocks and not filters.is_empty:
                result = await self._retriever.retrieve(
                    text, mode=self._mode, limit=plan.context_blocks
                )
            return result.blocks

        # Independent searches, run concurrently. A methodological question with the
        # section boost makes six.
        plans: list[tuple[str, str, SearchFilters, float]] = []
        for index, query in enumerate(queries):
            plans.append((f"{query.origin.value}:{index}", query.text, base, query.weight))
            if boosted is not None:
                plans.append(
                    (
                        f"{query.origin.value}:{index}:sections",
                        query.text,
                        boosted,
                        query.weight * 0.5,
                    )
                )

        results = await asyncio.gather(*(run(text, filters) for _, text, filters, _ in plans))

        branches: dict[str, list[str]] = {}
        block_by_key: dict[str, ContextBlock] = {}
        weights: dict[str, float] = {}
        for (name, _, _, weight), blocks in zip(plans, results, strict=True):
            keys: list[str] = []
            for block in blocks:
                # Keyed on a text digest because two spans in one section often
                # share a page range. blake2b, since hash() is salted per process.
                digest = hashlib.blake2b(block.text.encode(), digest_size=8).hexdigest()
                key = f"{block.section_id}:{digest}"
                block_by_key.setdefault(key, block)
                keys.append(key)
            branches[name] = keys
            weights[name] = weight

        if not branches:
            return []

        fused = reciprocal_rank_fusion(branches, weights=weights, limit=plan.context_blocks * 2)
        ordered: list[ContextBlock] = [block_by_key[key] for key, _ in fused if key in block_by_key]

        if plan.anchor_to_top_document:
            ordered = _anchor_to_top_document(ordered)

        if plan.max_chunks_per_document is not None:
            ordered = _cap_per_document(ordered, plan.max_chunks_per_document)

        # Relabel after fusion so markers run C1..Cn in the order the generator sees them.
        return [
            replace(b, label=f"C{i}") for i, b in enumerate(ordered[: plan.context_blocks], start=1)
        ]


def _anchor_to_top_document(blocks: list[ContextBlock]) -> list[ContextBlock]:
    """Move the best-matching document's passages to the front, keeping order within groups.

    A reorder instead of a filter, because the anchor is a guess when several papers
    share a topic; the alternatives stay in context behind it.
    """
    if not blocks:
        return blocks
    anchor = blocks[0].document_id
    return [b for b in blocks if b.document_id == anchor] + [
        b for b in blocks if b.document_id != anchor
    ]


def _cap_per_document(blocks: list[ContextBlock], cap: int) -> list[ContextBlock]:
    """Limit how many blocks one document contributes, so one paper cannot fill the context."""
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
