"""Classify, plan, then execute — the routed entry point to retrieval.

This is what the API and the CLI call. It owns the decision of whether to search
at all, which is the decision that removes most of a public bot's hallucination
surface.
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
    """Set when the answer is known to be lower quality than usual, so the
    interface can say so rather than presenting it as equivalent."""

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
            # The transformer needs them too: an ungrounded hypothetical
            # passage invents a research area the corpus does not contain, and
            # then retrieves against it.
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
            if answer is None:
                # Classified bibliographic, but nothing in it asks for a count,
                # a date, a venue or a list — so it is a question about content
                # wearing a collection question's clothes. Retrieval answers it,
                # or refuses on the evidence.
                plan = replace(plan, use_sql=False)
            if answer is not None:
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
        # Scope is decided in one place, by one component, before any passage
        # is fetched. See retrieval/selection.py — this used to be three
        # partial policies in three modules that interacted by accident.
        selection = await self._selector.select(
            queries[0].text if queries else "",
            titles=(),
            title_to_id=title_to_id,
            limit=plan.select_documents,
        )
        document_ids = selection.document_ids

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

        async def run(text: str, filters: SearchFilters) -> tuple[ContextBlock, ...]:
            result = await self._retriever.retrieve(
                text,
                mode=self._mode,
                filters=filters if not filters.is_empty else None,
                limit=plan.context_blocks,
            )
            # An empty filtered result means the filter was wrong, not that the
            # corpus lacks an answer.
            if not result.blocks and not filters.is_empty:
                result = await self._retriever.retrieve(
                    text, mode=self._mode, limit=plan.context_blocks
                )
            return result.blocks

        # Every branch concurrently. A methodological question produces three
        # queries and, with the section boost, six searches — each embedding,
        # querying twice. Run in sequence those dominate latency;
        # they are independent, so there is no reason to.
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
                # Keyed on the text digest, not the page range. Two distinct
                # spans in one section frequently share a page range, and keying
                # on that made them collide — the second was silently dropped,
                # losing real evidence from the fusion.
                #
                # A digest rather than hash(): the built-in is salted per
                # process, so the same block would key differently between runs
                # and make results irreproducible.
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

        # Relabel after fusion so citation markers run C1..Cn in the order the
        # generator will see them.
        #
        # `replace` rather than rebuilding field by field. The explicit version
        # dropped every field it did not list, so adding `document_kind` to
        # ContextBlock left it populated by the retriever and blank by the time
        # anything downstream saw it — silently, since a blank string is a
        # valid value. Only the label needs to change here.
        return [
            replace(b, label=f"C{i}") for i, b in enumerate(ordered[: plan.context_blocks], start=1)
        ]


def _anchor_to_top_document(blocks: list[ContextBlock]) -> list[ContextBlock]:
    """Move the best-matching document's passages to the front.

    A stable partition, so relative order is untouched within each group and the
    other documents survive behind the anchor rather than being removed. The
    reason not to filter is that the anchor is a guess: asked about "the
    FCS-MPTC paper" when three papers are about FCS-MPTC, the top block is the
    best evidence available for which one is meant, and it is still only
    evidence. Reordering makes a wrong guess recoverable — the alternatives are
    a few blocks further down — where a filter would make it final.

    See Plan.anchor_to_top_document for the measurement.
    """
    if not blocks:
        return blocks
    anchor = blocks[0].document_id
    return [b for b in blocks if b.document_id == anchor] + [
        b for b in blocks if b.document_id != anchor
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
