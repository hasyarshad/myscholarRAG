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


@dataclass(frozen=True, slots=True)
class RoutedResult:
    """What routing produced, and how it got there."""

    classification: Classification
    plan: Plan
    blocks: tuple[ContextBlock, ...] = ()
    bibliographic: BibliographicAnswer | None = None
    queries: tuple[SearchQuery, ...] = ()
    degraded: bool = False
    """A component was unavailable — currently, reranking. Reported rather than
    hidden, so a lower-quality answer is identifiable as one."""

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
        blocks, degraded = await self._search(queries, plan, title_to_id)
        timings["retrieve"] = _elapsed(started)

        return RoutedResult(
            classification=classification,
            plan=plan,
            blocks=tuple(blocks),
            queries=tuple(queries),
            degraded=degraded,
            timings_ms=timings,
        )

    async def _select_documents(self, queries: list[SearchQuery], plan: Plan) -> frozenset[UUID]:
        """Documents whose summaries match, for classes that need breadth.

        Empty for every other class, and empty when no summaries exist — in
        which case the search behaves exactly as it did before, which is what
        makes this safe to add to a corpus that has not been summarised.
        """
        if plan.max_chunks_per_document is None or not queries:
            return frozenset()
        try:
            embedding = await self._retriever.embed_query(queries[0].text)
            matches = await self._store.search_summaries(
                embedding, model_id=self._retriever.model_id, limit=plan.select_documents
            )
        except Exception:  # noqa: BLE001 - selection is an optimisation, not a requirement
            return frozenset()
        return frozenset(document_id for document_id, _ in matches)

    async def _search(
        self,
        queries: list[SearchQuery],
        plan: Plan,
        title_to_id: dict[str, UUID],
    ) -> tuple[list[ContextBlock], bool]:
        """Retrieve for every query and fuse the results.

        A filtered branch is always accompanied by an unfiltered one. An
        over-confident entity link would otherwise return nothing at all, and
        an imperfect result beats an empty one.
        """
        document_ids = frozenset(
            title_to_id[title] for title in plan.filter_document_titles if title in title_to_id
        )

        # Cross-paper questions choose their documents before their passages.
        #
        # Chunk search answers "which passage says X" and cannot answer "which
        # paper is about X", because a paper rarely describes itself in a
        # reader's words. Asked about machine learning, the six papers that
        # apply it — calling their methods LFBLSTM, AFL-1DCNN, feature fusion —
        # lose every slot to the one survey that discusses the phrase itself.
        # Five transform combinations were tried and none of them fixed it: the
        # gap is between the corpus's vocabulary and the question's, not
        # between two phrasings.
        #
        # Summaries are written in a reader's vocabulary, so selecting on them
        # first decides breadth by what each paper *is*.
        # Only when the entity linker resolved nothing. Union-ing the two would
        # *widen* an explicit filter — a question that named one paper would be
        # searched across nine — which is the opposite of what naming a paper
        # is for. The linker is the more specific signal and wins.
        if not document_ids and (summary_ids := await self._select_documents(queries, plan)):
            document_ids = summary_ids

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

        degraded = False

        async def run(text: str, filters: SearchFilters) -> tuple[ContextBlock, ...]:
            nonlocal degraded
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
            degraded = degraded or result.degraded
            return result.blocks

        # Every branch concurrently. A methodological question produces three
        # queries and, with the section boost, six searches — each embedding,
        # querying twice and reranking. Run in sequence those dominate latency;
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
            return [], degraded

        fused = reciprocal_rank_fusion(branches, weights=weights, limit=plan.rerank_k * 2)
        ordered: list[ContextBlock] = [block_by_key[key] for key, _ in fused if key in block_by_key]

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
            replace(b, label=f"C{i}") for i, b in enumerate(ordered[: plan.rerank_k], start=1)
        ], degraded


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
