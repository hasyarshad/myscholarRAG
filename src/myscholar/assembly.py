"""Building the query pipeline, in one place.

``container.py`` chooses the *adapters* — which model, which embedder, which
store. This chooses the *arrangement*: which retriever, wired to which router,
wired to which engine, with which budgets.

It exists because the arrangement was written out five times — the API, the
CLI, and three evaluation scripts — and that is worse than duplication. An
evaluation script that wires the pipeline slightly differently is not measuring
the system that ships, and nothing would report the difference: the numbers
would simply describe something else. The drift had already begun, with
``short_name`` reaching the API's engine and not the CLI's.

The rule this encodes: **the evaluation must assemble the system exactly as
production does.** The cheapest way to guarantee that is to give them one
function to call.
"""

from __future__ import annotations

from myscholar.adapters.persistence.repository import PostgresStore
from myscholar.config import Settings
from myscholar.container import Container
from myscholar.generation.answerer import Answerer
from myscholar.query.classifier import QueryClassifier
from myscholar.query.planner import QueryPlanner
from myscholar.query.router import QueryRouter
from myscholar.query.transforms import QueryTransformer
from myscholar.reasoning.engine import Engine
from myscholar.reasoning.graph import Budget
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode
from myscholar.retrieval.selection import DocumentSelector


def build_retriever(
    settings: Settings, container: Container, store: PostgresStore
) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        embedder=container.embedder,
        top_k=settings.retrieval.top_k,
        context_blocks=settings.retrieval.context_blocks,
        window=settings.retrieval.window,
        dense_weight=settings.retrieval.dense_weight,
        lexical_weight=settings.retrieval.lexical_weight,
        rrf_k=settings.retrieval.rrf_k,
        max_span_chunks=settings.retrieval.max_span_chunks,
    )


def build_router(
    settings: Settings,
    container: Container,
    store: PostgresStore,
    *,
    retriever: HybridRetriever | None = None,
    titles: list[str] | None = None,
) -> QueryRouter:
    """The routed entry point to retrieval.

    ``titles`` seeds entity linking and hypothetical-passage generation. The
    router loads them itself on first use, so passing them is an optimisation
    for callers that already hold the document list.
    """
    return QueryRouter(
        classifier=QueryClassifier(
            container.utility, titles=titles, person_name=settings.person.name
        ),
        planner=QueryPlanner(
            top_k=settings.retrieval.top_k,
            context_blocks=settings.retrieval.context_blocks,
            person_name=settings.person.name,
        ),
        transformer=QueryTransformer(container.utility, titles=titles),
        retriever=retriever or build_retriever(settings, container, store),
        store=store,
        selector=DocumentSelector(store=store, embedder=container.embedder),
        mode=RetrievalMode.HYBRID,
    )


def build_answerer(settings: Settings, container: Container) -> Answerer:
    return Answerer(
        generator=container.generator,
        utility=container.utility,
        person_name=settings.person.name,
        short_name=settings.person.short_name,
    )


def build_budget(settings: Settings) -> Budget:
    return Budget(
        max_steps=settings.reasoning.max_steps,
        max_retrieval_rounds=settings.reasoning.max_retrieval_rounds,
        deadline_seconds=settings.reasoning.deadline_seconds,
    )


def build_engine(
    settings: Settings,
    container: Container,
    store: PostgresStore,
    *,
    router: QueryRouter | None = None,
) -> Engine:
    return Engine(
        router=router or build_router(settings, container, store),
        answerer=build_answerer(settings, container),
        budget=build_budget(settings),
        short_name=settings.person.short_name,
    )
