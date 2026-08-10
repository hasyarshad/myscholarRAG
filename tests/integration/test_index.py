"""Checks against the real index built from the real corpus.

Marked ``integration``. Run after ingestion:

    make up && uv run myscholar ingest run && uv run pytest -m integration

These verify properties that only hold once actual papers are indexed —
retrieval finding the right document, the ligature repair having worked, and
the structural guards on the schema actually firing.
"""

from __future__ import annotations

import os

import pytest

from myscholar.adapters.persistence.repository import (
    IndexMismatchError,
    PostgresStore,
    create_engine,
)
from myscholar.ports.embedding import Embedder

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "MYSCHOLAR_DATABASE__URL",
    "postgresql+asyncpg://haseeb@127.0.0.1:5433/myscholar",
)
EMBEDDING_MODEL = os.environ.get("MYSCHOLAR_EMBEDDING__MODEL_ID", "BAAI/bge-base-en-v1.5")


@pytest.fixture
async def store() -> PostgresStore:
    store = PostgresStore(create_engine(DATABASE_URL))
    if not await store.health():
        pytest.skip(f"no database at {DATABASE_URL} (run: make db-up)")
    if (await store.counts())["chunks"] == 0:
        pytest.skip("index is empty (run: myscholar ingest run)")
    return store


@pytest.fixture
async def embedder() -> Embedder:
    from myscholar.adapters.embedding.torch_embedder import TorchEmbedder

    return TorchEmbedder(EMBEDDING_MODEL, device="cuda")


class TestIndexIntegrity:
    async def test_every_chunk_has_an_embedding(self, store: PostgresStore) -> None:
        """A chunk without a vector is invisible to dense search, silently."""
        counts = await store.counts()

        assert counts["embeddings"] == counts["chunks"]

    async def test_exactly_one_embedding_model_is_indexed(self, store: PostgresStore) -> None:
        assert len(await store.indexed_model_ids()) == 1

    async def test_a_mismatched_embedder_is_rejected(self, store: PostgresStore) -> None:
        """The structural guard against the defect in the previous code.

        Searching an index built by a different model returns plausible,
        meaningless results. This must fail loudly rather than quietly.
        """
        with pytest.raises(IndexMismatchError, match="Re-index"):
            await store.assert_embedder_matches("some-other/embedding-model")

    async def test_the_configured_embedder_is_accepted(self, store: PostgresStore) -> None:
        await store.assert_embedder_matches(EMBEDDING_MODEL)

    async def test_references_were_excluded(self, store: PostgresStore) -> None:
        """Roughly a quarter of each paper, none of it answerable."""
        kinds = await store.section_kind_counts()

        assert kinds.get("references", 0) == 0

    async def test_most_sections_were_classified(self, store: PostgresStore) -> None:
        """Section filtering is worthless if everything lands in OTHER."""
        kinds = await store.section_kind_counts()
        total = sum(kinds.values())

        assert kinds.get("other", 0) / total < 0.25


class TestExtractionQuality:
    async def test_ligatures_survived_extraction(self, store: PostgresStore) -> None:
        """The defect that motivated switching Docling's PDF backend.

        The default parser turned "flux" into " ux". On a corpus about motor
        flux control that silently breaks retrieval for a central term, so it
        is worth asserting on the real index rather than trusting the config.
        """
        from sqlalchemy import func, select

        from myscholar.adapters.persistence.orm import Chunk

        async with store.session() as session:
            intact = await session.scalar(
                select(func.count(Chunk.id)).where(Chunk.text.ilike("%flux%"))
            )
            corrupted = await session.scalar(
                select(func.count(Chunk.id)).where(Chunk.text.op("~")(r"\m ux\M"))
            )

        assert intact and intact > 0, "no chunk mentions flux — extraction is suspect"
        assert not corrupted, f"{corrupted} chunks contain ' ux', so ligatures were dropped"

    async def test_no_undecoded_formula_placeholders_leaked_in(self, store: PostgresStore) -> None:
        from sqlalchemy import func, select

        from myscholar.adapters.persistence.orm import Chunk

        async with store.session() as session:
            leaked = await session.scalar(
                select(func.count(Chunk.id)).where(Chunk.text.like("%formula-not-decoded%"))
            )

        assert not leaked


class TestRetrieval:
    @pytest.mark.parametrize(
        ("query", "expected_term"),
        [
            ("how are weighting factors tuned for predictive torque control?", "weighting"),
            ("what technique detects stator winding inter-turn faults?", "fault"),
            ("how was content moderation studied on Reddit?", "moderation"),
        ],
    )
    async def test_dense_search_finds_topically_relevant_chunks(
        self, store: PostgresStore, embedder: Embedder, query: str, expected_term: str
    ) -> None:
        vector = await embedder.embed_query(query)

        hits = await store.search_dense(vector, limit=5)

        assert hits, "dense search returned nothing"
        assert any(expected_term in hit.text.lower() for hit in hits[:5])

    @pytest.mark.parametrize("term", ["NSGA-II", "TOPSIS", "MPTC"])
    async def test_lexical_search_finds_rare_identifiers(
        self, store: PostgresStore, term: str
    ) -> None:
        """The case dense embeddings are worst at, and why hybrid search exists."""
        hits = await store.search_lexical(term, limit=5)

        assert hits, f"lexical search found nothing for {term!r}"

    async def test_dense_scores_are_ordered(self, store: PostgresStore, embedder: Embedder) -> None:
        vector = await embedder.embed_query("model predictive control")

        hits = await store.search_dense(vector, limit=10)

        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    async def test_span_fetch_stays_within_one_section(self, store: PostgresStore) -> None:
        """Parent expansion must never stitch text across sections or documents.

        The previous implementation expanded by global row id with no filter,
        so a window near a boundary could merge two papers into one citation.
        """
        from sqlalchemy import select

        from myscholar.adapters.persistence.orm import Chunk

        async with store.session() as session:
            chunk = (await session.execute(select(Chunk).limit(1))).scalar_one()

        span = await store.fetch_span(chunk.section_id, 0, 9999)

        assert span
        assert {hit.section_id for hit in span} == {chunk.section_id}
        ordinals = [hit.ordinal_in_section for hit in span]
        assert ordinals == sorted(ordinals)
        assert ordinals == list(range(len(ordinals))), "ordinals must be dense"
