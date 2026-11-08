"""The fakes behave enough like the real thing to be worth testing against."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from myscholar.adapters.fakes import FakeChatModel, HashEmbedder
from myscholar.adapters.fakes.store import InMemoryVectorStore, _Record
from myscholar.ports.llm import Message, Role
from myscholar.ports.store import SearchFilters


class TestFakeChatModel:
    async def test_returns_default_when_nothing_matches(self) -> None:
        model = FakeChatModel(default="nothing here")

        result = await model.complete([Message(role=Role.USER, content="anything")])

        assert result == "nothing here"

    async def test_matches_on_prompt_substring(self) -> None:
        model = FakeChatModel({"harmonic": "THD improves by 3.1%"})

        result = await model.complete(
            [Message(role=Role.USER, content="Tell me about HARMONIC distortion")]
        )

        assert result == "THD improves by 3.1%"

    async def test_stream_reassembles_to_the_same_text(self) -> None:
        model = FakeChatModel(default="one two three")

        chunks = [c async for c in model.stream([Message(role=Role.USER, content="q")])]

        assert "".join(chunks).strip() == "one two three"
        assert len(chunks) > 1, "streaming should emit more than one chunk"

    async def test_structured_without_a_queued_response_raises(self) -> None:
        from pydantic import BaseModel

        class Verdict(BaseModel):
            ok: bool

        model = FakeChatModel()

        with pytest.raises(LookupError, match="structured_responses"):
            await model.structured([Message(role=Role.USER, content="q")], Verdict)


class TestHashEmbedder:
    async def test_dimension_matches_declaration(self) -> None:
        embedder = HashEmbedder(dim=768)

        assert len(await embedder.embed_query("anything")) == embedder.dim

    async def test_indexing_and_querying_agree(self) -> None:
        embedder = HashEmbedder()
        text = "model predictive control of induction motors"

        [document_vector] = await embedder.embed_documents([text])
        query_vector = await embedder.embed_query(text)

        assert document_vector == query_vector

    async def test_empty_text_stays_normalised(self) -> None:
        embedder = HashEmbedder()

        vector = await embedder.embed_query("")

        assert sum(v * v for v in vector) == pytest.approx(1.0)

    @given(st.text(min_size=1, max_size=200))
    async def test_output_is_always_unit_length(self, text: str) -> None:
        embedder = HashEmbedder(dim=64)

        vector = await embedder.embed_query(text)

        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-9)


class TestInMemoryVectorStore:
    @staticmethod
    async def _populated() -> tuple[InMemoryVectorStore, HashEmbedder, UUID]:
        embedder = HashEmbedder(dim=128)
        store = InMemoryVectorStore(model_id=embedder.model_id)
        section = uuid4()
        texts = [
            "finite control set model predictive current control of PMSM drives",
            "the FCS-MPCC weighting factor is tuned using NSGA-II",
            "crowdsourced content moderation on Reddit during the pandemic",
        ]
        for ordinal, text in enumerate(texts):
            [vector] = await embedder.embed_documents([text])
            store.add(
                _Record(
                    chunk_id=uuid4(),
                    document_id=uuid4(),
                    section_id=section,
                    ordinal_in_section=ordinal,
                    text=text,
                    embedding=vector,
                    year=2023,
                )
            )
        return store, embedder, section

    async def test_lexical_search_finds_rare_identifiers(self) -> None:
        store, _, _ = await self._populated()

        hits = await store.search_lexical("NSGA-II", limit=5)

        assert hits
        assert "NSGA-II" in hits[0].text

    async def test_dense_search_returns_results_ranked_by_similarity(self) -> None:
        store, embedder, _ = await self._populated()

        hits = await store.search_dense(
            await embedder.embed_query("content moderation Reddit"), limit=3
        )

        assert "Reddit" in hits[0].text
        assert hits[0].score >= hits[-1].score

    async def test_year_filter_excludes_non_matching(self) -> None:
        store, embedder, _ = await self._populated()

        hits = await store.search_dense(
            await embedder.embed_query("motor"),
            limit=5,
            filters=SearchFilters(years=frozenset({1999})),
        )

        assert hits == []

    async def test_fetch_span_is_ordered_and_bounded(self) -> None:
        store, _, section_id = await self._populated()

        span = await store.fetch_span(section_id, 0, 1)

        assert [h.ordinal_in_section for h in span] == [0, 1]

    async def test_indexed_model_ids_reports_what_is_present(self) -> None:
        store, embedder, _ = await self._populated()

        assert await store.indexed_model_ids() == {embedder.model_id}
