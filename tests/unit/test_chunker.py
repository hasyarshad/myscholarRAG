"""Chunking, including property tests for the invariants retrieval depends on."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.domain.enums import ChunkKind
from myscholar.ingest.chunker import (
    ChunkingPolicy,
    build_context_header,
    chunk_section,
    estimate_tokens,
    split_sentences,
    strip_mathematics,
)

POLICY = ChunkingPolicy()


def _prose(text: str, page: int = 1) -> list[tuple[ChunkKind, str, int]]:
    return [(ChunkKind.PROSE, text, page)]


class TestSplitSentences:
    def test_splits_on_terminators(self) -> None:
        result = split_sentences("The motor runs. The flux is stable. Torque ripple falls.")

        assert len(result) == 3

    @pytest.mark.parametrize(
        "text",
        [
            "As shown in Fig. 3 the torque ripple falls.",
            "The results in Eq. 7 confirm this.",
            "Reported by H. Arshad and colleagues in 2023.",
            "See Sec. 4 for the derivation.",
        ],
    )
    def test_scientific_abbreviations_do_not_split(self, text: str) -> None:
        assert len(split_sentences(text)) == 1

    def test_empty_input_yields_nothing(self) -> None:
        assert split_sentences("   ") == []


class TestStripMathematics:
    def test_removes_display_latex(self) -> None:
        text = r"The cost function $$g = |T_{ref} - T| + \lambda |\psi|$$ is minimised."

        result = strip_mathematics(text)

        assert "lambda" not in result
        assert "cost function" in result
        assert "minimised" in result

    def test_removes_undecoded_formula_placeholders(self) -> None:
        text = "The relation is <!-- formula-not-decoded --> where T is one."

        result = strip_mathematics(text)

        assert "formula-not-decoded" not in result
        assert "where T is one" in result

    def test_keeps_surrounding_prose_intact(self) -> None:
        text = r"where \tau_s is the stator time constant of the machine"

        assert "stator time constant" in strip_mathematics(text)


class TestChunkSection:
    def test_short_section_becomes_one_chunk(self) -> None:
        chunks = chunk_section(_prose("A short methods section. Only two sentences here."))

        assert len(chunks) == 1

    def test_long_section_splits_into_several(self) -> None:
        text = " ".join(
            f"Sentence number {i} describes the control strategy in some detail." for i in range(80)
        )

        chunks = chunk_section(_prose(text))

        assert len(chunks) > 1

    def test_chunks_stay_near_the_target_size(self) -> None:
        text = " ".join(
            f"Sentence {i} about model predictive control of induction motors." for i in range(120)
        )

        chunks = chunk_section(_prose(text))

        assert all(c.token_count <= POLICY.max_tokens for c in chunks)

    def test_ordinals_are_contiguous_from_zero(self) -> None:
        text = " ".join(f"Sentence {i} with content." for i in range(100))

        chunks = chunk_section(_prose(text))

        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_tables_are_never_split(self) -> None:
        table = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(200))

        chunks = chunk_section([(ChunkKind.TABLE, table, 4)])

        assert len(chunks) == 1
        assert chunks[0].kind is ChunkKind.TABLE

    def test_equations_are_kept_as_their_own_chunks(self) -> None:
        blocks = [
            (ChunkKind.PROSE, "The cost function is defined below.", 3),
            (ChunkKind.EQUATION, r"$$g = |T_{ref} - T|$$", 3),
        ]

        kinds = [c.kind for c in chunk_section(blocks)]

        assert ChunkKind.EQUATION in kinds

    def test_tiny_trailing_chunk_is_merged_away(self) -> None:
        text = " ".join(f"Sentence {i} describing the approach." for i in range(60)) + " Short."

        chunks = chunk_section(_prose(text))

        prose = [c for c in chunks if c.kind is ChunkKind.PROSE]
        assert all(c.token_count >= POLICY.min_tokens for c in prose[:-1])

    def test_empty_blocks_are_dropped(self) -> None:
        assert chunk_section([(ChunkKind.PROSE, "   ", 1)]) == []

    def test_a_single_enormous_sentence_is_split(self) -> None:
        monster = "word " * 4000

        chunks = chunk_section(_prose(monster))

        assert len(chunks) > 1
        assert all(c.token_count <= POLICY.max_tokens for c in chunks)

    def test_page_number_is_carried_through(self) -> None:
        chunks = chunk_section(_prose("Some content on page seven.", page=7))

        assert chunks[0].page == 7


class TestChunkingProperties:
    @settings(max_examples=60, deadline=None)
    @given(
        st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                min_size=5,
                max_size=200,
            ),
            min_size=1,
            max_size=40,
        )
    )
    def test_no_chunk_ever_exceeds_the_ceiling(self, sentences: list[str]) -> None:
        text = ". ".join(sentences)

        chunks = chunk_section(_prose(text))

        assert all(c.token_count <= POLICY.max_tokens for c in chunks)

    @settings(max_examples=60, deadline=None)
    @given(st.text(min_size=1, max_size=3000))
    def test_ordinals_are_always_contiguous(self, text: str) -> None:
        chunks = chunk_section(_prose(text))

        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    @settings(max_examples=40, deadline=None)
    @given(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Zs")),
                min_size=20,
                max_size=150,
            ),
            min_size=3,
            max_size=25,
        )
    )
    def test_no_content_word_is_lost(self, sentences: list[str]) -> None:
        text = ". ".join(s.strip() for s in sentences if s.strip())
        if not text.strip():
            return

        chunks = chunk_section(_prose(text))
        combined = " ".join(c.text for c in chunks)

        for word in {w for w in text.split() if len(w) > 3}:
            assert word in combined


class TestContextHeader:
    def test_includes_paper_venue_year_and_section(self) -> None:
        header = build_context_header(
            title="Weighting Factors Optimization of MPTC",
            year=2021,
            venue="IEEE Access",
            section_heading="VII. EXPERIMENTAL RESULTS",
            section_kind="results",
        )

        assert "Weighting Factors Optimization" in header
        assert "IEEE Access" in header
        assert "2021" in header
        assert "EXPERIMENTAL RESULTS" in header

    def test_degrades_gracefully_without_metadata(self) -> None:
        header = build_context_header(
            title="Some Paper",
            year=None,
            venue=None,
            section_heading="",
            section_kind="methods",
        )

        assert "Some Paper" in header
        assert "methods" in header


class TestEstimateTokens:
    def test_never_returns_zero(self) -> None:
        assert estimate_tokens("") == 1

    def test_scales_with_length(self) -> None:
        assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)
