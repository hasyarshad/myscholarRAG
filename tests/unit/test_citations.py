"""Citation verification.

These are the tests that matter most in the project. Every check here runs
without a model, so if one has a hole, a hallucination passes silently.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.domain.documents import ContextBlock
from myscholar.generation.citations import (
    drop_unsupported,
    is_factual,
    parse_markers,
    strip_markers,
    verify,
)


def block(label: str, text: str) -> ContextBlock:
    return ContextBlock(
        label=label,
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="A Paper About Predictive Control",
        section_heading="IV. Results",
        text=text,
        page_start=4,
        page_end=5,
    )


BLOCKS = [
    block("C1", "The proposed scheme reduces torque ripple by 18% compared with the baseline."),
    block("C2", "Weighting factors were selected using NSGA-II with TOPSIS decision making."),
]


class TestParseMarkers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A claim [C1].", ["C1"]),
            ("Two sources [C1][C2].", ["C1", "C2"]),
            ("Combined [C1, C2].", ["C1", "C2"]),
            ("Spaced [ C3 ].", ["C3"]),
            ("Lowercase [c1].", ["C1"]),
            ("Repeated [C1] and again [C1].", ["C1"]),
            ("No markers here.", []),
        ],
    )
    def test_marker_forms(self, text: str, expected: list[str]) -> None:
        assert parse_markers(text) == expected

    def test_brackets_that_are_not_markers_are_ignored(self) -> None:
        assert parse_markers("An array [0] and a note [see below].") == []


class TestIsFactual:
    @pytest.mark.parametrize(
        "sentence",
        [
            "The scheme reduces torque ripple by 18% compared with the baseline.",
            "Weighting factors were selected using an evolutionary algorithm.",
            "The authors report improved dynamic performance under load.",
            "This approach outperforms conventional field oriented control.",
        ],
    )
    def test_assertions_need_a_citation(self, sentence: str) -> None:
        assert is_factual(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "I don't have anything on that topic in these papers.",
            "There is no discussion of wind turbines in the corpus.",
            "In summary:",
            "Would you like more detail on any of these?",
            "However.",
        ],
    )
    def test_framing_and_refusals_do_not(self, sentence: str) -> None:
        assert not is_factual(sentence)

    def test_a_sentence_of_only_markers_is_not_factual(self) -> None:
        assert not is_factual("[C1][C2]")


class TestVerify:
    def test_a_well_cited_answer_passes(self) -> None:
        text = "The scheme reduces torque ripple by 18% [C1]. Factors came from NSGA-II [C2]."

        result = verify(text, BLOCKS)

        assert result.valid
        assert [c.label for c in result.citations] == ["C1", "C2"]

    def test_a_dangling_marker_fails(self) -> None:
        """The most common hallucination: the model invents a source.

        Catching it costs a set lookup, which is why this check exists rather
        than a model being asked whether its citations are real.
        """
        result = verify("The scheme reduces torque ripple by 18% [C7].", BLOCKS)

        assert not result.valid
        assert result.dangling
        assert result.dangling[0].detail == "C7"

    def test_an_uncited_factual_claim_fails(self) -> None:
        text = "The scheme reduces torque ripple by 18% [C1]. It also halves switching losses."

        result = verify(text, BLOCKS)

        assert not result.valid
        assert "switching losses" in result.uncited[0].sentence

    def test_a_fabricated_quotation_fails(self) -> None:
        """A quote must be checkable against what the author actually wrote."""
        text = 'The paper states "the method eliminates all torque ripple entirely" [C1].'

        result = verify(text, BLOCKS)

        assert not result.valid
        assert any(i.kind == "unquoted" for i in result.issues)

    def test_a_real_quotation_passes(self) -> None:
        text = 'The paper reports "reduces torque ripple by 18% compared with the baseline" [C1].'

        result = verify(text, BLOCKS)

        assert result.valid

    def test_a_quotation_differing_only_in_whitespace_passes(self) -> None:
        text = 'It notes "reduces  torque   ripple by 18%\ncompared with the baseline" [C1].'

        assert verify(text, BLOCKS).valid

    def test_a_refusal_needs_no_citations(self) -> None:
        text = "I don't have anything on wind turbine control in these papers."

        assert verify(text, BLOCKS).valid

    def test_an_answer_with_no_blocks_and_no_claims_is_valid(self) -> None:
        assert verify("I can't answer that from these papers.", []).valid

    def test_a_claim_with_no_blocks_at_all_fails(self) -> None:
        assert not verify("The scheme reduces torque ripple by 18%.", []).valid

    def test_the_summary_describes_what_to_fix(self) -> None:
        """Fed back into a regeneration, so it has to name the problem."""
        result = verify("Torque ripple falls by 18% [C9]. It also improves efficiency.", BLOCKS)

        summary = result.summary()
        assert "C9" in summary
        assert "no citation" in summary or "efficiency" in summary


class TestStripAndDrop:
    def test_markers_are_removed_for_display(self) -> None:
        assert strip_markers("A claim [C1] and another [C2].") == "A claim and another."

    def test_unsupported_sentences_are_dropped(self) -> None:
        text = "Torque ripple falls by 18% [C1]. It also cures cancer completely."

        result = verify(text, BLOCKS)
        cleaned = drop_unsupported(text, result.issues)

        assert "cures cancer" not in cleaned
        assert "18%" in cleaned

    def test_dropping_everything_leaves_an_empty_string(self) -> None:
        """The caller must then refuse rather than return nothing."""
        text = "It cures cancer completely. It also ends all war."

        cleaned = drop_unsupported(text, verify(text, BLOCKS).issues)

        assert cleaned == ""


class TestVerificationProperties:
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=40))
    def test_any_marker_outside_the_range_is_caught(self, index: int) -> None:
        """The property the whole mechanism rests on.

        Whatever the model emits, a label with no corresponding block must be
        reported. No exceptions, no thresholds.
        """
        text = f"The scheme reduces torque ripple by 18% [C{index}]."

        result = verify(text, BLOCKS)

        assert result.valid == (index <= len(BLOCKS))

    @settings(max_examples=100, deadline=None)
    @given(st.text(max_size=400))
    def test_verification_never_raises(self, text: str) -> None:
        """Model output is arbitrary; the verifier must survive all of it."""
        verify(text, BLOCKS)

    @settings(max_examples=60, deadline=None)
    @given(st.lists(st.integers(1, 2), min_size=1, max_size=8))
    def test_valid_markers_always_resolve_to_citations(self, indices: list[int]) -> None:
        text = " ".join(f"The scheme reduces torque ripple by 18% [C{i}]." for i in indices)

        result = verify(text, BLOCKS)

        assert result.valid
        assert {c.label for c in result.citations} == {f"C{i}" for i in indices}

    @settings(max_examples=60, deadline=None)
    @given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200))
    def test_stripping_markers_never_leaves_one(self, text: str) -> None:
        assert not parse_markers(strip_markers(text))
