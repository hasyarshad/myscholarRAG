"""Citation verification. Every check runs without a model."""

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.domain.documents import ContextBlock
from myscholar.generation.citations import (
    CitationIssue,
    drop_unsupported,
    is_factual,
    parse_markers,
    rejoin,
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
        text = "It cures cancer completely. It also ends all war."

        cleaned = drop_unsupported(text, verify(text, BLOCKS).issues)

        assert cleaned == ""


class TestVerificationProperties:
    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=40))
    def test_any_marker_outside_the_range_is_caught(self, index: int) -> None:
        text = f"The scheme reduces torque ripple by 18% [C{index}]."

        result = verify(text, BLOCKS)

        assert result.valid == (index <= len(BLOCKS))

    @settings(max_examples=100, deadline=None)
    @given(st.text(max_size=400))
    def test_verification_never_raises(self, text: str) -> None:
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


class TestATopicSentenceMayNameSomethingWithADigitInIt:
    """The leading topic-sentence exemption allows a digit inside a name, as in COVID-19."""

    def test_a_name_containing_a_digit_is_still_a_topic_sentence(self) -> None:
        opener = (
            "Muhammad Haseeb Arshad has conducted research related to COVID-19, "
            "focusing on content moderation on Reddit"
        )

        assert not is_factual(opener, leading=True), (
            "the sentence that answers the question was treated as an uncited claim"
        )

    def test_a_stated_figure_still_needs_its_citation(self) -> None:
        for claim in (
            "His framework has reached 99.64% accuracy on the held-out set",
            "He has published work in 2023 on predictive control",
            "The study has analysed 1.8 million Reddit posts",
        ):
            assert is_factual(claim, leading=True), f"unchecked figure: {claim}"

    def test_the_exemption_is_still_only_for_the_opening_sentence(self) -> None:
        later = "He has conducted research related to COVID-19"

        assert is_factual(later, leading=False)


class TestWhatSurvivesDroppingReadsAsProse:
    """Dropping a sentence must not leave a broken seam, such as an orphaned "In another work"."""

    def _issues(self, *sentences: str) -> list[CitationIssue]:
        return [CitationIssue(kind="unsupported", detail="x", sentence=s) for s in sentences]

    def test_a_backward_reference_left_at_the_front_is_lifted_off(self) -> None:
        text = (
            "Weighting factors are tuned by optimisation [C1]. "
            "In another work, MOPSO with Vikor was applied [C2]."
        )

        kept = drop_unsupported(
            text, self._issues("Weighting factors are tuned by optimisation [C1].")
        )

        assert kept == "MOPSO with Vikor was applied [C2]."

    def test_a_backward_reference_further_in_still_has_something_to_refer_to(self) -> None:
        text = "First [C1]. Second [C2]. In addition, third [C3]."

        kept = drop_unsupported(text, self._issues("Second [C2]."))

        assert kept == "First [C1]. In addition, third [C3]."

    def test_doubled_terminal_punctuation_is_collapsed(self) -> None:
        text = "A claim [C1]. The current exceeded its rated value [C4].."

        kept = drop_unsupported(text, self._issues("A claim [C1]."))

        assert kept == "The current exceeded its rated value [C4]."

    def test_dropping_everything_returns_nothing_rather_than_punctuation(self) -> None:
        assert drop_unsupported("Only one [C1].", self._issues("Only one [C1].")) == ""


class TestBothDropPathsTidyTheirSeams:
    """``rejoin`` is shared by citation salvage and the grounding drop in the reasoning loop."""

    def test_an_orphaned_opener_is_lifted_off(self) -> None:
        kept = ["In another work, MOPSO with Vikor was applied [C2].", "It reduces ripple [C3]."]

        assert rejoin(kept) == "MOPSO with Vikor was applied [C2]. It reduces ripple [C3]."

    def test_doubled_terminal_punctuation_collapses(self) -> None:
        assert rejoin(["The value exceeded its rating [C4].."]) == (
            "The value exceeded its rating [C4]."
        )

    def test_nothing_surviving_returns_nothing(self) -> None:
        assert rejoin([]) == ""

    def test_a_backward_reference_further_in_is_left_alone(self) -> None:
        kept = ["First [C1].", "In addition, second [C2]."]

        assert rejoin(kept) == "First [C1]. In addition, second [C2]."
