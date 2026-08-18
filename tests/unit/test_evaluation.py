"""The measuring instruments.

A metric that flatters is worse than no metric, because it is believed. These
tests exist for the arithmetic and the edge cases where a plausible
implementation quietly reports success — an unreachable judge counted as
faithful, a refusal counted as a clean answer, a denominator of zero rendered
as 100%.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from myscholar.adapters.fakes import FakeChatModel
from myscholar.domain.documents import ContextBlock
from myscholar.evaluation.baseline import NaivePipeline, contradicts, declines
from myscholar.evaluation.faithfulness import ClaimVerdict, FaithfulnessJudge, FaithfulnessResult


def _block(text: str) -> ContextBlock:
    return ContextBlock(
        label="C1",
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="A Paper",
        section_heading="Results",
        text=text,
        page_start=1,
        page_end=1,
    )


class TestDeclines:
    @pytest.mark.parametrize(
        "text",
        [
            "The excerpts do not mention wind turbines.",
            "There is no information about battery degradation here.",
            "I cannot determine that from the provided passages.",
            "This is not discussed in the supplied text.",
            "The provided text does not address quantum error correction.",
        ],
    )
    def test_a_textual_refusal_counts(self, text: str) -> None:
        """The naive pipeline has no refusal mechanism but its text declines
        anyway. Scoring that as a hallucination would overstate the difference
        the report exists to measure."""
        assert declines(text)

    @pytest.mark.parametrize(
        "text",
        [
            "He used NSGA-II with TOPSIS to select weighting factors.",
            "The framework reached 98.5% accuracy on the held-out recordings.",
            "Chaotic SVPWM spreads harmonics, which reduces EMI.",
        ],
    )
    def test_a_real_answer_does_not(self, text: str) -> None:
        assert not declines(text)

    def test_an_empty_answer_is_not_a_refusal(self) -> None:
        """It is a failure, and conflating the two would hide it."""
        assert not declines("")


class TestFaithfulnessArithmetic:
    def test_the_rate_counts_unsupported_over_judged(self) -> None:
        result = FaithfulnessResult(
            verdicts=(
                ClaimVerdict("a", supported=True),
                ClaimVerdict("b", supported=False),
                ClaimVerdict("c", supported=True),
                ClaimVerdict("d", supported=True),
            )
        )

        assert result.rate == 0.25

    def test_an_unjudged_claim_is_excluded_rather_than_assumed_good(self) -> None:
        """The loop's grounding check treats an unavailable model as supported,
        which is right for serving and wrong for measuring — it would turn
        every failed call into evidence of faithfulness."""
        result = FaithfulnessResult(
            verdicts=(
                ClaimVerdict("a", supported=False),
                ClaimVerdict("b", supported=None),
            )
        )

        assert result.judged == 1
        assert result.unjudged == 1
        assert result.rate == 1.0

    def test_nothing_judged_reports_absent_not_perfect(self) -> None:
        """Zero over zero rendered as 0% would read as a flawless answer."""
        assert FaithfulnessResult(verdicts=()).rate is None


class TestFaithfulnessJudge:
    async def test_citation_markers_are_stripped_before_judging(self) -> None:
        """ "[C3]" appears in no excerpt. Leaving markers in asks whether the
        context supports a string it cannot contain, so the configurations
        that cite would score worse for having citations at all."""
        model = FakeChatModel()
        judge = FaithfulnessJudge(model)

        await judge.score("He used NSGA-II [C1][C2].", (_block("NSGA-II was used."),))

        assert "[C1]" not in model.calls[0][0].content

    async def test_no_context_means_nothing_to_judge(self) -> None:
        model = FakeChatModel()

        result = await FaithfulnessJudge(model).score("He used NSGA-II.", ())

        assert result.verdicts == ()
        assert model.calls == []

    async def test_an_unreachable_judge_records_no_verdict(self) -> None:
        class _Broken(FakeChatModel):
            async def structured(self, *args: Any, **kwargs: Any) -> Any:
                msg = "judge unavailable"
                raise RuntimeError(msg)

        result = await FaithfulnessJudge(_Broken()).score(
            "He used NSGA-II.", (_block("NSGA-II was used."),)
        )

        assert result.judged == 0
        assert result.unjudged == 1
        assert result.rate is None

    async def test_non_factual_sentences_are_not_judged(self) -> None:
        """The judge reuses the verifier's definition of a claim rather than
        inventing a second one.

        That definition is deliberately inverted — everything counts as factual
        unless recognisably framing — so the two can never disagree about what
        needed a citation and what needed support.
        """
        model = FakeChatModel()

        await FaithfulnessJudge(model).score("Yes.", (_block("Something."),))

        assert model.calls == []


class TestNaivePipeline:
    async def test_the_naive_prompt_shape_is_preserved(self) -> None:
        """Comparing against a straw man proves nothing. The baseline sends
        the context and the question concatenated, with no system prompt and
        no instructions."""

        class _Retriever:
            async def retrieve(self, *args: Any, **kwargs: Any) -> Any:
                class _Result:
                    blocks = (_block("NSGA-II was used."),)
                    degraded = False

                return _Result()

        model = FakeChatModel(default="an answer")
        pipeline = NaivePipeline(retriever=_Retriever(), model=model)  # type: ignore[arg-type]

        await pipeline.answer("what was used?")

        prompt = model.calls[0][0].content
        assert prompt.startswith("<|content_start>")
        assert "<|content_end> what was used?" in prompt
        assert len(model.calls[0]) == 1, "the baseline sends no system message"


class TestContradiction:
    """Refusing is not the only right answer to a false premise."""

    @pytest.mark.parametrize(
        "text",
        [
            "Instead, it used NSGA-II to optimise the weighting factors.",
            "In fact, chaotic SVPWM reduces EMI by spreading harmonics.",
            "The excerpts do not conclude that current sensors outperform acoustic sensing.",
            "However, the paper emphasises acoustic signals as a viable alternative.",
        ],
    )
    def test_a_correction_counts(self, text: str) -> None:
        assert contradicts(text)

    @pytest.mark.parametrize(
        "text",
        [
            "He used NSGA-II with TOPSIS to select weighting factors.",
            "The framework reached 98.5% accuracy on held-out recordings.",
        ],
    )
    def test_an_ordinary_answer_does_not(self, text: str) -> None:
        assert not contradicts(text)

    def test_contradiction_is_not_refusal(self) -> None:
        """Kept separate so a good answer that happens to say "instead" is
        never counted toward over-refusal."""
        text = "Instead, it used NSGA-II to optimise the weighting factors."

        assert contradicts(text)
        assert not declines(text)
