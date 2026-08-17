"""Questions that assume something untrue.

The hardest family in the evaluation and the one the system scored worst on.
A false premise is not a retrieval failure and retrieving harder cannot fix it:
the right passages are already in hand, and they are what disprove the
question. Everything downstream then helps the model explain a thing that never
happened.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from myscholar.adapters.fakes import FakeChatModel
from myscholar.domain.documents import ContextBlock
from myscholar.generation.answerer import Answerer, PremiseCheck


def _block(text: str) -> ContextBlock:
    return ContextBlock(
        label="C1",
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="A Chaos Based SVPWM Technique",
        section_heading="Results",
        text=text,
        page_start=1,
        page_end=1,
    )


def _answerer(model: FakeChatModel) -> Answerer:
    return Answerer(generator=FakeChatModel(), utility=model, person_name="Haseeb")


CONTEXT = (_block("Chaotic SVPWM spreads the switching harmonics, reducing peak EMI."),)


class TestWhenItRuns:
    @pytest.mark.parametrize(
        "question",
        [
            "Why did his NSGA-II study conclude that weighting factors are unnecessary?",
            "He found that sensorless control performs worse than sensored. What caused it?",
            "In his 2023 paper he showed that chaotic SVPWM increases EMI. Why?",
            "Why does his intrusion detection work recommend packet-level features?",
        ],
    )
    async def test_assertion_shaped_questions_are_checked(self, question: str) -> None:
        model = FakeChatModel(
            structured_responses=[
                PremiseCheck(premise="X", contradicted=True, correction="Actually Y.")
            ]
        )

        assert await _answerer(model).check_premise(question, CONTEXT) is not None

    @pytest.mark.parametrize(
        "question",
        [
            "What method is used to detect stator winding faults?",
            "Which journals has he published in?",
            "What accuracy did the framework achieve?",
        ],
    )
    async def test_plain_questions_cost_nothing(self, question: str) -> None:
        """A question that asserts nothing has no premise to test, and the call
        would be spent for certain on every ordinary question."""
        model = FakeChatModel()

        assert await _answerer(model).check_premise(question, CONTEXT) is None
        assert model.calls == []

    async def test_nothing_retrieved_means_nothing_to_contradict(self) -> None:
        model = FakeChatModel()

        assert await _answerer(model).check_premise("Why did he show X?", ()) is None
        assert model.calls == []


class TestWhatItReturns:
    async def test_a_contradicted_premise_is_reported(self) -> None:
        model = FakeChatModel(
            structured_responses=[
                PremiseCheck(
                    premise="chaotic SVPWM increases EMI",
                    contradicted=True,
                    correction="It reduces EMI by spreading harmonics.",
                )
            ]
        )

        result = await _answerer(model).check_premise("Why does it increase EMI?", CONTEXT)

        assert result is not None
        assert "reduces EMI" in result.correction

    async def test_silence_is_not_contradiction(self) -> None:
        """The excerpts being quiet about a claim is not evidence against it.

        Treating it as such would turn every question whose answer is missing
        into an accusation that the asker invented it.
        """
        model = FakeChatModel(
            structured_responses=[PremiseCheck(premise="something", contradicted=False)]
        )

        assert await _answerer(model).check_premise("Why did he show X?", CONTEXT) is None

    async def test_a_contradiction_without_a_premise_is_discarded(self) -> None:
        """Nothing to tell the generator, so nothing is claimed."""
        model = FakeChatModel(
            structured_responses=[PremiseCheck(premise="", contradicted=True, correction="no")]
        )

        assert await _answerer(model).check_premise("Why did he show X?", CONTEXT) is None

    async def test_an_unavailable_check_is_not_a_failed_check(self) -> None:
        """A checker that cannot be reached must not manufacture accusations."""
        result = await _answerer(FakeChatModel()).check_premise("Why did he show X?", CONTEXT)

        assert result is None
