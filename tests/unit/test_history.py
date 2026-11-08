"""Follow-up resolution.

The guards are tested directly: a rewrite that changes the question yields a cited
answer to something the visitor did not ask.
"""

from __future__ import annotations

from typing import Any

import pytest

from myscholar.adapters.fakes import FakeChatModel
from myscholar.query.history import FollowUpResolver, Turn, check, looks_elliptical

HISTORY = [Turn(question="How many did he publish in 2026?", answer="1 paper in 2026.")]


class TestLooksElliptical:
    @pytest.mark.parametrize(
        "question",
        [
            "how many in 2025",
            "and 2024?",
            "what about 2023",
            "and the second one?",
            "how about that",
            "why?",
        ],
    )
    def test_fragments_are_elliptical(self, question: str) -> None:
        assert looks_elliptical(question)

    @pytest.mark.parametrize(
        "question",
        [
            "what data did it use?",
            "what method did it use?",
            "how does it work",
            "what did they find?",
            "how accurate was it",
        ],
    )
    def test_an_unbound_pronoun_is_elliptical(self, question: str) -> None:
        # Short, with ordinary openings: only the unbound pronoun marks these as follow-ups.
        assert looks_elliptical(question)

    @pytest.mark.parametrize(
        "question",
        ["Where did he do his PhD?", "What subject did he study at university?"],
    )
    def test_he_and_his_do_not_count_as_unbound(self, question: str) -> None:
        # They refer to the person the corpus is about, so they are bound in every question.
        assert not looks_elliptical(question)

    @pytest.mark.parametrize(
        "question",
        [
            "How many papers did he publish in 2023?",
            "What method did he use for fault detection?",
            "Where did he do his PhD?",
            "Which journals has he published in?",
        ],
    )
    def test_complete_questions_are_not(self, question: str) -> None:
        assert not looks_elliptical(question)

    def test_a_long_question_carries_its_own_context(self) -> None:
        assert not looks_elliptical(
            "and how did the adaptive discrete-time sliding mode controller "
            "handle the grid-tied inverter under parameter variation?"
        )

    def test_an_empty_question_is_not_elliptical(self) -> None:
        assert not looks_elliptical("   ")


class TestGuards:
    def test_a_dropped_year_is_rejected(self) -> None:
        assert check("how many in 2025", "How many papers did he publish in 2026?")

    def test_a_kept_year_passes(self) -> None:
        assert not check("how many in 2025", "How many papers did he publish in 2025?")

    def test_an_answer_instead_of_a_question_is_rejected(self) -> None:
        assert check("and 2024?", "He published " + "three papers in 2024. " * 20)

    def test_an_empty_rewrite_is_rejected(self) -> None:
        assert check("and 2024?", "   ")

    def test_a_shorter_rewrite_is_rejected(self) -> None:
        assert check("what about the Reddit one?", "the Reddit study")

    def test_only_the_first_line_is_considered(self) -> None:
        # Small models append commentary after the question.
        assert not check(
            "and 2024?", 'How many papers did he publish in 2024?\n\nNote: I inferred "papers".'
        )


class TestResolver:
    async def test_a_fragment_is_expanded(self) -> None:
        model = FakeChatModel(responses={"Follow-up": "How many papers did he publish in 2025?"})

        result = await FollowUpResolver(model).resolve("how many in 2025", HISTORY)

        assert result.rewritten
        assert result.question == "How many papers did he publish in 2025?"
        assert result.original == "how many in 2025"

    async def test_no_history_means_no_rewrite(self) -> None:
        model = FakeChatModel(responses={"Follow-up": "something else entirely"})

        result = await FollowUpResolver(model).resolve("how many in 2025", [])

        assert not result.rewritten
        assert result.question == "how many in 2025"
        assert model.calls == []

    async def test_a_complete_question_is_untouched(self) -> None:
        model = FakeChatModel(responses={"Follow-up": "a rewrite that should never happen"})

        result = await FollowUpResolver(model).resolve(
            "What method did he use for fault detection?", HISTORY
        )

        assert not result.rewritten
        assert model.calls == []

    async def test_a_rejected_rewrite_falls_back_to_the_question(self) -> None:
        model = FakeChatModel(responses={"Follow-up": "How many papers were published in 2026?"})

        result = await FollowUpResolver(model).resolve("how many in 2025", HISTORY)

        assert not result.rewritten
        assert result.question == "how many in 2025"
        assert "2025" in result.reason

    async def test_a_model_failure_falls_back_to_the_question(self) -> None:
        class _Broken(FakeChatModel):
            async def complete(self, *args: Any, **kwargs: Any) -> str:
                msg = "model unavailable"
                raise RuntimeError(msg)

        result = await FollowUpResolver(_Broken()).resolve("how many in 2025", HISTORY)

        assert not result.rewritten
        assert result.question == "how many in 2025"

    async def test_no_model_falls_back_to_the_question(self) -> None:
        result = await FollowUpResolver(None).resolve("how many in 2025", HISTORY)

        assert not result.rewritten


class TestRefusalsAreNotAntecedents:
    async def test_a_refused_turn_is_marked_as_unanswered(self) -> None:
        model = FakeChatModel(responses={"Follow-up": "How many papers in 2025?"})
        history = [
            Turn(question="what is the weather?", answer="I only answer...", refused=True),
        ]

        await FollowUpResolver(model).resolve("how many in 2025", history)

        prompt = model.calls[0][0].content
        assert "(not answered)" in prompt
        assert "I only answer" not in prompt
