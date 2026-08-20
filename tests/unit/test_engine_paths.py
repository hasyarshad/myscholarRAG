"""Paths through the adaptive loop that bypass generation.

Three question classes are answered without ever reaching the generator, and
each one has a way of losing its answer on the way out. The bibliographic case
regressed twice — once from a missing guard and once when an unrelated
``git checkout`` reverted the fix — which is why it is tested rather than only
commented.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from myscholar.domain.answers import Answer, Refusal
from myscholar.domain.enums import QueryClass, RefusalReason
from myscholar.query.bibliographic import BibliographicAnswer, BibliographicIntent
from myscholar.query.classifier import Classification
from myscholar.query.planner import Plan
from myscholar.reasoning.engine import Engine, LoopState, _refusal_text
from myscholar.reasoning.graph import Budget


@dataclass
class _Routed:
    classification: Classification
    plan: Plan
    blocks: tuple[Any, ...] = ()
    bibliographic: BibliographicAnswer | None = None
    queries: tuple[Any, ...] = ()
    degraded: bool = False
    timings_ms: dict[str, int] = field(default_factory=dict)


class _Router:
    def __init__(self, routed: _Routed) -> None:
        self._routed = routed

    async def route(self, question: str) -> _Routed:
        return self._routed


class _Answerer:
    """Fails loudly if the loop reaches it.

    These paths must not generate: doing so would mean a model was asked to
    phrase a count it could get wrong, which is the failure the SQL path exists
    to remove.
    """

    async def grade(self, *args: Any, **kwargs: Any) -> Any:
        msg = "grading should not run on a bypassed path"
        raise AssertionError(msg)

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        msg = "generation should not run on a bypassed path"
        raise AssertionError(msg)

    async def nearest_topics(self, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        return ()


def _engine(routed: _Routed) -> Engine:
    return Engine(
        router=_Router(routed),  # type: ignore[arg-type]
        answerer=_Answerer(),  # type: ignore[arg-type]
        budget=Budget(),
    )


class TestBibliographicPath:
    async def test_the_sql_answer_survives_to_the_caller(self) -> None:
        """The regression: the terminal node rebuilt the response from the
        (empty) draft, replacing "4 papers in 2023." with an empty string."""
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(
                text="4 papers in 2023.", intent=BibliographicIntent.COUNT
            ),
        )

        response = await _engine(routed).answer("How many papers in 2023?")

        assert isinstance(response, Answer)
        assert response.text == "4 papers in 2023."

    async def test_it_is_reported_with_full_confidence(self) -> None:
        """It came from the database, not from a model's recollection."""
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(text="20 papers.", intent=BibliographicIntent.COUNT),
        )

        response = await _engine(routed).answer("How many papers are there?")

        assert isinstance(response, Answer)
        assert response.confidence == 1.0

    async def test_no_generation_happens(self) -> None:
        """The _Answerer raises if reached, so reaching it fails the test."""
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(text="3 papers.", intent=BibliographicIntent.COUNT),
        )

        await _engine(routed).answer("How many in 2021?")


class TestCannedPaths:
    @pytest.mark.parametrize(
        ("query_class", "clarification"),
        [(QueryClass.OUT_OF_SCOPE, False), (QueryClass.AMBIGUOUS, True)],
    )
    async def test_a_canned_reply_survives_to_the_caller(
        self, query_class: QueryClass, clarification: bool
    ) -> None:
        routed = _Routed(
            classification=Classification(query_class=query_class),
            plan=Plan(
                query_class=query_class,
                retrieve=False,
                canned_response="A canned reply.",
                needs_clarification=clarification,
            ),
        )

        response = await _engine(routed).answer("anything")

        assert isinstance(response, Refusal)
        assert response.text == "A canned reply."

    async def test_the_stream_ends_with_the_response(self) -> None:
        """The API renders on the response, so it has to arrive last."""
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(text="7 papers.", intent=BibliographicIntent.COUNT),
        )

        items = [item async for item in _engine(routed).stream_events("how many?")]

        assert isinstance(items[-1], Answer)
        assert items[-1].text == "7 papers."


# ---------------------------------------------------------------------------
# The final attempt must be checked as thoroughly as the first.
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    """Just enough of a ContextBlock for the loop."""

    label: str = "C1"
    text: str = "the corpus says something supportable"
    document_id: Any = "doc"
    section_id: Any = "sec"
    document_title: str = "A Paper"
    section_heading: str = "Results"
    page_start: int = 1
    page_end: int = 2
    document_kind: str = "paper"
    is_review: bool = False
    authors: tuple[str, ...] = ()
    source_url: str = "https://doi.org/10.0000/test"


@dataclass
class _Graded:
    kept: tuple[Any, ...]
    rejected: tuple[Any, ...] = ()
    unavailable: bool = False

    @property
    def sufficient(self) -> bool:
        return bool(self.kept)


class _Generator:
    """Emits a fixed draft, and calls every claim ungrounded."""

    def __init__(self, draft: str, *, grounded: bool = False) -> None:
        self.draft = draft
        self._grounded = grounded
        self.generations = 0

    async def grade(self, question: str, blocks: Any) -> _Graded:
        return _Graded(kept=tuple(blocks))

    async def check_premise(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        self.generations += 1
        return self.draft

    async def is_grounded(self, statement: str, excerpt: str) -> bool:
        return self._grounded

    async def rewrite(self, *args: Any, **kwargs: Any) -> str | None:
        return None

    async def nearest_topics(self, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        return ()


class _GeneratorPartial(_Generator):
    """Rejects only sentences containing a marker string."""

    def __init__(self, draft: str, *, reject: str) -> None:
        super().__init__(draft)
        self._reject = reject

    async def is_grounded(self, statement: str, excerpt: str) -> bool:
        return self._reject not in statement


def _loop_engine(answerer: _Generator) -> Engine:
    routed = _Routed(
        classification=Classification(query_class=QueryClass.SINGLE_PAPER_FACTUAL),
        plan=Plan(query_class=QueryClass.SINGLE_PAPER_FACTUAL),
        blocks=(_Block(),),
    )
    return Engine(
        router=_Router(routed),  # type: ignore[arg-type]
        answerer=answerer,  # type: ignore[arg-type]
        budget=Budget(),
    )


class TestTheLastAttemptIsStillChecked:
    """Found in use, not in testing.

    A three-attempt answer ended with citation issues and a spent regeneration
    budget, so the loop went straight to the terminal node — making the text
    actually returned the one output the groundedness check never saw. A
    fabricated accuracy figure carrying a valid marker survived on that path.
    """

    async def test_grounding_runs_even_when_citations_had_to_be_salvaged(self) -> None:
        answerer = _Generator("Supported claim [C1]. Invented claim [C9].")
        _, trace = await _loop_engine(answerer).answer_with_trace("anything")

        assert "verify" in trace.path
        assert "check_grounding" in trace.path, (
            f"the salvaged draft skipped the grounding check; path was {trace.path}"
        )

    async def test_an_unsupported_statement_is_dropped_when_others_survive(self) -> None:
        """Partial rejection removes the bad sentence and keeps the rest."""
        answerer = _GeneratorPartial(
            "The framework uses an LSTM [C1]. It reached 98.53% accuracy [C1].",
            reject="98.53",
        )
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        text = getattr(response, "text", "")
        assert "98.53" not in text, f"the rejected statement survived: {text!r}"
        assert "LSTM" in text, f"the supported statement was lost too: {text!r}"

    async def test_a_wholly_rejected_draft_is_kept_and_marked_degraded(self) -> None:
        """A judgement that would delete everything is treated as suspect.

        Refusing here measured 11.8% over-refusal — the system declining
        questions the corpus answers. The deterministic checks are the gate:
        every marker resolves to a retrieved, graded passage, so a draft that
        passes them stands, flagged rather than discarded.
        """
        answerer = _Generator("The framework uses an LSTM [C1].")
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer), "a verified draft was thrown away"
        assert "LSTM" in response.text
        assert response.degraded, "lower confidence was not signalled"

    async def test_a_grounded_answer_is_returned_untouched(self) -> None:
        """The guard must not eat good answers."""
        answerer = _Generator("A supported claim [C1].", grounded=True)
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer)
        assert "A supported claim" in response.text


class TestVerificationSeverity:
    """A missing marker and an invented one both cost their sentence, not the answer.

    They were once treated differently: `uncited` dropped the sentence,
    `dangling` forced a regeneration, because an invented source is the failure
    the whole mechanism exists to catch. That distinction only ever chose
    between dropping and retrying, and retrying is gone — it was measured to
    cost +0.08 wrong claims per question, twenty points of correct refusal, and
    fourteen points of attribution precision.

    The guarantee is unchanged and is the one that matters: **a marker pointing
    at a passage that was never retrieved never reaches the reader.** It leaves
    with its sentence instead of triggering another attempt.
    """

    async def test_a_missing_marker_costs_the_sentence_not_the_answer(self) -> None:
        answerer = _Generator(
            "The framework uses an LSTM [C1]. It was evaluated on real recordings.",
            grounded=True,
        )
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer)
        assert "LSTM" in response.text, "the cited sentence was lost"
        assert answerer.generations == 1, (
            f"a formatting slip cost {answerer.generations} generations; it should cost none"
        )

    async def test_an_invented_marker_never_reaches_the_reader(self) -> None:
        """A citation pointing at nothing is the failure this exists to catch."""
        answerer = _Generator(
            "The framework uses an LSTM [C1]. A claim about the work [C9].",
            grounded=True,
        )
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer)
        assert "[C9]" not in response.text, "an invented citation survived to the reader"
        assert "LSTM" in response.text, "the sound sentence was lost with the bad one"

    async def test_an_invented_marker_costs_no_regeneration(self) -> None:
        """The behaviour that changed, pinned so the deletion is not undone by accident."""
        answerer = _Generator("A claim about the work [C9].", grounded=True)
        await _loop_engine(answerer).answer_with_trace("anything")

        assert answerer.generations == 1, (
            f"a dangling citation cost {answerer.generations} generations; retrying was removed"
        )


class TestRefusalTextMatchesTheReason:
    """A refusal has to say which of three different things went wrong.

    They shared one sentence — "I don't have anything on that in these papers"
    — while the loop computed the reason correctly and then dropped it. That
    turned a stalled model call on the development machine into a confident and
    false claim about the corpus, on a question the papers answer in detail.
    """

    def test_running_out_of_time_does_not_claim_the_corpus_is_empty(self) -> None:
        text = _refusal_text(RefusalReason.BUDGET_EXHAUSTED, ("motor drives",))

        assert "ran out of time" in text
        assert "don't have anything" not in text, (
            "a timeout claimed the corpus does not cover the question"
        )
        assert "motor drives" not in text, (
            "a timeout listed nearest topics, implying a search that finished"
        )

    def test_ungrounded_says_passages_were_found(self) -> None:
        text = _refusal_text(RefusalReason.UNGROUNDED_AFTER_RETRY, ("sensorless control",))

        assert "found passages" in text
        assert "sensorless control" in text
        assert "don't have anything" not in text

    def test_nothing_relevant_still_says_so_and_offers_the_nearest(self) -> None:
        text = _refusal_text(RefusalReason.NO_RELEVANT_CONTENT, ("MPC", "fault detection"))

        assert "don't have anything on that in these papers" in text
        assert "MPC and fault detection" in text

    def test_every_reason_produces_a_distinct_message(self) -> None:
        """The point of the change, stated as a property rather than by example."""
        texts = {
            _refusal_text(reason, ("a topic",))
            for reason in (
                RefusalReason.BUDGET_EXHAUSTED,
                RefusalReason.UNGROUNDED_AFTER_RETRY,
                RefusalReason.NO_RELEVANT_CONTENT,
            )
        }

        assert len(texts) == 3, f"reasons collapsed onto the same wording: {texts}"


class TestTheBestAttemptIsTheOneReturned:
    """Regeneration is not guaranteed to improve a draft.

    Asked about someone's strengths in machine learning the loop produced 1272
    characters with four unsupported statements, then 679 with one, then 730
    with three — and returned the third, because each attempt overwrote the
    last. The second was the best answer the system ever held and the reader
    never saw it.
    """

    def test_the_attempt_that_lost_least_is_kept(self) -> None:
        state = LoopState(question="q", budget=Budget())
        # Attempt one: two rejections, recorded as the best so far.
        state = replace(state, draft="A [C1]. B [C1].", best_draft="", best_ungrounded=())
        first_rejected = ("A [C1].", "B [C1].")
        state = replace(state, best_draft=state.draft, best_ungrounded=first_rejected)
        # Attempt two loses only one, so it displaces the first.
        second = "C [C1]. D [C1]."
        state = replace(state, draft=second, best_draft=second, best_ungrounded=("C [C1].",))
        # Attempt three is worse again and must not displace it.
        third_rejected = ("E [C1].", "F [C1].", "G [C1].")
        state = replace(state, draft="E [C1]. F [C1]. G [C1].", ungrounded=third_rejected)

        better = state.best_draft and len(state.best_ungrounded) < len(state.ungrounded)

        assert better, "the worse final attempt would have been returned"
        assert state.best_draft == second
