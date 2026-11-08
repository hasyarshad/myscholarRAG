"""Paths through the adaptive loop.

Bibliographic, out-of-scope and ambiguous questions bypass generation; other questions
are generated, then verified and checked for grounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from myscholar.domain.answers import Answer, Refusal
from myscholar.domain.documents import ContextBlock
from myscholar.domain.enums import QueryClass, RefusalReason
from myscholar.query.bibliographic import BibliographicAnswer, BibliographicIntent
from myscholar.query.classifier import Classification
from myscholar.query.planner import Plan
from myscholar.reasoning.engine import Engine, _refusal_text
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
    """Fails the test if the loop reaches generation."""

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
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(text="20 papers.", intent=BibliographicIntent.COUNT),
        )

        response = await _engine(routed).answer("How many papers are there?")

        assert isinstance(response, Answer)
        assert response.confidence == 1.0

    async def test_no_generation_happens(self) -> None:
        # _Answerer.generate raises if reached.
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
        routed = _Routed(
            classification=Classification(query_class=QueryClass.BIBLIOGRAPHIC),
            plan=Plan(query_class=QueryClass.BIBLIOGRAPHIC, retrieve=False, use_sql=True),
            bibliographic=BibliographicAnswer(text="7 papers.", intent=BibliographicIntent.COUNT),
        )

        items = [item async for item in _engine(routed).stream_events("how many?")]

        assert isinstance(items[-1], Answer)
        assert items[-1].text == "7 papers."


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


class _Generator:
    """Emits a fixed draft and answers every grounding check with ``grounded``."""

    def __init__(self, draft: str, *, grounded: bool = False) -> None:
        self.draft = draft
        self._grounded = grounded
        self.generations = 0

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
    async def test_grounding_runs_even_when_citations_had_to_be_salvaged(self) -> None:
        answerer = _Generator("Supported claim [C1]. Invented claim [C9].")
        _, trace = await _loop_engine(answerer).answer_with_trace("anything")

        assert "verify" in trace.path
        assert "check_grounding" in trace.path, (
            f"the salvaged draft skipped the grounding check; path was {trace.path}"
        )

    async def test_an_unsupported_statement_is_dropped_when_others_survive(self) -> None:
        answerer = _GeneratorPartial(
            "The framework uses an LSTM [C1]. It reached 98.53% accuracy [C1].",
            reject="98.53",
        )
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        text = getattr(response, "text", "")
        assert "98.53" not in text, f"the rejected statement survived: {text!r}"
        assert "LSTM" in text, f"the supported statement was lost too: {text!r}"

    async def test_a_wholly_rejected_draft_is_kept_and_marked_degraded(self) -> None:
        # Every marker already resolves to a retrieved passage, and refusing here
        # measured 11.8% over-refusal.
        answerer = _Generator("The framework uses an LSTM [C1].")
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer), "a verified draft was thrown away"
        assert "LSTM" in response.text
        assert response.degraded, "lower confidence was not signalled"

    async def test_a_grounded_answer_is_returned_untouched(self) -> None:
        answerer = _Generator("A supported claim [C1].", grounded=True)
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer)
        assert "A supported claim" in response.text


class TestVerificationSeverity:
    """A missing or invented marker costs its sentence, not the answer.

    Regenerating instead added 0.08 wrong claims per question and cost 20 points of correct
    refusal and 14 of attribution precision.
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
        answerer = _Generator(
            "The framework uses an LSTM [C1]. A claim about the work [C9].",
            grounded=True,
        )
        response, _ = await _loop_engine(answerer).answer_with_trace("anything")

        assert isinstance(response, Answer)
        assert "[C9]" not in response.text, "an invented citation survived to the reader"
        assert "LSTM" in response.text, "the sound sentence was lost with the bad one"

    async def test_an_invented_marker_costs_no_regeneration(self) -> None:
        answerer = _Generator("A claim about the work [C9].", grounded=True)
        await _loop_engine(answerer).answer_with_trace("anything")

        assert answerer.generations == 1, (
            f"a dangling citation cost {answerer.generations} generations, expected 1"
        )


class TestRefusalTextMatchesTheReason:
    """Each refusal reason has its own wording, so a timeout never claims the corpus is silent."""

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
        texts = {
            _refusal_text(reason, ("a topic",))
            for reason in (
                RefusalReason.BUDGET_EXHAUSTED,
                RefusalReason.UNGROUNDED_AFTER_RETRY,
                RefusalReason.NO_RELEVANT_CONTENT,
            )
        }

        assert len(texts) == 3, f"reasons collapsed onto the same wording: {texts}"


class TestAFailedGenerationRefusesRatherThanReturningNothing:
    """A generator outage ends in a refusal, not an Answer with empty text."""

    def _block(self) -> ContextBlock:
        return ContextBlock(
            label="C1",
            document_id=uuid4(),
            section_id=uuid4(),
            document_title="Weighting Factors Optimization of MPTC",
            section_heading="Results",
            text="Switching loss falls as the horizon shortens.",
            page_start=1,
            page_end=1,
        )

    def _routed(self) -> _Routed:
        return _Routed(
            classification=Classification(query_class=QueryClass.METHODOLOGICAL),
            plan=Plan(query_class=QueryClass.METHODOLOGICAL),
            blocks=(self._block(),),
        )

    async def test_a_generator_outage_produces_a_refusal(self) -> None:
        class _Failing(_Answerer):
            async def check_premise(self, *args: Any, **kwargs: Any) -> Any:
                return None

            async def generate(self, *args: Any, **kwargs: Any) -> str:
                msg = "hosted model unavailable"
                raise RuntimeError(msg)

        engine = Engine(
            router=_Router(self._routed()),  # type: ignore[arg-type]
            answerer=_Failing(),  # type: ignore[arg-type]
            budget=Budget(),
        )
        response = await engine.answer("How is switching loss reduced?")

        assert isinstance(response, Refusal), f"got an Answer: {response!r}"
        assert response.text.strip(), "a refusal must still say something"
