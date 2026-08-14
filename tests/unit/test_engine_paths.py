"""Paths through the adaptive loop that bypass generation.

Three question classes are answered without ever reaching the generator, and
each one has a way of losing its answer on the way out. The bibliographic case
regressed twice — once from a missing guard and once when an unrelated
``git checkout`` reverted the fix — which is why it is tested rather than only
commented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from myscholar.domain.answers import Answer, Refusal
from myscholar.domain.enums import QueryClass
from myscholar.query.bibliographic import BibliographicAnswer, BibliographicIntent
from myscholar.query.classifier import Classification
from myscholar.query.planner import Plan
from myscholar.reasoning.engine import Engine
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
