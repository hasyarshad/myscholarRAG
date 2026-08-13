"""The HTTP contract the widget is written against.

Driven with fakes, so these run with no GPU, no database and no network. The
event ordering assertions matter most: the widget's rendering depends on
`answer` arriving last, and on stage events being parseable individually.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from myscholar.api.app import Services, create_app
from myscholar.api.limits import Gate, Limits, OverloadedError, RateLimitedError, RateLimiter
from myscholar.api.schemas import MAX_QUESTION_CHARS
from myscholar.api.sse import frame
from myscholar.domain.answers import Answer, Citation, Refusal, TraceEvent
from myscholar.domain.enums import RefusalReason, Stage


@dataclass
class _Doc:
    id: Any
    slug: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    verified: bool
    sections: int = 1
    chunks: int = 10


class _Store:
    def __init__(self, *, healthy: bool = True, chunks: int = 10) -> None:
        self._healthy = healthy
        self._chunks = chunks
        self.doc = _Doc(
            id=uuid4(),
            slug="a-paper",
            title="A Paper About Predictive Control",
            authors=["M. H. Arshad"],
            year=2021,
            venue="IEEE Access",
            verified=True,
        )

    async def list_documents(self) -> list[_Doc]:
        return [self.doc]

    async def counts(self) -> dict[str, int]:
        return {"documents": 1, "sections": 1, "chunks": self._chunks, "embeddings": self._chunks}

    async def health(self) -> bool:
        return self._healthy

    async def assert_embedder_matches(self, model_id: str) -> None:
        return


class _Engine:
    """Emits the response's trace as live events, then the response.

    Mirrors the real engine's contract: stages arrive as they happen and the
    finished response arrives last.
    """

    def __init__(self, response: Answer | Refusal) -> None:
        self._response = response
        self.questions: list[str] = []

    async def stream_events(self, question: str) -> AsyncIterator[Any]:
        self.questions.append(question)
        for event in self._response.trace:
            yield event
        yield self._response

    async def answer_with_trace(self, question: str) -> tuple[Any, Any]:
        self.questions.append(question)
        return self._response, None


def _answer(store: _Store) -> Answer:
    return Answer(
        text="Weighting factors are tuned with NSGA-II [C1].",
        citations=(
            Citation(
                label="C1",
                document_id=store.doc.id,
                section_id=uuid4(),
                document_title=store.doc.title,
                section_heading="IV. Results",
                page_start=4,
                page_end=5,
            ),
        ),
        confidence=1.0,
        trace=(
            TraceEvent(stage=Stage.RETRIEVING, detail="8 blocks"),
            TraceEvent(stage=Stage.GRADING, detail="6 kept, 2 dropped"),
            TraceEvent(stage=Stage.VERIFYING, detail="citations valid"),
        ),
    )


def _client(response: Answer | Refusal | None = None, *, store: _Store | None = None) -> TestClient:
    store = store or _Store()
    services = Services(
        settings=None,  # type: ignore[arg-type]
        engine=_Engine(response or _answer(store)),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        slugs={str(store.doc.id): store.doc.slug},
        ready=bool(store._chunks),  # noqa: SLF001
    )
    return TestClient(create_app(services))


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Decode a stream into (event, payload) pairs, as a client must."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        name = ""
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data.append(line.removeprefix("data: "))
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events


class TestSseFraming:
    def test_a_frame_has_an_event_and_data_line(self) -> None:
        text = frame("stage", '{"stage":"grading"}')

        assert text.startswith("event: stage\n")
        assert text.endswith("\n\n")

    def test_newlines_are_split_across_data_lines(self) -> None:
        """A raw newline terminates the frame early, truncating the event.

        Pydantic escapes newlines, so this only bites on raw string payloads —
        which is exactly when it would be missed.
        """
        text = frame("note", "line one\nline two")

        assert text.count("data: ") == 2
        assert "data: line one\ndata: line two" in text


class TestAsk:
    def test_the_answer_arrives_last(self) -> None:
        """The widget renders on `answer`; anything after it would be missed."""
        response = _client().post("/v1/ask", json={"question": "How are factors tuned?"})

        events = parse_sse(response.text)
        assert events[-1][0] == "answer"
        assert [name for name, _ in events].count("answer") == 1

    def test_pipeline_stages_are_streamed(self) -> None:
        """Progress, not tokens — verification runs after generation, so raw
        output would show text that is then rejected."""
        response = _client().post("/v1/ask", json={"question": "anything"})

        stages = [payload["stage"] for name, payload in parse_sse(response.text) if name == "stage"]
        assert "grading" in stages
        assert "verifying" in stages

    def test_citations_carry_what_the_widget_displays(self) -> None:
        response = _client().post("/v1/ask", json={"question": "anything"})

        answer = parse_sse(response.text)[-1][1]
        citation = answer["citations"][0]
        assert citation["label"] == "C1"
        assert citation["document_title"].startswith("A Paper")
        assert citation["page_start"] == 4
        assert citation["slug"] == "a-paper"

    def test_a_refusal_is_marked_and_carries_its_reason(self) -> None:
        refusal = Refusal(
            reason=RefusalReason.NO_RELEVANT_CONTENT,
            text="I don't have anything on that.",
            nearest_topics=("motor drives", "converter control"),
        )

        response = _client(refusal).post("/v1/ask", json={"question": "wind turbines?"})

        answer = parse_sse(response.text)[-1][1]
        assert answer["refused"] is True
        assert answer["refusal_reason"] == "no_relevant_content"
        assert "motor drives" in answer["nearest_topics"]

    def test_an_over_long_question_is_rejected_before_streaming(self) -> None:
        """Knowable in advance, so it gets a real status rather than an event."""
        response = _client().post("/v1/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})

        assert response.status_code == 422

    def test_an_empty_question_is_rejected(self) -> None:
        assert _client().post("/v1/ask", json={"question": ""}).status_code == 422

    def test_an_empty_index_returns_503_rather_than_a_stream(self) -> None:
        client = _client(store=_Store(chunks=0))

        response = client.post("/v1/ask", json={"question": "anything"})

        assert response.status_code == 503
        assert response.json()["code"] == "not_ready"

    def test_rate_limiting_returns_429_with_retry_after(self) -> None:
        client = _client()

        codes = [
            client.post("/v1/ask", json={"question": f"question {i}"}).status_code
            for i in range(12)
        ]

        assert 429 in codes
        limited = next(
            r for r in [client.post("/v1/ask", json={"question": "again"})] if r.status_code == 429
        )
        assert "Retry-After" in limited.headers


class TestHealth:
    def test_liveness_answers_without_checking_anything(self) -> None:
        assert _client().get("/healthz").json()["status"] == "ok"

    def test_readiness_checks_the_database_and_the_index(self) -> None:
        body = _client().get("/readyz").json()

        assert body["status"] == "ready"
        assert body["checks"] == {"database": True, "index": True}

    def test_an_empty_index_is_not_ready(self) -> None:
        """A process that is up with an empty index is the likeliest real
        failure, and a TCP connect would not notice it."""
        response = _client(store=_Store(chunks=0)).get("/readyz")

        assert response.status_code == 503
        assert response.json()["checks"]["index"] is False

    def test_an_unreachable_database_is_not_ready(self) -> None:
        response = _client(store=_Store(healthy=False)).get("/readyz")

        assert response.status_code == 503
        assert response.json()["checks"]["database"] is False


class TestCorpus:
    def test_papers_are_listed(self) -> None:
        papers = _client().get("/v1/papers").json()

        assert papers[0]["slug"] == "a-paper"
        assert papers[0]["verified"] is True

    def test_corpus_info_states_coverage_and_suggestions(self) -> None:
        body = _client().get("/v1/corpus").json()

        assert body["documents"] == 1
        assert body["years"] == [2021]
        assert body["suggestions"]


class TestLimits:
    def test_the_window_allows_traffic_under_the_limit(self) -> None:
        limiter = RateLimiter(Limits(per_minute=3))

        for _ in range(3):
            limiter.check("caller")

    def test_exceeding_the_minute_limit_raises(self) -> None:
        limiter = RateLimiter(Limits(per_minute=2))
        limiter.check("caller")
        limiter.check("caller")

        with pytest.raises(RateLimitedError):
            limiter.check("caller")

    def test_callers_are_limited_independently(self) -> None:
        limiter = RateLimiter(Limits(per_minute=1))
        limiter.check("first")

        limiter.check("second")

    async def test_the_gate_serialises_work(self) -> None:
        gate = Gate(Limits(max_concurrent=1, max_queued=4))

        async with gate:
            assert gate.queue_depth == 0

    async def test_the_first_caller_is_admitted_even_with_no_queue(self) -> None:
        """max_queued bounds waiting, not running.

        An earlier version checked the queue unconditionally, so max_queued=0
        rejected every request including the first.
        """
        gate = Gate(Limits(max_concurrent=1, max_queued=0))

        async with gate:
            pass

    async def test_a_full_queue_is_rejected_rather_than_stalled(self) -> None:
        """Telling a visitor to retry beats a spinner behind five other questions."""
        gate = Gate(Limits(max_concurrent=1, max_queued=0))

        async with gate:  # occupies the only slot
            with pytest.raises(OverloadedError):
                await gate.__aenter__()
