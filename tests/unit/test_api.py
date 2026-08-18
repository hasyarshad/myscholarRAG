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
from myscholar.config import load_settings
from myscholar.domain.answers import Answer, Citation, Refusal, TraceEvent
from myscholar.domain.enums import RefusalReason, Stage
from myscholar.query.history import FollowUpResolver, Turn


@dataclass
class _Doc:
    id: Any
    slug: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    verified: bool
    kind: str = "paper"
    """The corpus endpoint reports papers only — it counted the CV and the
    site pages too, and the widget then told visitors there were 25 papers."""

    sections: int = 1
    chunks: int = 10


class _Store:
    def __init__(self, *, healthy: bool = True, chunks: int = 10) -> None:
        self._healthy = healthy
        self._chunks = chunks
        self.turns: dict[str, list[Turn]] = {}
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

    # -- conversations -----------------------------------------------------

    async def recent_turns(self, conversation_id: str, *, limit: int = 4) -> list[Turn]:
        return list(self.turns.get(conversation_id, []))

    async def append_turn(
        self, conversation_id: str, *, question: str, answer: str, refused: bool
    ) -> None:
        self.turns.setdefault(conversation_id, []).append(
            Turn(question=question, answer=answer, refused=refused)
        )


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
        settings=load_settings(),
        engine=_Engine(response or _answer(store)),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        resolver=FollowUpResolver(None),
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


class TestCors:
    """A misconfigured origin fails in the browser and nowhere else.

    The API stays healthy, curl works, every test passes — and the widget
    reports "could not reach the assistant". That happened: the allow list
    hardcoded port 3000 and the dev server was started on 3100.
    """

    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3100",
            "https://haseeb-arshad.me",
        ],
    )
    def test_expected_origins_pass_preflight(self, origin: str) -> None:
        response = _client().options(
            "/v1/ask",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    def test_an_unknown_origin_is_refused(self) -> None:
        """Exact origins, never a wildcard."""
        response = _client().options(
            "/v1/ask",
            headers={
                "Origin": "https://not-my-site.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 400

    def test_a_configured_origin_is_honoured(self) -> None:
        from myscholar.api.app import create_app as build

        store = _Store()
        services = Services(
            settings=load_settings(),
            engine=_Engine(_answer(store)),  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            resolver=FollowUpResolver(None),
            ready=True,
        )
        client = TestClient(build(services, origins=["https://elsewhere.example"]))

        response = client.options(
            "/v1/ask",
            headers={
                "Origin": "https://elsewhere.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200


class TestConversations:
    """Follow-ups resolve against turns the server owns."""

    def test_a_session_is_issued_before_anything_else(self) -> None:
        """Sent first so a client that disconnects mid-answer can still
        continue the conversation on its next request."""
        client = _client()

        events = parse_sse(client.post("/v1/ask", json={"question": "hello?"}).text)

        assert events[0][0] == "session"
        assert events[0][1]["session_id"]

    def test_the_turn_is_recorded(self) -> None:
        store = _Store()
        client = _client(store=store)

        body = client.post("/v1/ask", json={"question": "what is it about?"}).text
        session = parse_sse(body)[0][1]["session_id"]

        assert [t.question for t in store.turns[session]] == ["what is it about?"]

    def test_the_same_session_accumulates_turns(self) -> None:
        store = _Store()
        client = _client(store=store)

        first = parse_sse(client.post("/v1/ask", json={"question": "one"}).text)[0][1]["session_id"]
        client.post("/v1/ask", json={"question": "two", "session_id": first})

        assert [t.question for t in store.turns[first]] == ["one", "two"]

    def test_a_client_invented_identifier_is_replaced(self) -> None:
        """A guessable id would hand one visitor another's questions as the
        context for resolving a follow-up."""
        store = _Store()
        client = _client(store=store)

        body = client.post("/v1/ask", json={"question": "one", "session_id": "session-1"}).text
        issued = parse_sse(body)[0][1]["session_id"]

        assert issued != "session-1"
        assert "session-1" not in store.turns

    def test_the_original_question_is_stored_not_the_rewrite(self) -> None:
        """The transcript should read as the visitor typed it."""
        store = _Store()
        client = _client(store=store)

        body = client.post("/v1/ask", json={"question": "and 2024?"}).text
        session = parse_sse(body)[0][1]["session_id"]

        assert store.turns[session][0].question == "and 2024?"

    def test_a_failure_to_record_does_not_lose_the_answer(self) -> None:
        store = _Store()

        async def _boom(*args: Any, **kwargs: Any) -> None:
            msg = "database gone"
            raise RuntimeError(msg)

        store.append_turn = _boom  # type: ignore[method-assign]
        client = _client(store=store)

        events = parse_sse(client.post("/v1/ask", json={"question": "anything"}).text)

        assert [name for name, _ in events][-1] == "answer"
