"""The HTTP service.

One endpoint that matters: ``POST /v1/ask``, which streams the pipeline's
progress and then the verified answer.

See docs/learning/07-streaming-apis.md.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myscholar.adapters.persistence.repository import PostgresStore, create_engine
from myscholar.api.limits import Gate, OverloadedError, RateLimitedError, RateLimiter
from myscholar.api.schemas import (
    AnswerEvent,
    AskRequest,
    CitationView,
    CorpusInfo,
    ErrorEvent,
    HealthResponse,
    PaperView,
    SessionEvent,
    StageEvent,
)
from myscholar.api.sse import SSE_HEADERS, frame
from myscholar.assembly import build_engine
from myscholar.config import Settings, load_settings
from myscholar.container import build_container
from myscholar.domain.answers import Answer, Refusal, TraceEvent
from myscholar.query.history import FollowUpResolver, Turn
from myscholar.reasoning.engine import Engine

logger = logging.getLogger(__name__)

SUGGESTIONS = [
    "How are weighting factors tuned in predictive torque control?",
    "How many papers were published in 2023?",
    "What method detects stator winding inter-turn faults?",
    "How has the work on model predictive control developed?",
]


@dataclass
class Services:
    """Built once at startup. Model loading is slow and must not happen per request."""

    settings: Settings
    engine: Engine
    store: PostgresStore
    resolver: FollowUpResolver
    slugs: dict[str, str] = field(default_factory=dict)
    ready: bool = False
    detail: dict[str, str] = field(default_factory=dict)


async def build_services(settings: Settings | None = None) -> Services:
    settings = settings or load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    services = Services(
        settings=settings,
        engine=build_engine(settings, container, store),
        store=store,
        # The small model, not the generator. Expanding an ellipsis is a
        # rewriting task, and paying 14b latency for it on every follow-up
        # would be most of the win given back.
        resolver=FollowUpResolver(container.utility),
    )

    try:
        # Idempotent, and the conversation table would otherwise only exist on
        # a machine that had run an ingest. A serving process doing DDL is a
        # compromise: acceptable while the database is a local one owned by the
        # same person, and to be replaced by a migration step before the API
        # runs against a hosted database with a restricted role.
        await store.create_schema()
        documents = await store.list_documents()
        services.slugs = {str(d.id): d.slug for d in documents}
        # The index/embedder agreement check from Phase 1. Failing here at
        # startup beats serving plausible nonsense: searching an index built by
        # a different model returns ranked, meaningless results.
        await store.assert_embedder_matches(container.embedder.model_id)
        services.ready = bool(documents)
        if not documents:
            services.detail["index"] = "no documents indexed"
    except Exception as exc:  # startup reports problems, it does not crash
        services.detail["startup"] = f"{type(exc).__name__}: {exc}"
        logger.exception("service startup checks failed")

    return services


def create_app(services: Services | None = None, *, origins: list[str] | None = None) -> FastAPI:
    state: dict[str, Services | None] = {"services": services}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        # Models and database connections are built once. Doing it per request
        # would add seconds of model loading to every question.
        if state["services"] is None:
            state["services"] = await build_services()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="myScholar",
        version="0.1.0",
        description="Ask about a corpus of research papers. Answers carry verified citations.",
    )

    # Exact origins, never a wildcard. Worth being clear that CORS is a
    # browser-only control and stops nothing from curl — the real gate in
    # production is a shared secret injected by the Next.js proxy.
    allowed = origins or (services.settings.api.allowed_origins if services else None)
    if allowed is None:
        allowed = load_settings().api.allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    limiter = RateLimiter()
    gate = Gate()

    async def get_services() -> Services:
        if state["services"] is None:
            state["services"] = await build_services()
        current = state["services"]
        if current is None:  # pragma: no cover - just assigned above
            msg = "services failed to build"
            raise RuntimeError(msg)
        return current

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness. Deliberately checks nothing — it answers if the process runs."""
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz(response: Response) -> HealthResponse:
        """Readiness, which is a different question from liveness.

        Checks that the database answers and that the index is non-empty. A
        process that is up with an empty index is the most likely real failure,
        and a TCP connect would not notice it.
        """
        current = await get_services()
        database = await current.store.health()
        counts = await current.store.counts() if database else {}
        indexed = bool(counts.get("chunks"))

        checks = {"database": database, "index": indexed}
        ok = all(checks.values())
        response.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ready" if ok else "degraded",
            checks=checks,
            detail={**current.detail, **{k: str(v) for k, v in counts.items()}},
        )

    @app.get("/v1/papers", response_model=list[PaperView])
    async def papers() -> list[PaperView]:
        current = await get_services()
        return [
            PaperView(
                slug=d.slug,
                title=d.title,
                authors=d.authors,
                year=d.year,
                venue=d.venue,
                verified=d.verified,
            )
            for d in await current.store.list_documents()
        ]

    @app.get("/v1/corpus", response_model=CorpusInfo)
    async def corpus() -> CorpusInfo:
        """What the assistant covers, so the widget states it rather than guessing."""
        current = await get_services()
        documents = await current.store.list_documents()
        counts = await current.store.counts()
        return CorpusInfo(
            documents=counts.get("documents", 0),
            chunks=counts.get("chunks", 0),
            years=sorted({d.year for d in documents if d.year}, reverse=True),
            suggestions=SUGGESTIONS,
        )

    @app.post("/v1/ask")
    async def ask(payload: AskRequest, request: Request) -> Response:
        """Answer a question, streaming the pipeline's progress.

        Progress rather than tokens: verification runs after generation, so
        streaming raw output would show text that is then rejected.
        """
        current = await get_services()
        caller = _caller(request)

        # Failures knowable before the stream opens get a real HTTP status.
        # Only failures *during* streaming use an error event.
        try:
            limiter.check(caller)
        except RateLimitedError as exc:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(exc.retry_after)},
                content={"code": "rate_limited", "message": str(exc)},
            )

        if not current.ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"code": "not_ready", "message": "the index is unavailable"},
            )

        return StreamingResponse(
            _stream(current, gate, payload.question, request, payload.session_id),
            headers=SSE_HEADERS,
        )

    return app


async def _stream(
    services: Services,
    gate: Gate,
    question: str,
    request: Request,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """Emit stage events as the pipeline runs, then the verified answer."""
    started = time.monotonic()

    # Issued here, never taken from the request. A client-chosen identifier
    # could be guessed or reused, which would hand one visitor another
    # visitor's questions as the context for resolving a follow-up.
    conversation = (
        session_id if session_id and _looks_issued(session_id) else secrets.token_urlsafe(24)
    )
    asked = question

    try:
        async with gate:
            if await request.is_disconnected():
                return

            yield frame("session", SessionEvent(session_id=conversation))

            # Restarted after the concurrency gate. Measured from before it,
            # the elapsed time includes however long the request queued behind
            # another, so a metadata lookup that takes 6ms was reported as 28
            # seconds — describing the server's load rather than the work, in
            # the one place the interface claims to show the work.
            started = time.monotonic()

            yield frame(
                "stage",
                # No detail: the widget already labels this stage, and repeating
                # the label as its own detail read as a stutter.
                StageEvent(stage="classifying"),
            )

            # Each stage as the pipeline reaches it, not replayed afterwards.
            # A visitor watching "verifying: 2 citation issues" followed by a
            # regeneration learns something true about the system that no
            # amount of README prose conveys.
            # The follow-up is expanded once, here, and everything downstream
            # sees an ordinary self-contained question. Failure is a no-op:
            # the question as asked is used, which is exactly the behaviour
            # before conversations existed.
            resolution = await services.resolver.resolve(
                question, await _history(services, conversation)
            )
            if resolution.rewritten:
                yield frame(
                    "stage",
                    StageEvent(
                        stage="classifying",
                        detail=f"read as: {resolution.question}",
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    ),
                )

            response: Answer | Refusal | None = None
            async for item in services.engine.stream_events(resolution.question):
                if isinstance(item, TraceEvent):
                    yield frame(
                        "stage",
                        StageEvent(
                            stage=item.stage.value,
                            detail=item.detail,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                        ),
                    )
                    continue
                response = item

            if response is None:  # pragma: no cover - the stream always ends with one
                msg = "the pipeline produced no response"
                raise RuntimeError(msg)

    except OverloadedError:
        yield frame(
            "error",
            ErrorEvent(
                code="overloaded",
                message="Too many questions at once. Try again in a moment.",
                retryable=True,
            ),
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a mid-stream failure is reported in-band
        logger.exception("answering failed")
        yield frame(
            "error",
            ErrorEvent(code="internal", message=f"{type(exc).__name__}", retryable=True),
        )
        return

    # Recorded after the fact, and only once an answer exists. Storing the
    # question up front would leave a dangling turn whenever a visitor closed
    # the tab mid-answer, and a follow-up would then resolve against a
    # question that was never answered.
    try:
        await services.store.append_turn(
            conversation,
            question=asked,
            answer=response.text,
            refused=isinstance(response, Refusal),
        )
    except Exception:  # losing a turn must not lose the answer
        logger.exception("could not record the conversation turn")

    yield frame("answer", _answer_event(services, response))


async def _history(services: Services, conversation: str) -> list[Turn]:
    """Previous turns, or none if they cannot be read.

    Conversation memory is a convenience layered on top of answering, and it
    must never be able to prevent it. This was not hypothetical: an unmigrated
    database turned every question into an "internal error" event, because a
    lookup for a feature nobody had asked for was on the path to the answer.
    """
    try:
        return await services.store.recent_turns(conversation)
    except Exception:  # a conversation feature must not break answering
        logger.exception("could not read conversation history; answering without it")
        return []


#: Shape of a server-issued identifier: url-safe base64, 24 bytes.
_ISSUED = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


def _looks_issued(session_id: str) -> bool:
    """Reject anything that did not come from ``token_urlsafe``.

    Not a security control by itself — the ids are random enough that guessing
    one is infeasible — but it stops a client inventing "session-1" and
    colliding with another visitor by accident.
    """
    return bool(_ISSUED.match(session_id))


def _answer_event(services: Services, response: Answer | Refusal) -> AnswerEvent:
    if isinstance(response, Refusal):
        return AnswerEvent(
            text=response.text,
            refused=True,
            refusal_reason=response.reason.value,
            nearest_topics=list(response.nearest_topics),
        )
    return AnswerEvent(
        text=response.text,
        citations=[
            CitationView(
                label=c.label,
                document_title=c.document_title,
                section_heading=c.section_heading,
                page_start=c.page_start,
                page_end=c.page_end,
                slug=services.slugs.get(str(c.document_id), ""),
            )
            for c in response.citations
        ],
        confidence=response.confidence,
        degraded=response.degraded,
    )


def _caller(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind a proxy the socket address is the proxy, so the forwarded header is
    used when present. That header is client-controlled and therefore spoofable
    — acceptable here because rate limiting is a courtesy control, not a
    security boundary, and the real one is at the edge.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = create_app()
