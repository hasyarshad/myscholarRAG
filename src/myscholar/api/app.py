"""The HTTP service.

``POST /v1/ask`` streams the pipeline's progress and then the verified answer.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myscholar.adapters.persistence.repository import PostgresStore, create_engine
from myscholar.api.limits import (
    SECRET_HEADER,
    Gate,
    OverloadedError,
    RateLimitedError,
    RateLimiter,
    authorised,
)
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
from myscholar.domain.enums import DocumentKind, Stage
from myscholar.query.history import FollowUpResolver, Turn
from myscholar.reasoning.engine import Engine

logger = logging.getLogger(__name__)

SUGGESTIONS = [
    "How are weighting factors tuned in FCS-MPTC for induction motor drive?",
    "How many papers were published in 2023?",
    "What method detects stator winding inter-turn faults?",
    "How has the work on model predictive control for MMCs developed?",
]


@dataclass
class Services:
    """Built once at startup, because model loading is slow."""

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
        # Rewriting a follow-up is a small task, so it uses the utility model.
        resolver=FollowUpResolver(container.utility),
    )

    try:
        # Local development only. A hosted database's serving role has no DDL
        # rights, and `myscholar db migrate` runs separately.
        if settings.database.migrate_on_startup:
            await store.create_schema()
        documents = await store.list_documents()
        services.slugs = {str(d.id): d.slug for d in documents}
        # An index built by a different embedding model returns ranked but meaningless results.
        await store.assert_embedder_matches(container.embedder.model_id)
        services.ready = bool(documents)
        if not documents:
            services.detail["index"] = "no documents indexed"
    except Exception as exc:  # reported through /readyz instead of crashing
        services.detail["startup"] = f"{type(exc).__name__}: {exc}"
        logger.exception("service startup checks failed")

    return services


def create_app(services: Services | None = None, *, origins: list[str] | None = None) -> FastAPI:
    state: dict[str, Services | None] = {"services": services}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        if state["services"] is None:
            state["services"] = await build_services()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="myScholar",
        version="0.1.0",
        description=(
            "Ask about a personal research corpus: papers, CV and project "
            "write-ups. Answers carry verified citations."
        ),
    )

    # Exact origins, no wildcard. CORS only constrains browsers; other clients
    # are stopped by the shared secret that the Next.js proxy adds.
    allowed = origins or (services.settings.api.allowed_origins if services else None)
    if allowed is None:
        allowed = load_settings().api.allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", SECRET_HEADER],
    )

    configured = services.settings if services else load_settings()
    secret = configured.api.shared_secret.get_secret_value() if configured.api.shared_secret else ""

    @app.middleware("http")
    async def require_secret(request: Request, call_next: Any) -> Response:
        """Require the shared secret on ``/v1/`` routes.

        Health probes stay open: the host's monitor has no secret, and a probe
        that can fail authentication misreports liveness. CORS preflight is
        exempt because browsers send OPTIONS without custom headers.
        """
        if request.method == "OPTIONS" or not request.url.path.startswith("/v1/"):
            return await call_next(request)  # type: ignore[no-any-return]
        if not authorised(request.headers.get(SECRET_HEADER), secret):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "unauthorized"}
            )
        return await call_next(request)  # type: ignore[no-any-return]

    limiter = RateLimiter()
    gate = Gate()

    async def get_services() -> Services:
        current = state["services"]
        if current is None:
            current = state["services"] = await build_services()
        return current

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Liveness: answers whenever the process is running."""
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz(response: Response) -> HealthResponse:
        """Readiness: the database answers and the index is non-empty.

        An empty index is the most likely real failure, and a TCP check would not notice it.
        """
        current = await get_services()
        database = await current.store.health()
        counts = await current.store.counts() if database else {}
        indexed = bool(counts.get("chunks"))

        checks = {"database": database, "index": indexed}
        ok = all(checks.values())

        # Startup notes describe the state at boot, so drop them once their check passes.
        notes = {k: v for k, v in current.detail.items() if not (ok and k in checks)}
        response.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ready" if ok else "degraded",
            checks=checks,
            detail={**notes, **{k: str(v) for k, v in counts.items()}},
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
        """What the assistant covers, for the widget to display."""
        current = await get_services()
        # Papers only: the widget renders this as "N papers, YYYY-YYYY".
        documents = [
            d for d in await current.store.list_documents() if d.kind == DocumentKind.PAPER
        ]
        counts = await current.store.counts()
        return CorpusInfo(
            documents=len(documents),
            chunks=counts.get("chunks", 0),
            years=sorted({d.year for d in documents if d.year}, reverse=True),
            suggestions=SUGGESTIONS,
        )

    @app.post("/v1/ask")
    async def ask(payload: AskRequest, request: Request) -> Response:
        """Answer a question, streaming the pipeline's progress.

        Tokens are not streamed: verification runs after generation, so raw
        output could show text that is then rejected.
        """
        current = await get_services()
        caller = _caller(request)

        # Failures known before the stream opens get an HTTP status; later ones
        # are sent as error events.
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
    # Session ids are issued by the server; one without the issued shape is replaced.
    conversation = (
        session_id if session_id and _looks_issued(session_id) else secrets.token_urlsafe(24)
    )
    asked = question

    try:
        async with gate:
            if await request.is_disconnected():
                return

            yield frame("session", SessionEvent(session_id=conversation))

            # Timed from here so elapsed_ms excludes time spent queued at the gate.
            started = time.monotonic()

            yield frame(
                "stage",
                # No detail: the widget already labels this stage.
                StageEvent(stage=Stage.CLASSIFYING.value),
            )

            # A follow-up is expanded once, here, so the engine sees a
            # self-contained question. If that fails, the question is used as asked.
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
    except Exception as exc:  # a mid-stream failure is reported in-band
        logger.exception("answering failed")
        yield frame(
            "error",
            ErrorEvent(code="internal", message=f"{type(exc).__name__}", retryable=True),
        )
        return

    # Recorded only once an answer exists, so an abandoned request leaves no
    # unanswered turn for a later follow-up to resolve against.
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
    """Previous turns, or none if they cannot be read (an unmigrated database, for example)."""
    try:
        return await services.store.recent_turns(conversation)
    except Exception:  # conversation history must not block answering
        logger.exception("could not read conversation history; answering without it")
        return []


#: Shape of a server-issued identifier: url-safe base64, 24 bytes.
_ISSUED = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


def _looks_issued(session_id: str) -> bool:
    """Whether an id has the shape of one from ``token_urlsafe``.

    Not a security control (issued ids are infeasible to guess). It stops a
    client sending an id like "session-1" and colliding with another visitor.
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
                source_url=c.source_url,
            )
            for c in response.citations
        ],
        confidence=response.confidence,
        degraded=response.degraded,
    )


def _caller(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind a proxy the socket address is the proxy's, so the first
    ``X-Forwarded-For`` entry is used when present. Clients can spoof it, which
    is acceptable because the rate limit is a courtesy control; access is
    controlled by the shared secret.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


app = create_app()
