"""Request and response shapes.

These are the contract the widget is written against, and the thing
``openapi-typescript`` generates client types from, so changes here are
breaking changes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: A question longer than this is not a question. The cap bounds both the
#: embedding cost and the prompt-injection surface.
MAX_QUESTION_CHARS = 1200


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Issued by the server on the first turn and echoed back on later "
            "ones. Conversation turns are held server-side and are never "
            "accepted from a client."
        ),
    )


class CitationView(BaseModel):
    """A citation as the widget renders it.

    Every field has been verified: the label resolved to a block that was
    actually retrieved, and any quoted span really appears in it.
    """

    label: str
    document_title: str
    section_heading: str
    page_start: int
    page_end: int
    slug: str = ""


class SessionEvent(BaseModel):
    """The conversation this exchange belongs to.

    Sent first, so a client that disconnects mid-answer can still continue the
    conversation on its next request.

    The identifier is generated here rather than accepted from the request. A
    client-chosen one could be guessed or reused, and would then hand one
    visitor another visitor's questions as the context for resolving a
    follow-up.
    """

    session_id: str


class StageEvent(BaseModel):
    """One step of the pipeline, as it happens.

    This is what gets streamed instead of tokens. It shows the visitor the
    system doing the work that makes it trustworthy — a `verifying` event
    reporting two citation issues, followed by a regeneration, conveys more
    than any amount of prose.
    """

    stage: str
    detail: str = ""
    elapsed_ms: int = 0


class AnswerEvent(BaseModel):
    text: str
    citations: list[CitationView] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    nearest_topics: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    degraded: bool = False
    """A component was unavailable — currently reranking. Reported rather than
    hidden, so a lower-quality answer is identifiable as one."""


class ErrorEvent(BaseModel):
    """Only for failures that occur *during* a stream.

    Anything knowable before the response begins returns a real HTTP status
    instead, which is far easier for a client to handle.
    """

    code: str
    message: str
    retryable: bool = False


class PaperView(BaseModel):
    slug: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    verified: bool = False


class CorpusInfo(BaseModel):
    """What the assistant covers, so the widget can say so without guessing."""

    documents: int
    chunks: int
    years: list[int] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: dict[str, str] = Field(default_factory=dict)
