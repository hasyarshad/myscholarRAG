"""Request and response shapes: the contract the widget is written against."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Bounds the embedding cost and the prompt-injection surface.
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
    """A verified citation: the label resolved to a retrieved block containing any quoted span."""

    label: str
    document_title: str
    section_heading: str
    page_start: int
    page_end: int
    slug: str = ""

    source_url: str = ""
    """A DOI for a paper, otherwise the page itself; empty when there is neither."""


class SessionEvent(BaseModel):
    """The conversation this exchange belongs to. The identifier is issued by the server.

    Sent first, so a client that disconnects mid-answer can still continue the
    conversation on its next request.
    """

    session_id: str


class StageEvent(BaseModel):
    """One pipeline stage, streamed as it happens in place of tokens."""

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
    """The answer is known to be lower quality than usual."""


class ErrorEvent(BaseModel):
    """A failure during the stream. Failures known before it opens get an HTTP status instead."""

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
    """What the assistant covers, for the widget to display."""

    documents: int
    chunks: int
    years: list[int] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: dict[str, str] = Field(default_factory=dict)
