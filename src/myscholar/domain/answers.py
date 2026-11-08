"""Answers, citations and refusals: the response side of the domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from myscholar.domain.enums import RefusalReason, Stage


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved reference from an answer to a span of source.

    Built only for labels present in the retrieved context.
    """

    label: str
    document_id: UUID
    section_id: UUID
    document_title: str
    section_heading: str
    page_start: int
    page_end: int
    quote: str | None = None

    source_url: str = ""
    """A DOI for a paper, otherwise the page itself; empty when there is neither."""


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One step of the pipeline, streamed to the client."""

    stage: Stage
    detail: str = ""
    elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer. In a generated answer every factual sentence carries a citation."""

    text: str
    citations: tuple[Citation, ...] = ()
    confidence: float = 0.0
    trace: tuple[TraceEvent, ...] = ()
    degraded: bool = False
    """True when the answer is lower quality than usual and should be flagged."""

    @property
    def refused(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Refusal:
    """A decision not to answer. ``nearest_topics`` names what the corpus does cover."""

    reason: RefusalReason
    text: str
    nearest_topics: tuple[str, ...] = ()
    trace: tuple[TraceEvent, ...] = ()

    @property
    def refused(self) -> bool:
        return True


Response = Answer | Refusal


@dataclass(slots=True)
class Budget:
    """Loop bounds (steps, retrieval rounds, deadline) checked against a caller's clock."""

    max_steps: int = 24
    max_retrieval_rounds: int = 3
    deadline_seconds: float = 45.0

    steps_used: int = 0
    retrieval_rounds_used: int = 0
    strategies_tried: set[str] = field(default_factory=set)

    def exhausted(self, elapsed_seconds: float) -> bool:
        return (
            self.steps_used >= self.max_steps
            or self.retrieval_rounds_used >= self.max_retrieval_rounds
            or elapsed_seconds >= self.deadline_seconds
        )
