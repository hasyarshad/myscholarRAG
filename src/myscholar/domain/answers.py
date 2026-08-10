"""Answers, citations and refusals — the response side of the domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from myscholar.domain.enums import RefusalReason, Stage


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved reference from an answer back to a specific span of source.

    Every field here has been verified to exist: the block label was present in
    the retrieved context, and any quoted text is a real substring of it. An
    unverifiable citation is rejected before the answer is returned rather than
    displayed with a caveat.
    """

    label: str
    document_id: UUID
    section_id: UUID
    document_title: str
    section_heading: str
    page_start: int
    page_end: int
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One step of the pipeline, streamed to the client.

    These are what make the adaptive loop visible in the UI — the user sees
    "retrieving (attempt 2)" because the grader genuinely rejected round one.
    """

    stage: Stage
    detail: str = ""
    elapsed_ms: int = 0
    round_number: int = 1


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer. Every factual sentence carries at least one citation."""

    text: str
    citations: tuple[Citation, ...] = ()
    confidence: float = 0.0
    trace: tuple[TraceEvent, ...] = ()
    degraded: bool = False
    """True when a component was skipped, e.g. reranking unavailable in production."""

    @property
    def refused(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Refusal:
    """A deliberate decision not to answer.

    ``nearest_topics`` is what separates a useful refusal from a dead end: the
    bot says what it *does* cover instead of only what it doesn't.
    """

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
    """Bounds that guarantee the adaptive loop terminates.

    Three independent limits, any of which forces a refusal. Without these a
    self-correcting loop can retry indefinitely on a question it will never be
    able to answer.
    """

    max_steps: int = 24
    max_retrieval_rounds: int = 3
    max_regenerations: int = 2
    deadline_seconds: float = 45.0

    steps_used: int = 0
    retrieval_rounds_used: int = 0
    regenerations_used: int = 0
    strategies_tried: set[str] = field(default_factory=set)

    def exhausted(self, elapsed_seconds: float) -> bool:
        return (
            self.steps_used >= self.max_steps
            or self.retrieval_rounds_used >= self.max_retrieval_rounds
            or elapsed_seconds >= self.deadline_seconds
        )
