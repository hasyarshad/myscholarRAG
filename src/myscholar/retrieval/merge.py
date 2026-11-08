"""Expanding retrieved chunks back into contiguous spans.

Chunks are small so their vectors stay specific; the generator needs coherent prose, so
neighbouring chunks are merged after retrieval. Spans are keyed on ``section_id``, so a
span never joins two sections or two papers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Span:
    """A contiguous run of chunk ordinals within one section."""

    section_id: UUID
    document_id: UUID
    start: int
    end: int
    """Inclusive."""

    seed_ids: frozenset[UUID]
    """The retrieved chunks this span grew from, for provenance."""

    max_score: float
    """The best score among the seeds, used to order spans for the generator."""

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def contains(self, ordinal: int) -> bool:
        return self.start <= ordinal <= self.end


@dataclass(frozen=True, slots=True)
class SeedHit:
    """What merging needs from a retrieval hit."""

    chunk_id: UUID
    document_id: UUID
    section_id: UUID
    ordinal: int
    score: float


def merge_windows(
    hits: Sequence[SeedHit],
    *,
    window: int = 2,
    max_chunks: int = 8,
    section_lengths: dict[UUID, int] | None = None,
    promote_ratio: float = 0.5,
) -> list[Span]:
    """Expand each hit by ``window`` chunks either side, then merge overlaps.

    ``section_lengths`` maps a section to its chunk count, used to clamp spans
    to the section and to decide promotion. Without it, spans are clamped only
    at zero and promotion is skipped.

    ``promote_ratio``: when a merged span already covers this fraction of its
    section, it is extended to the whole section. Half a Methods section reads
    worse than all of it, and the token cost is similar.

    ``max_chunks`` caps a span's length so a cluster of hits in one dense
    section cannot swallow the whole context budget. Truncation is centred on
    the highest-scoring seed, since that is the chunk retrieval was most
    confident about.

    Returns spans ordered by descending ``max_score``, so the generator sees the
    strongest evidence first.
    """
    if not hits:
        return []

    section_lengths = section_lengths or {}

    # Expand each hit to a window, clamped to the section.
    intervals: list[tuple[UUID, UUID, int, int, UUID, float]] = []
    for hit in hits:
        start = max(0, hit.ordinal - window)
        end = hit.ordinal + window
        if (length := section_lengths.get(hit.section_id)) is not None:
            end = min(end, length - 1)
        if end < start:
            end = start
        intervals.append((hit.section_id, hit.document_id, start, end, hit.chunk_id, hit.score))

    # Sort by (section, start) and sweep once. Intervals from different sections are
    # never adjacent in this order, so a merge cannot span two sections.
    intervals.sort(key=lambda i: (str(i[0]), i[2], i[3]))

    merged: list[Span] = []
    for section_id, document_id, start, end, chunk_id, score in intervals:
        previous = merged[-1] if merged else None
        overlaps = (
            previous is not None
            and previous.section_id == section_id
            # Touching windows merge too: chunks 0-2 and 3-5 are contiguous text.
            and start <= previous.end + 1
        )
        if overlaps and previous is not None:
            merged[-1] = Span(
                section_id=section_id,
                document_id=document_id,
                start=previous.start,
                end=max(previous.end, end),
                seed_ids=previous.seed_ids | {chunk_id},
                max_score=max(previous.max_score, score),
            )
        else:
            merged.append(
                Span(
                    section_id=section_id,
                    document_id=document_id,
                    start=start,
                    end=end,
                    seed_ids=frozenset({chunk_id}),
                    max_score=score,
                )
            )

    scores = {hit.chunk_id: hit.score for hit in hits}
    finished = [
        _cap(_promote(span, section_lengths, promote_ratio), max_chunks, scores, hits)
        for span in merged
    ]
    finished.sort(key=lambda s: -s.max_score)
    return finished


def _promote(span: Span, section_lengths: dict[UUID, int], ratio: float) -> Span:
    """Extend a span to its whole section once it already covers most of it."""
    length = section_lengths.get(span.section_id)
    if length is None or length <= 0 or span.length / length < ratio:
        return span
    return Span(
        section_id=span.section_id,
        document_id=span.document_id,
        start=0,
        end=length - 1,
        seed_ids=span.seed_ids,
        max_score=span.max_score,
    )


def _cap(
    span: Span,
    max_chunks: int,
    scores: dict[UUID, float],
    hits: Sequence[SeedHit],
) -> Span:
    """Truncate an over-long span, centred on its best seed."""
    if span.length <= max_chunks:
        return span

    seeds_in_span = [h for h in hits if h.chunk_id in span.seed_ids]
    best = max(seeds_in_span, key=lambda h: scores.get(h.chunk_id, 0.0), default=None)
    centre = best.ordinal if best is not None else (span.start + span.end) // 2

    half = max_chunks // 2
    start = max(span.start, centre - half)
    end = min(span.end, start + max_chunks - 1)
    start = max(span.start, end - max_chunks + 1)

    return Span(
        section_id=span.section_id,
        document_id=span.document_id,
        start=start,
        end=end,
        # Only seeds that survived the truncation remain provenance for this span.
        seed_ids=frozenset(h.chunk_id for h in seeds_in_span if start <= h.ordinal <= end)
        or span.seed_ids,
        max_score=span.max_score,
    )
