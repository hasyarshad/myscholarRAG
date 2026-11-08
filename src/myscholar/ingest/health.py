"""Invariants a healthy corpus satisfies, checked against the index.

There are no ground-truth section labels, so these are structural checks: a violation
means ingestion broke something upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    FAIL = "fail"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class Finding:
    severity: Severity
    document: str
    message: str


#: Chunks per page beyond which a text layer is probably repeating itself.
#: Healthy papers here run 8-20; the worst produced 132.
MAX_CHUNKS_PER_PAGE = 40

#: Below this a document is probably missing most of its text (a scanned PDF with
#: no text layer, or an extraction that returned headers only).
MIN_CHUNKS_PER_PAGE = 2

#: A paper whose sections are mostly unclassified is one the sectionizer failed
#: on, and section kind drives the boost branches and the references exclusion.
MAX_UNCLASSIFIED_SHARE = 0.6


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """What the index holds for one document."""

    title: str
    kind: str
    pages: int
    chunks: int
    sections: int
    unclassified_sections: int
    excluded_sections: int
    empty_chunks: int
    has_summary: bool
    verified: bool
    doi: str | None


def check(documents: list[DocumentStats], *, model_ids: set[str] | None = None) -> list[Finding]:
    """Check every invariant over the whole corpus.

    Returns findings instead of raising, since a corpus with one damaged paper can still
    be served.
    """
    findings: list[Finding] = []

    def fail(document: str, message: str) -> None:
        findings.append(Finding(Severity.FAIL, document, message))

    def warn(document: str, message: str) -> None:
        findings.append(Finding(Severity.WARN, document, message))

    if not documents:
        return [Finding(Severity.FAIL, "corpus", "nothing indexed")]

    if model_ids and len(model_ids) > 1:
        # One HNSW index covers every model's vectors and is filtered by model only
        # after the scan, so a second model halves the candidate pool (it cost 0.07
        # MRR). Fix by dropping the unused model or giving each a partial index.
        fail(
            "corpus",
            f"{len(model_ids)} embedding models indexed ({', '.join(sorted(model_ids))}). "
            f"They share one HNSW index, so each degrades the other's recall. "
            f"Keep one: DELETE FROM chunk_embeddings WHERE model_id = '<unused>'",
        )

    seen_titles: dict[str, str] = {}
    for d in documents:
        pages = max(d.pages, 1)
        density = d.chunks / pages

        if d.chunks == 0:
            fail(d.title, "indexed with no chunks")
        elif d.kind == "paper" and density > MAX_CHUNKS_PER_PAGE:
            fail(
                d.title,
                f"{d.chunks} chunks over {pages} page(s), {density:.0f} per page against "
                f"8-20 for a normal paper. Check the PDF's text layer for repeated lines.",
            )
        elif d.kind == "paper" and d.pages > 2 and density < MIN_CHUNKS_PER_PAGE:
            fail(
                d.title,
                f"only {d.chunks} chunks over {pages} pages; most of the text is missing, "
                f"as from a scanned PDF with no text layer.",
            )

        if d.empty_chunks:
            fail(d.title, f"{d.empty_chunks} chunk(s) with no text")

        if d.excluded_sections:
            fail(
                d.title,
                f"{d.excluded_sections} reference or acknowledgement section(s) reached the "
                f"index, where they answer nothing.",
            )

        if d.sections and d.unclassified_sections / d.sections > MAX_UNCLASSIFIED_SHARE:
            warn(
                d.title,
                f"{d.unclassified_sections} of {d.sections} sections unclassified; section "
                f"kind drives the boost branches, so this degrades routing.",
            )

        if d.kind == "paper" and not d.verified:
            warn(d.title, "bibliographic metadata unverified; excluded from counting questions")

        if d.kind == "paper" and not d.doi:
            warn(d.title, "no DOI, so `sync` cannot check it against the publisher record")

        if not d.has_summary:
            warn(d.title, "no summary, so it is invisible to document selection")

        key = d.title.casefold()
        if key in seen_titles:
            fail(d.title, "a second document has the same title: a re-ingest that did not replace")
        seen_titles[key] = d.title

    return findings
