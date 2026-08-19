"""Invariants a healthy corpus satisfies, checked against the index.

Ingestion is the stage where this project has gone wrong most often, and every
time it went wrong quietly: ligatures dropped, a title read off a publisher
banner, a text layer repeating every line, a year taken from an education date,
an award bound to the wrong date. All of them produced an index that loaded
cleanly, answered plausibly and reported nothing unusual.

The common cause is that nothing compared the output to the input. A document
with eight hundred chunks looks exactly like one with eighty, and a section
labelled `other` looks exactly like one labelled `methods`.

These checks are deliberately not accuracy measurements — there are no
ground-truth section labels to measure against, and inventing some would be a
day's work for one corpus. They are invariants: statements that are true of
every well-formed document here, so a violation means something upstream broke
even when nobody can say what the right answer was.
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

#: Below this a document is probably missing most of its text — a scanned PDF
#: with no text layer, or an extraction that silently returned headers only.
MIN_CHUNKS_PER_PAGE = 2

#: A paper whose sections are mostly unclassified is one the sectionizer failed
#: on, and section kind drives the boost branches and the references exclusion.
MAX_UNCLASSIFIED_SHARE = 0.6

#: Sections that should never reach the index. References are a quarter of a
#: paper and answer nothing.
EXCLUDED_KINDS = frozenset({"references", "acknowledgements"})


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


def check(documents: list[DocumentStats]) -> list[Finding]:
    """Every invariant, over the whole corpus.

    Returns findings rather than raising, because a corpus with one damaged
    paper is still worth serving — the point is that the damage is *reported*
    rather than absorbed.
    """
    findings: list[Finding] = []

    def fail(document: str, message: str) -> None:
        findings.append(Finding(Severity.FAIL, document, message))

    def warn(document: str, message: str) -> None:
        findings.append(Finding(Severity.WARN, document, message))

    if not documents:
        return [Finding(Severity.FAIL, "corpus", "nothing indexed")]

    seen_titles: dict[str, str] = {}
    for d in documents:
        pages = max(d.pages, 1)
        density = d.chunks / pages

        if d.chunks == 0:
            fail(d.title, "indexed with no chunks")
        elif d.kind == "paper" and density > MAX_CHUNKS_PER_PAGE:
            fail(
                d.title,
                f"{d.chunks} chunks over {pages} page(s) — {density:.0f} per page against "
                f"8-20 for a normal paper. Check the PDF's text layer for repeated lines.",
            )
        elif d.kind == "paper" and d.pages > 2 and density < MIN_CHUNKS_PER_PAGE:
            fail(
                d.title,
                f"only {d.chunks} chunks over {pages} pages — most of the text is missing, "
                f"which a scanned PDF with no text layer would produce.",
            )

        if d.empty_chunks:
            fail(d.title, f"{d.empty_chunks} chunk(s) with no text")

        if d.excluded_sections:
            fail(
                d.title,
                f"{d.excluded_sections} reference or acknowledgement section(s) reached the "
                f"index — they are a quarter of a paper and answer nothing.",
            )

        if d.sections and d.unclassified_sections / d.sections > MAX_UNCLASSIFIED_SHARE:
            warn(
                d.title,
                f"{d.unclassified_sections} of {d.sections} sections unclassified — section "
                f"kind drives the boost branches, so this degrades routing silently.",
            )

        if d.kind == "paper" and not d.verified:
            warn(d.title, "bibliographic metadata unverified; excluded from counting questions")

        if d.kind == "paper" and not d.doi:
            warn(d.title, "no DOI, so `sync` cannot check it against the publisher record")

        if not d.has_summary:
            warn(d.title, "no summary, so it is invisible to document selection")

        key = d.title.casefold()
        if key in seen_titles:
            fail(d.title, "a second document has the same title — a re-ingest that did not replace")
        seen_titles[key] = d.title

    return findings
