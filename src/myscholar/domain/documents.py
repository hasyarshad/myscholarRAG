"""Documents, sections and chunks — the ingestion side of the domain."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from myscholar.domain.enums import ChunkKind, DocumentKind, SectionKind


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Bibliographic facts, human-verified during ingestion.

    ``verified`` matters: once a human has confirmed these, bibliographic
    questions are answered from this record by SQL rather than by asking a
    language model to recall them. That removes an entire class of confident
    fabrication for the cost of about ten minutes of review.
    """

    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    verified: bool = False

    @property
    def citation_label(self) -> str:
        """Short human-readable label, e.g. 'Arshad et al., 2023'."""
        if not self.authors:
            stem = self.title[:40]
        elif len(self.authors) == 1:
            stem = self.authors[0]
        else:
            stem = f"{self.authors[0]} et al."
        return f"{stem}, {self.year}" if self.year else stem


@dataclass(frozen=True, slots=True)
class Document:
    id: UUID
    slug: str
    kind: DocumentKind
    metadata: PaperMetadata
    source_path: str
    content_sha256: str
    page_count: int = 0


@dataclass(frozen=True, slots=True)
class Section:
    id: UUID
    document_id: UUID
    ordinal: int
    kind: SectionKind
    heading: str
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class Chunk:
    """The unit that gets embedded and retrieved.

    Three text fields, deliberately distinct:

    - ``text`` is verbatim source, used for generation and for verifying that
      quoted spans are real.
    - ``context_header`` names the paper and section.
    - ``embed_text`` is the header plus the text, and is the *only* one that
      gets embedded.

    The separation is what lets a chunk reading "the proposed method reduces
    THD by 3.1%" be retrievable for "what has he achieved on harmonic
    distortion?" — the bare sentence names neither the topic nor the paper.
    """

    id: UUID
    document_id: UUID
    section_id: UUID
    ordinal_in_section: int
    kind: ChunkKind
    text: str
    context_header: str
    token_count: int
    page: int

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n\n{self.text}" if self.context_header else self.text


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """A contiguous span of chunks handed to the model as one citable unit.

    Produced by parent expansion: retrieval finds small precise chunks, then
    neighbours within the same section are merged so the model sees coherent
    prose rather than isolated fragments. ``label`` (C1, C2, ...) is what the
    model cites and what the citation verifier resolves.
    """

    label: str
    document_id: UUID
    section_id: UUID
    document_title: str
    section_heading: str
    text: str
    page_start: int
    page_end: int
    score: float = 0.0

    is_review: bool = False
    """True when the source paper surveys other researchers' work.

    Rendered into the prompt because a passage cannot show it. A survey's
    methods section describes what other people built, and without this the
    assistant credited Haseeb with inventing GGAP-RBF — an algorithm his review
    merely summarised. Attributing a stranger's work to the corpus owner is the
    worst error a portfolio assistant can make, and it reads as authoritative.
    """

    document_kind: str = ""
    """Paper, CV or site page.

    Carried so the interface can say what was actually searched. The widget
    used to label every retrieval "Searching the papers", which was untrue for
    any question the CV answered — and it is the CV questions where a visitor
    is most likely to be checking whether the thing works.
    """

    def render(self) -> str:
        """Format for inclusion in a prompt."""
        review = " [REVIEW: describes other researchers' work]" if self.is_review else ""
        return (
            f"[{self.label}]{review} {self.document_title} — {self.section_heading} "
            f"(p. {self.page_start}-{self.page_end})\n{self.text}"
        )
