"""Documents, sections, chunks and context blocks."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from myscholar.domain.enums import ChunkKind, DocumentKind, SectionKind


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Bibliographic facts for one document.

    ``verified`` is set once the record is resolved against Crossref or confirmed by
    hand. Only verified records answer bibliographic questions.
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

    ``text`` is verbatim source, used for generation and quote verification. Only
    ``embed_text`` (``context_header`` naming the paper and section, plus the text) is
    embedded, so a sentence that names neither its topic nor its paper is still found.
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
    """A contiguous span of chunks given to the model as one citable unit.

    ``label`` (C1, C2, ...) is what the model cites and the citation verifier resolves.
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

    Rendered into the prompt so the model does not credit the corpus owner with
    methods a review only summarises.
    """

    authors: tuple[str, ...] = ()
    """Who wrote the source document.

    Rendered into the prompt because related-work passages name other researchers
    and rarely their own authors.
    """

    document_kind: str = ""
    """Paper, CV or site page, so the interface can say what was searched."""

    source_url: str = ""
    """Where a reader can check the claim: a DOI, or the page itself.

    Never rendered into the prompt, so the model cannot write out a link; it points
    at sources only through verified citation markers.
    """

    def render(self, *, owner: str = "") -> str:
        """Format for inclusion in a prompt.

        ``owner`` is the corpus owner's name. When given, each excerpt with known
        authors is marked as theirs or not.
        """
        review = " [REVIEW: describes other researchers' work]" if self.is_review else ""
        byline = ""
        if self.authors:
            surname = owner.split()[-1].lower() if owner else ""
            mine = surname and any(surname in a.lower() for a in self.authors)
            whose = "BY THE AUTHOR" if mine else "NOT BY THE AUTHOR"
            byline = f" [{whose}: {', '.join(self.authors[:4])}]"
        return (
            f"[{self.label}]{review}{byline} {self.document_title} — {self.section_heading} "
            f"(p. {self.page_start}-{self.page_end})\n{self.text}"
        )
