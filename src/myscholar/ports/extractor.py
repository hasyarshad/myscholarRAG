"""The document extraction interface.

Turns a PDF into structured text: a heading hierarchy, blocks of prose, tables
and equations, each tagged with the page it came from. Page provenance matters
because citations resolve to a page number.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from myscholar.domain.enums import ChunkKind


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    """One contiguous piece of a document, as the extractor found it."""

    kind: ChunkKind
    text: str
    page: int
    heading: str | None = None
    """Set when this block *is* a heading, rather than content beneath one."""

    @property
    def is_heading(self) -> bool:
        return self.heading is not None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A parsed document, before sectioning and chunking."""

    source_path: Path | str
    """A filesystem path for documents, a URL for pages."""

    content_sha256: str
    page_count: int
    blocks: tuple[ExtractedBlock, ...]
    first_page_text: str
    """Used for bibliographic metadata extraction — authors, venue, DOI."""

    title: str = ""
    """The document's own title, as the extractor labelled it.

    Captured separately rather than left in the block stream: treating it as a
    section heading produces a section spanning the abstract and author block,
    headed by the entire paper title.
    """

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def headings(self) -> tuple[str, ...]:
        return tuple(b.heading for b in self.blocks if b.heading is not None)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, source: Path | str) -> ExtractedDocument:
        """Parse a document from a file path or a URL.

        The union is deliberate. Squeezing a URL into a ``Path`` looks tidy and
        silently corrupts it — ``Path("https://x.com/y")`` normalises the double
        slash away and produces ``https:/x.com/y``.
        """


class ExtractionError(RuntimeError):
    """A document could not be parsed."""
