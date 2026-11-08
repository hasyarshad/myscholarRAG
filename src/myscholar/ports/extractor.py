"""The document extraction interface.

Turns a document into headings, prose, tables and equations, each tagged with its
page, because citations resolve to page numbers.
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
    """Set only when this block is itself a heading."""

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
    """Used for bibliographic metadata extraction (authors, venue, DOI)."""

    title: str = ""
    """The document's title, kept out of the block stream.

    As a heading it would open a section spanning the abstract and author block.
    """

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def headings(self) -> tuple[str, ...]:
        return tuple(b.heading for b in self.blocks if b.heading is not None)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, source: Path | str) -> ExtractedDocument:
        """Parse a document from a file path or a URL.

        A URL stays a string: ``Path("https://x.com/y")`` collapses the double slash.
        """


class ExtractionError(RuntimeError):
    """A document could not be parsed."""
