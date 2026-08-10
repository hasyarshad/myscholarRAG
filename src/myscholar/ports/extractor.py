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

    source_path: Path
    content_sha256: str
    page_count: int
    blocks: tuple[ExtractedBlock, ...]
    first_page_text: str
    """Used for bibliographic metadata extraction — title, authors, venue, DOI."""

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def headings(self) -> tuple[str, ...]:
        return tuple(b.heading for b in self.blocks if b.heading is not None)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, path: Path) -> ExtractedDocument:
        """Parse a document. Raises ``ExtractionError`` if it cannot be read."""


class ExtractionError(RuntimeError):
    """A document could not be parsed."""
