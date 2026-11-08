"""Resolving a DOI to the record its publisher deposited.

The DOI is matched by regex on the first page, so it cannot be hallucinated. The
registry record is more accurate than model-extracted fields checked by hand:
against Crossref, three of twenty hand-verified years were wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedMetadata:
    """A bibliographic record from an authoritative registry."""

    doi: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    cited_by: int | None = None
    """Citations recorded by the registry. None means the registry reported no count,
    which is different from zero citations."""

    source: str = ""
    """The registry that answered."""


class MetadataResolutionError(RuntimeError):
    """The registry could not be reached or returned something unusable."""


class MetadataResolver(ABC):
    """Looks up a DOI in a bibliographic registry."""

    @abstractmethod
    async def resolve(self, doi: str) -> ResolvedMetadata | None:
        """Return the registry's record, or None if the DOI is unknown to it.

        An unknown DOI is normal (preprints and new DOIs take time to appear) and
        is distinct from a failure, so the caller can fall back to extraction.
        """

    async def aclose(self) -> None:
        """Release any held connections. The default holds none."""
        return
