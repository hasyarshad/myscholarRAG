"""Resolving a DOI to the record its publisher deposited.

The DOI is the one bibliographic field that cannot be hallucinated: it is
matched by regex against the literal first page, so it is either there or it
is not. Everything else — title, authors, year, venue — was being read off the
page by a model and then confirmed by hand.

That review turned out to be the weak link. Checked against Crossref, three of
twenty hand-verified years were wrong, so "how many papers in 2023" and "what
was his first paper" were both answering confidently with the wrong number. A
review file pre-filled with guesses asks a person to *spot* an error rather
than to supply the answer, and three slipped through.

Resolving the DOI instead replaces the guess with the record the publisher
deposited. It is not merely more convenient than the review — it is more
accurate, and it needs no one's attention.
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
    """Citations recorded by the registry, or None when it reports none.

    Distinguished from zero on purpose: a genuinely uncited paper and a
    registry that does not track citations are different claims, and only one
    of them can honestly be reported.
    """

    source: str = ""
    """Which registry answered, so a stored record says where it came from."""


class MetadataResolutionError(RuntimeError):
    """The registry could not be reached or returned something unusable."""


class MetadataResolver(ABC):
    """Looks up a DOI in a bibliographic registry."""

    @abstractmethod
    async def resolve(self, doi: str) -> ResolvedMetadata | None:
        """Return the registry's record, or None if the DOI is unknown to it.

        An unknown DOI is a normal outcome — preprints and newly minted DOIs
        take time to appear — and is distinguished from a failure so that a
        caller can fall back to extraction rather than abort a run.
        """

    async def aclose(self) -> None:
        """Release any held connections. A resolver holding none need not act."""
        return
