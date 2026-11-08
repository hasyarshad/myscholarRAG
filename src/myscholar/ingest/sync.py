"""Keeping the corpus in step with what has been published.

Re-resolves every indexed DOI against the registry and compares the index with the
publications page. Papers missing from the index are reported, not created from metadata,
since the assistant could not discuss them. Planning is separate from applying, so a plan
can be printed first and a registry outage never leaves the corpus half-rewritten.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from myscholar.ports.metadata import (
    MetadataResolutionError,
    MetadataResolver,
    ResolvedMetadata,
)

logger = logging.getLogger(__name__)

_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def normalise_doi(doi: str) -> str:
    """Lowercased and stripped of the resolver prefix, for comparison only.

    DOIs are case-insensitive but are deposited in mixed case.
    """
    return doi.strip().rstrip(".,;").removeprefix("https://doi.org/").casefold()


def dois_in(text: str) -> set[str]:
    """Every DOI in a blob of text, normalised."""
    return {normalise_doi(match) for match in _DOI.findall(text)}


@dataclass(frozen=True, slots=True)
class FieldChange:
    field: str
    was: object
    now: object

    def __str__(self) -> str:
        return f"{self.field}: {self.was!r} -> {self.now!r}"


@dataclass(frozen=True, slots=True)
class Correction:
    """What the registry says should change about one indexed document."""

    slug: str
    doi: str
    title: str
    changes: tuple[FieldChange, ...]
    resolved: ResolvedMetadata

    @property
    def fields(self) -> dict[str, object]:
        return {change.field: change.now for change in self.changes}


@dataclass(slots=True)
class SyncPlan:
    """Everything sync intends to do, before it does any of it."""

    corrections: list[Correction] = field(default_factory=list)
    unknown_to_registry: list[str] = field(default_factory=list)
    """DOIs the registry does not know, which is normal for a fresh deposit."""

    without_doi: list[str] = field(default_factory=list)
    """Indexed documents carrying no DOI, which cannot be corrected at all."""

    unreachable: list[tuple[str, str]] = field(default_factory=list)
    """Lookups that failed. Their documents are left unchanged."""

    missing_from_index: list[str] = field(default_factory=list)
    """Published but not searchable: a paper listed on the site with no PDF."""

    missing_from_site: list[str] = field(default_factory=list)
    """Searchable but not published, usually because the page needs updating."""

    @property
    def is_empty(self) -> bool:
        return not (self.corrections or self.missing_from_index or self.missing_from_site)


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """The bibliographic view of an indexed row.

    Decoupled from the ORM so planning can be tested without a database.
    """

    slug: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    doi: str | None
    cited_by: int | None = None


def _changes(document: IndexedDocument, resolved: ResolvedMetadata) -> tuple[FieldChange, ...]:
    """Fields where the registry disagrees, ignoring absences.

    A record that omits a field does not assert it is empty, so an existing value is
    never overwritten with None.
    """
    candidates: list[tuple[str, object, object]] = [
        ("title", document.title, resolved.title),
        ("authors", list(document.authors), list(resolved.authors)),
        ("year", document.year, resolved.year),
        ("venue", document.venue, resolved.venue),
        ("cited_by", document.cited_by, resolved.cited_by),
    ]

    changes: list[FieldChange] = []
    for name, was, now in candidates:
        if now is None or now == [] or now == "":
            continue
        if name == "title" and _same_title(str(was), str(now)):
            continue
        if was != now:
            changes.append(FieldChange(field=name, was=was, now=now))
    return tuple(changes)


_TITLE_NOISE = re.compile(r"[^a-z0-9]+")


def _same_title(left: str, right: str) -> bool:
    """Whether two titles differ only in punctuation, case or spacing.

    Publishers deposit "Two -Phase" where the PDF sets "Two-Phase", and rewriting a
    title also changes its slug.
    """
    return _TITLE_NOISE.sub("", left.casefold()) == _TITLE_NOISE.sub("", right.casefold())


async def plan(
    documents: Sequence[IndexedDocument],
    *,
    resolver: MetadataResolver,
    published_dois: Iterable[str] = (),
) -> SyncPlan:
    """Work out what sync would change, without changing anything."""
    result = SyncPlan()
    indexed_dois: set[str] = set()

    for document in documents:
        if not document.doi:
            result.without_doi.append(document.slug)
            continue

        indexed_dois.add(normalise_doi(document.doi))
        try:
            resolved = await resolver.resolve(document.doi)
        except MetadataResolutionError as exc:
            # Left untouched: an outage says nothing about whether the paper exists.
            result.unreachable.append((document.doi, str(exc)))
            logger.warning("could not resolve %s: %s", document.doi, exc)
            continue

        if resolved is None:
            result.unknown_to_registry.append(document.doi)
            continue

        if changes := _changes(document, resolved):
            result.corrections.append(
                Correction(
                    slug=document.slug,
                    doi=document.doi,
                    title=document.title,
                    changes=changes,
                    resolved=resolved,
                )
            )

    published = {normalise_doi(d) for d in published_dois}
    if published:
        result.missing_from_index = sorted(published - indexed_dois)
        result.missing_from_site = sorted(indexed_dois - published)

    return result
