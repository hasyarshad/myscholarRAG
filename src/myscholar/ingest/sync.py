"""Keeping the corpus in step with what has actually been published.

The trigger is a push: Haseeb drops an accepted paper into ``papers/`` and
commits it. Everything downstream of that should follow without anyone editing
a YAML file, because the manual step was not merely tedious — it was the least
accurate part of the pipeline. Three of twenty reviewed years were wrong, so
"how many papers in 2023" and "what was his first paper" were both answering
confidently with the wrong number.

Three things happen here, and they are separated because they fail
independently:

1. **Correction.** Every indexed document with a DOI is re-resolved against
   the registry, and differences are written back. This is what repairs a
   corpus that is already wrong, and it costs no re-embedding — bibliographic
   fields are columns, not vectors.
2. **Reconciliation.** The publications page is the list of what exists; the
   index is the list of what is searchable. The difference is reported rather
   than filled in, because a document with metadata and no text would let the
   assistant claim a paper it cannot discuss.
3. **Ingestion** of anything new, which the pipeline already handles.

Planning is separated from applying throughout. A registry outage during a
sync must not leave the corpus half-rewritten, and a plan that can be printed
before it runs is one a person can sanity-check.
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

#: Fields a registry may correct. ``doi`` is absent on purpose: it is the key
#: the lookup was made with, so a registry cannot be a source of truth about
#: it without circularity.
CORRECTABLE = ("title", "authors", "year", "venue", "cited_by")


def normalise_doi(doi: str) -> str:
    """Lowercased and stripped of the resolver prefix, for comparison only.

    DOIs are case-insensitive by specification but are deposited in mixed
    case, so the same paper appears as ``10.1109/TASE...`` on one page and
    ``10.1109/tase...`` on another. Comparing them raw reports a corpus of
    twenty as having forty distinct papers.
    """
    return doi.strip().rstrip(".,;").removeprefix("https://doi.org/").casefold()


def dois_in(text: str) -> set[str]:
    """Every DOI in a blob of text, normalised.

    Used against the rendered publications page, which is HTML rather than a
    feed. That is coarse but robust: a DOI has a distinctive shape, and it
    survives the page being restyled, which a CSS selector would not.
    """
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
    """DOIs the registry has never heard of — normal for a fresh deposit."""

    without_doi: list[str] = field(default_factory=list)
    """Indexed documents carrying no DOI, which cannot be corrected at all."""

    unreachable: list[tuple[str, str]] = field(default_factory=list)
    """Lookups that failed. Their documents are left exactly as they were."""

    missing_from_index: list[str] = field(default_factory=list)
    """Published but not searchable — a paper listed on the site with no PDF."""

    missing_from_site: list[str] = field(default_factory=list)
    """Searchable but not published — usually a page that needs updating."""

    @property
    def is_empty(self) -> bool:
        return not (self.corrections or self.missing_from_index or self.missing_from_site)


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """The bibliographic view of an indexed row, decoupled from the ORM.

    Planning takes this rather than a ``Document`` so it can be exercised
    without a database, which is most of what makes the reconciliation
    testable.
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

    A registry that omits a venue is not asserting there isn't one, so None is
    never written over a value that is already present. Dropping a known venue
    because a record was thin would be a regression disguised as a correction.
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
    """Titles differing only in punctuation, case or spacing are the same title.

    Publishers deposit "Two -Phase" where the PDF sets "Two-Phase", and IEEE
    capitalises inconsistently between the two. Rewriting a title over that is
    churn, and worse, it changes the slug — which is a unique key.
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
            # Left untouched rather than dropped from the corpus. A registry
            # being down says nothing about whether the paper exists.
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
