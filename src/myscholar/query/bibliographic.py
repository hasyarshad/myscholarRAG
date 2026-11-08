"""Answering questions about the publication list with SQL over verified metadata.

No passage states how many papers appeared in 2023. Given chunks that mention the year,
a model states a wrong count, so these questions never reach the vector index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from myscholar.adapters.persistence.repository import DocumentRow, PostgresStore
from myscholar.domain.enums import DocumentKind


class BibliographicIntent(StrEnum):
    COUNT = "count"
    LIST = "list"
    MOST_RECENT = "most_recent"
    EARLIEST = "earliest"
    BY_YEAR = "by_year"
    VENUES = "venues"


_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], BibliographicIntent], ...] = (
    (
        # A bare "recent" counts: classification has already established that the
        # question is about the collection.
        re.compile(r"\brecent\b|\blatest\b|\bnewest\b", re.IGNORECASE),
        BibliographicIntent.MOST_RECENT,
    ),
    (
        re.compile(r"\boldest\b|\bearliest\b|\bfirst\s+(paper|publication|work)", re.IGNORECASE),
        BibliographicIntent.EARLIEST,
    ),
    (
        re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b", re.IGNORECASE),
        BibliographicIntent.COUNT,
    ),
    (re.compile(r"\b(venue|journal|conference)s?\b", re.IGNORECASE), BibliographicIntent.VENUES),
    (
        re.compile(r"\bby\s+year\b|\bper\s+year\b|\beach\s+year\b|\bover\s+time\b", re.IGNORECASE),
        BibliographicIntent.BY_YEAR,
    ),
    (
        re.compile(
            r"\blist\b|\bshow\b|\bwhat\s+(papers|publications|works)\b|\bwhich\s+papers\b",
            re.IGNORECASE,
        ),
        BibliographicIntent.LIST,
    ),
)


@dataclass(frozen=True, slots=True)
class BibliographicAnswer:
    """A result computed from the database."""

    text: str
    intent: BibliographicIntent
    documents: tuple[DocumentRow, ...] = ()
    unverified: int = 0
    """Documents excluded because their metadata is not yet verified."""


def detect_intent(question: str) -> BibliographicIntent | None:
    """What the question wants from the metadata, or None if nothing does.

    None sends the question to retrieval. A LIST default answered "which of his
    papers covers wind turbine pitch control?" with every paper in the corpus.
    """
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(question):
            return intent
    return None


def extract_years(question: str) -> list[int]:
    return sorted({int(y) for y in _YEAR.findall(question)})


class BibliographicQuery:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def answer(self, question: str) -> BibliographicAnswer | None:
        """The metadata answer, or None if the question does not want one."""
        if (intent := detect_intent(question)) is None:
            return None

        # Papers only: the CV and site pages are indexed but are not publications.
        papers = [d for d in await self._store.list_documents() if d.kind == DocumentKind.PAPER]
        verified = [d for d in papers if d.verified]
        excluded = len(papers) - len(verified)

        if not verified:
            return BibliographicAnswer(
                text=(
                    "I can't answer questions about the publication list yet: none of the "
                    "bibliographic metadata has been verified, and I won't guess at it."
                ),
                intent=BibliographicIntent.COUNT,
                unverified=excluded,
            )

        years = extract_years(question)
        scoped = [d for d in verified if d.year in years] if years else verified

        match intent:
            case BibliographicIntent.COUNT:
                return self._count(scoped, years, excluded)
            case BibliographicIntent.MOST_RECENT:
                return self._by_recency(verified, excluded, newest=True)
            case BibliographicIntent.EARLIEST:
                return self._by_recency(verified, excluded, newest=False)
            case BibliographicIntent.BY_YEAR:
                return self._by_year(verified, excluded)
            case BibliographicIntent.VENUES:
                return self._venues(scoped, excluded)
            case BibliographicIntent.LIST:
                return self._list(scoped, years, excluded)

    @staticmethod
    def _plural(count: int) -> str:
        return "paper" if count == 1 else "papers"

    def _count(
        self, documents: list[DocumentRow], years: list[int], excluded: int
    ) -> BibliographicAnswer:
        count = len(documents)
        where = f" in {', '.join(str(y) for y in years)}" if years else ""
        return BibliographicAnswer(
            text=f"{count} {self._plural(count)}{where}.",
            intent=BibliographicIntent.COUNT,
            documents=tuple(documents),
            unverified=excluded,
        )

    def _list(
        self, documents: list[DocumentRow], years: list[int], excluded: int
    ) -> BibliographicAnswer:
        ordered = sorted(documents, key=lambda d: (-(d.year or 0), d.title))
        where = f" from {', '.join(str(y) for y in years)}" if years else ""
        lines = [
            f"{d.year or '—'} · {d.title}" + (f" ({d.venue})" if d.venue else "") for d in ordered
        ]
        return BibliographicAnswer(
            text=f"{len(ordered)} {self._plural(len(ordered))}{where}:\n" + "\n".join(lines),
            intent=BibliographicIntent.LIST,
            documents=tuple(ordered),
            unverified=excluded,
        )

    def _by_recency(
        self, documents: list[DocumentRow], excluded: int, *, newest: bool
    ) -> BibliographicAnswer:
        intent = BibliographicIntent.MOST_RECENT if newest else BibliographicIntent.EARLIEST
        dated = [d for d in documents if d.year is not None]
        if not dated:
            return BibliographicAnswer(
                text="None of the papers has a verified publication year.",
                intent=intent,
                unverified=excluded,
            )
        target = (max if newest else min)(d.year or 0 for d in dated)
        matching = [d for d in dated if d.year == target]
        listed = "\n".join(f"· {d.title}" + (f" ({d.venue})" if d.venue else "") for d in matching)
        label = "most recent" if newest else "earliest"
        return BibliographicAnswer(
            text=f"The {label} work is from {target}:\n{listed}",
            intent=intent,
            documents=tuple(matching),
            unverified=excluded,
        )

    def _by_year(self, documents: list[DocumentRow], excluded: int) -> BibliographicAnswer:
        counts: dict[int, int] = {}
        for document in documents:
            if document.year is not None:
                counts[document.year] = counts.get(document.year, 0) + 1
        lines = [f"{year}: {count}" for year, count in sorted(counts.items(), reverse=True)]
        return BibliographicAnswer(
            text="Publications by year:\n" + "\n".join(lines),
            intent=BibliographicIntent.BY_YEAR,
            documents=tuple(documents),
            unverified=excluded,
        )

    def _venues(self, documents: list[DocumentRow], excluded: int) -> BibliographicAnswer:
        counts: dict[str, int] = {}
        for document in documents:
            if document.venue:
                counts[document.venue] = counts.get(document.venue, 0) + 1
        if not counts:
            return BibliographicAnswer(
                text="No verified venue information is recorded.",
                intent=BibliographicIntent.VENUES,
                unverified=excluded,
            )
        lines = [
            f"{venue} ({count})" if count > 1 else venue
            for venue, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return BibliographicAnswer(
            text="Published in:\n" + "\n".join(f"· {line}" for line in lines),
            intent=BibliographicIntent.VENUES,
            documents=tuple(documents),
            unverified=excluded,
        )
