"""Answering questions about the collection with SQL.

"How many papers did he publish in 2023?" has no answer inside any paper — it
is a property of the collection. Retrieval nonetheless returns chunks that
mention 2023 (a copyright line, a citation, a results table), and a model handed
those will state a number. It will be wrong and it will sound certain.

So these questions never reach the vector index. The count comes from the
database, and the model's only job is to phrase a result it was handed.

Only verified rows are used. Answering from unverified extraction would swap one
confident-but-wrong source for another.

See docs/learning/04-query-translation-and-routing.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from myscholar.adapters.persistence.repository import DocumentRow, PostgresStore


class BibliographicIntent(StrEnum):
    COUNT = "count"
    LIST = "list"
    MOST_RECENT = "most_recent"
    BY_YEAR = "by_year"
    VENUES = "venues"


_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

_INTENT_PATTERNS: tuple[tuple[re.Pattern[str], BibliographicIntent], ...] = (
    (
        re.compile(r"\bmost\s+recent|\blatest\b|\bnewest\b", re.IGNORECASE),
        BibliographicIntent.MOST_RECENT,
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
    """A result computed from the database, with the facts it was built from."""

    text: str
    intent: BibliographicIntent
    documents: tuple[DocumentRow, ...] = ()
    unverified: int = 0
    """Documents excluded because their metadata is not yet verified."""

    facts: dict[str, object] = field(default_factory=dict)
    """The raw numbers, so a generator can phrase them without recomputing."""


def detect_intent(question: str) -> BibliographicIntent:
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(question):
            return intent
    return BibliographicIntent.LIST


def extract_years(question: str) -> list[int]:
    return sorted({int(y) for y in _YEAR.findall(question)})


class BibliographicQuery:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    async def answer(self, question: str) -> BibliographicAnswer:
        documents = await self._store.list_documents()
        verified = [d for d in documents if d.verified]
        excluded = len(documents) - len(verified)

        if not verified:
            return BibliographicAnswer(
                text=(
                    "I can't answer questions about the publication list yet: none of the "
                    "bibliographic metadata has been verified, and I won't guess at it."
                ),
                intent=BibliographicIntent.COUNT,
                unverified=excluded,
            )

        intent = detect_intent(question)
        years = extract_years(question)
        scoped = [d for d in verified if d.year in years] if years else verified

        match intent:
            case BibliographicIntent.COUNT:
                return self._count(scoped, years, excluded)
            case BibliographicIntent.MOST_RECENT:
                return self._most_recent(verified, excluded)
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
            facts={"count": count, "years": years},
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
            facts={"count": len(ordered), "titles": [d.title for d in ordered]},
        )

    def _most_recent(self, documents: list[DocumentRow], excluded: int) -> BibliographicAnswer:
        dated = [d for d in documents if d.year is not None]
        if not dated:
            return BibliographicAnswer(
                text="None of the papers has a verified publication year.",
                intent=BibliographicIntent.MOST_RECENT,
                unverified=excluded,
            )
        newest_year = max(d.year or 0 for d in dated)
        newest = [d for d in dated if d.year == newest_year]
        listed = "\n".join(f"· {d.title}" + (f" ({d.venue})" if d.venue else "") for d in newest)
        return BibliographicAnswer(
            text=f"The most recent work is from {newest_year}:\n{listed}",
            intent=BibliographicIntent.MOST_RECENT,
            documents=tuple(newest),
            unverified=excluded,
            facts={"year": newest_year, "count": len(newest)},
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
            facts={"by_year": counts},
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
            facts={"venues": counts},
        )
