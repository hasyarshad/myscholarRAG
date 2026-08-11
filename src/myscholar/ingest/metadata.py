"""Bibliographic metadata: extract, then have a human confirm it.

Questions like "how many papers did he publish in 2023?" or "what venue was the
NSGA-II paper in?" are the ones naive RAG answers most confidently and most
wrongly — there is no single passage stating the answer, so the model composes
one from fragments.

The fix is not a better prompt. It is to answer those questions with SQL over
columns a human has checked. For twenty papers that review costs about ten
minutes, once, and removes the entire failure class.

So extraction produces a *proposal*, written to YAML. Nothing is marked verified
until the file has been reviewed and re-imported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from myscholar.domain.documents import PaperMetadata
from myscholar.ports.llm import ChatModel, Message, Role

_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


class ExtractedMetadata(BaseModel):
    """Schema handed to the model, so decoding cannot produce malformed output."""

    title: str = Field(description="Full paper title, without trailing punctuation")
    authors: list[str] = Field(default_factory=list, description="Author names in order")
    year: int | None = Field(default=None, ge=1980, le=2100)
    venue: str | None = Field(default=None, description="Journal or conference name")
    doi: str | None = None


_PROMPT = """Extract bibliographic details from the first page of an academic paper.

Rules:
- Use only what appears in the text. Never guess a venue or year.
- Authors in the order printed, as "F. Surname".
- The venue is the journal or conference, not the publisher.
- Leave a field null when it is genuinely absent.

First page:
---
{page}
---"""


def find_doi(text: str) -> str | None:
    match = _DOI.search(text)
    return match.group(0).rstrip(".,;") if match else None


def find_year(text: str) -> int | None:
    """The most frequent plausible year on the page.

    Publication years repeat — in the header, the footer, the copyright line —
    whereas years inside citations vary. Frequency is a better signal than
    position.
    """
    years = [int(y) for y in _YEAR.findall(text)]
    return max(set(years), key=years.count) if years else None


def slugify(title: str, fallback: str, *, taken: set[str] | None = None) -> str:
    """Build a URL-safe identifier, unique against ``taken``.

    Uniqueness matters because the slug is a unique key. Two papers from the
    same publisher can extract to the same title if extraction goes wrong, and
    a collision then fails the whole document rather than degrading — which is
    exactly what happened on the first full run.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (title or fallback).lower()).strip("-")[:120]
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-")[:120] or "document"

    if taken is None or slug not in taken:
        return slug

    # Disambiguate with the fallback (the filename stem) before resorting to
    # a counter, since that at least stays human-readable.
    from_fallback = re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-")[:120]
    if from_fallback and from_fallback not in taken:
        return from_fallback

    for suffix in range(2, 100):
        candidate = f"{slug[:115]}-{suffix}"
        if candidate not in taken:
            return candidate
    return f"{slug[:110]}-{abs(hash(fallback)) % 100000}"


@dataclass(frozen=True, slots=True)
class MetadataProposal:
    """An extraction result, awaiting human confirmation."""

    source_path: Path
    metadata: PaperMetadata
    confidence_notes: tuple[str, ...] = ()


class MetadataExtractor:
    def __init__(self, model: ChatModel | None = None) -> None:
        self._model = model

    async def propose(
        self,
        *,
        first_page: str,
        source_path: Path,
        extracted_title: str = "",
    ) -> MetadataProposal:
        notes: list[str] = []
        regex_doi = find_doi(first_page)
        regex_year = find_year(first_page)

        extracted: ExtractedMetadata | None = None
        if self._model is not None:
            try:
                extracted = await self._model.structured(
                    [Message(role=Role.USER, content=_PROMPT.format(page=first_page[:3500]))],
                    ExtractedMetadata,
                )
            except Exception as exc:  # noqa: BLE001 - fall back rather than abort a run
                notes.append(f"model extraction failed ({type(exc).__name__}); using filename")

        # The extractor's title item is the typeset title, so it beats both the
        # model's reading of the page and the filename. Filenames in this corpus
        # are lowercased and truncated by the download tool.
        title = extracted_title or (extracted.title if extracted else "") or source_path.stem
        # Regex findings win over the model's for DOI and year: a pattern match
        # against the literal page text cannot hallucinate, and both fields are
        # exactly the sort a model will confidently invent.
        doi = regex_doi or (extracted.doi if extracted else None)
        year = regex_year or (extracted.year if extracted else None)

        if extracted and extracted.year and regex_year and extracted.year != regex_year:
            notes.append(f"year disagreement: model said {extracted.year}, page shows {regex_year}")
        if not year:
            notes.append("no year found")
        if not (extracted and extracted.authors):
            notes.append("no authors extracted")

        return MetadataProposal(
            source_path=source_path,
            metadata=PaperMetadata(
                title=title.strip(),
                authors=tuple(extracted.authors) if extracted else (),
                year=year,
                venue=(extracted.venue if extracted else None),
                doi=doi,
                verified=False,
            ),
            confidence_notes=tuple(notes),
        )


def write_review_file(proposals: list[MetadataProposal], path: Path) -> None:
    """Write proposals to YAML for human review.

    Ordered with the least confident first, so the entries most likely to be
    wrong are the ones read while attention is freshest.
    """
    payload = [
        {
            "file": p.source_path.name,
            "title": p.metadata.title,
            "authors": list(p.metadata.authors),
            "year": p.metadata.year,
            "venue": p.metadata.venue,
            "doi": p.metadata.doi,
            "verified": False,
            **({"_check": list(p.confidence_notes)} if p.confidence_notes else {}),
        }
        for p in sorted(proposals, key=lambda p: (-len(p.confidence_notes), p.metadata.title))
    ]

    header = (
        "# Extracted bibliographic metadata, awaiting review.\n"
        "#\n"
        "# Correct anything wrong, then set verified: true on each entry and run\n"
        "#   myscholar papers import-metadata\n"
        "#\n"
        "# Only verified entries are used to answer bibliographic questions, which\n"
        "# are otherwise the ones a language model gets confidently wrong.\n"
        "# Entries with a _check key had something the extractor was unsure about;\n"
        "# they are listed first. Delete _check once reviewed.\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")


def read_review_file(path: Path) -> dict[str, PaperMetadata]:
    """Read a reviewed YAML file, keyed by filename."""
    raw: Any = yaml.safe_load(path.read_text("utf-8")) or []
    result: dict[str, PaperMetadata] = {}
    for entry in raw:
        if not isinstance(entry, dict) or "file" not in entry:
            continue
        result[str(entry["file"])] = PaperMetadata(
            title=str(entry.get("title") or "").strip(),
            authors=tuple(entry.get("authors") or ()),
            year=entry.get("year"),
            venue=entry.get("venue"),
            doi=entry.get("doi"),
            verified=bool(entry.get("verified", False)),
        )
    return result
