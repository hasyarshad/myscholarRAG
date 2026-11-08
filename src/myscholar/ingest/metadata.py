"""Bibliographic metadata: extract a proposal, then have a person confirm it.

Questions such as "how many papers did he publish in 2023?" have no single passage to
retrieve, so they are answered with SQL over verified columns. Extraction writes a YAML
proposal, and nothing is marked verified until that file is reviewed and re-imported.
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

    The publication year repeats in the header, footer and copyright line, while years in
    citations vary.
    """
    years = [int(y) for y in _YEAR.findall(text)]
    return max(set(years), key=years.count) if years else None


def slugify(title: str, fallback: str, *, taken: set[str] | None = None) -> str:
    """Build a URL-safe identifier, unique against ``taken``.

    The slug is a unique key, so a collision would fail the whole document.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (title or fallback).lower()).strip("-")[:120]
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-")[:120] or "document"

    if taken is None or slug not in taken:
        return slug

    # Try the fallback (the filename stem) before a counter, since it stays readable.
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

    source_path: Path | str
    metadata: PaperMetadata
    confidence_notes: tuple[str, ...] = ()


class MetadataExtractor:
    def __init__(self, model: ChatModel | None = None) -> None:
        self._model = model

    async def propose(
        self,
        *,
        first_page: str,
        source_path: Path | str,
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
            except Exception as exc:  # noqa: BLE001 - fall back to the filename
                notes.append(f"model extraction failed ({type(exc).__name__}); using filename")

        # Prefer the extractor's typeset title to the model's reading and to the
        # filename, which is often lowercased and truncated.
        fallback = Path(str(source_path)).stem
        title = extracted_title or (extracted.title if extracted else "") or fallback
        # For DOI and year, a regex match on the page text is preferred to the
        # model, which can invent both.
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
    """Write proposals to YAML for human review, least confident first."""
    payload = [
        {
            "file": Path(str(p.source_path)).name,
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
        "# Only verified entries are used to answer bibliographic questions.\n"
        "# Entries with a _check key had something the extractor was unsure about;\n"
        "# they are listed first. Delete _check once reviewed.\n"
        "#\n"
        "# Keep values quoted. Titles here contain colons, and an unquoted value\n"
        '# containing ": " is read by YAML as a nested mapping.\n\n'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Quote every scalar: an unquoted title containing ": " parses as a mapping.
    body = yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, default_style='"', width=100
    )
    path.write_text(header + body, "utf-8")


class MetadataFileError(RuntimeError):
    """The review file could not be read.

    Raised in place of a raw YAML error, with advice on the usual cause.
    """


def read_review_file(path: Path) -> dict[str, PaperMetadata]:
    """Read a reviewed YAML file, keyed by filename."""
    try:
        raw: Any = yaml.safe_load(path.read_text("utf-8")) or []
    except yaml.YAMLError as exc:
        # Usually a title containing ": ", which YAML reads as a nested mapping
        # unless the value is quoted.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        line = ""
        if mark is not None:
            lines = path.read_text("utf-8").splitlines()
            if mark.line < len(lines):
                line = f"\n\n    {lines[mark.line].strip()}"
        msg = (
            f"{path} is not valid YAML{where}.{line}\n\n"
            f"The usual cause is a value containing a colon followed by a space, "
            f"such as a title like 'Revisiting media literacy: a new scale'. "
            f"Wrap the whole value in double quotes:\n\n"
            f'    title: "Revisiting media literacy: a new scale"'
        )
        raise MetadataFileError(msg) from exc

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
