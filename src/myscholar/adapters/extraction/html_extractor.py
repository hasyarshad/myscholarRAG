"""Extracting a web page into the same shape as a PDF.

The portfolio site holds the CV, projects and roles, which no paper covers. Pages
are fetched from the live, server-rendered site and parsed with the standard
library's HTMLParser.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path

import httpx

from myscholar.adapters.extraction.docling_extractor import normalise
from myscholar.domain.enums import ChunkKind
from myscholar.ports.extractor import (
    DocumentExtractor,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
)

#: Page chrome. Navigation and footers repeat on every page, so indexing them
#: would make every page match queries about the site's structure.
_SKIP = frozenset({"script", "style", "nav", "footer", "header", "noscript", "svg", "form"})

#: Only h1 and h2 begin a section; deeper headings are content within it.
#: Splitting at every heading left "Education" empty and made each degree its
#: own section.
_SECTION_HEADINGS = frozenset({"h1", "h2"})
_SUB_HEADINGS = frozenset({"h3", "h4", "h5", "h6"})
_HEADINGS = _SECTION_HEADINGS | _SUB_HEADINGS
_BLOCKS = frozenset({"p", "li", "td", "dd", "dt", "blockquote", "figcaption"})

#: Boilerplate that survives the structural filter.
_NOISE = re.compile(
    # © is matched separately: it is not a word character, so a trailing \b
    # would never match after it.
    r"^\s*(©"
    r"|(skip to (main )?content|menu|close|toggle|back to top|copyright"
    r"|all rights reserved|cookie|accept|home)\b)",
    re.IGNORECASE,
)

#: Contentless decoration: list ordinals ("01", "02") and bare link labels.
#: Other short strings are kept, because on a CV they are the data: a
#: 24-character minimum discarded 99 of 200 blocks on /cv, including dates,
#: funders and the skills list.
_CHROME = re.compile(r"^(\d{1,3}|doi|pdf|link|url|[ivxlc]+)$", re.IGNORECASE)


def is_content(text: str) -> bool:
    """Whether a block carries anything worth indexing."""
    return bool(re.search(r"\w", text)) and not _CHROME.match(text) and not _NOISE.match(text)


@dataclass
class _Reader(HTMLParser):
    """Collects headings and text blocks in document order."""

    blocks: list[ExtractedBlock] = field(default_factory=list)
    title: str = ""
    page_title: str = ""
    """From the <title> element; an h1 is often a tagline sentence."""

    def __post_init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._tag: str | None = None
        self._buffer: list[str] = []
        self._seen_section_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title" or tag in _HEADINGS or tag in _BLOCKS:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == self._tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._tag:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = normalise(" ".join(self._buffer))
        tag, self._tag, self._buffer = self._tag, None, []

        if tag == "title":
            if not self.page_title:
                self.page_title = text
            return

        if not text or _NOISE.match(text):
            return

        if tag in _SECTION_HEADINGS:
            # The first h1 is the page title, unless a section heading came before it.
            if tag == "h1" and not self.title and not self._seen_section_heading:
                self.title = text
                return
            self._seen_section_heading = True
            self.blocks.append(
                ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1, heading=text)
            )
        elif tag in _SUB_HEADINGS:
            # Kept as content: it names what follows (a degree, role or project).
            self.blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1))
        elif is_content(text):
            self.blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1))


#: A block that is nothing but a date or a date range. The CV puts the date
#: first, with each part of an entry on its own line:
#:
#:     "2023-2025" / "Alberta Innovates Scholarship" / "University of Alberta"
#:
#: Concatenated into a chunk, a date reads as part of the previous entry.
_DATE_ONLY = re.compile(
    r"^\s*((since|from|until)\s+)?"
    # Repeatable, so a month range sharing one year ("Jan-Dec 2022") matches.
    r"((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*[\-–—/]?\s*)*"  # noqa: RUF001
    r"(19|20)\d{2}"
    # The CV separates years with en and em dashes.
    r"([\s\-–—/,]+((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?"  # noqa: RUF001
    r"((19|20)?\d{2}|present|now|date))*"
    r"\s*$",
    re.IGNORECASE,
)


def group_dated_entries(blocks: list[ExtractedBlock]) -> list[ExtractedBlock]:
    """Fold each date-led run of blocks into one block that opens with its date.

    Blocks before the first date are left alone, so prose sections are untouched.
    """
    grouped: list[ExtractedBlock] = []
    entry: list[ExtractedBlock] = []

    def flush() -> None:
        if not entry:
            return
        head, *rest = entry
        parts = [head.text.strip().rstrip(".,"), *(b.text.strip() for b in rest)]
        text = " — ".join(p for p in parts if p)
        # Terminated, because a chunk joins entries with only a space; without a
        # full stop the next entry's date reads as part of this one.
        grouped.append(replace(head, text=text if text.endswith(".") else text + "."))
        entry.clear()

    for block in blocks:
        if block.heading:
            # A heading ends the previous entry and never joins one.
            flush()
            grouped.append(block)
        elif _DATE_ONLY.match(block.text):
            flush()
            entry.append(block)
        elif entry:
            entry.append(block)
        else:
            grouped.append(block)

    flush()
    return grouped


class HtmlExtractor(DocumentExtractor):
    """Fetches a URL and parses it into blocks."""

    def __init__(self, *, timeout: float = 20.0, user_agent: str = "myscholar/0.1") -> None:
        self._timeout = timeout
        self._user_agent = user_agent

    def extract(self, source: Path | str) -> ExtractedDocument:
        url = str(source)
        try:
            response = httpx.get(
                url,
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._user_agent},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"could not fetch {url}: {exc}"
            raise ExtractionError(msg) from exc

        reader = _Reader()
        reader.feed(response.text)
        reader.close()

        reader.blocks = group_dated_entries(reader.blocks)

        if not reader.blocks:
            msg = (
                f"{url} produced no content. If the page renders client-side, "
                f"its text is not in the HTML and a browser would be needed."
            )
            raise ExtractionError(msg)

        # Hashed on the extracted text: build ids and asset hashes in the HTML
        # change on every deploy.
        digest = hashlib.sha256(
            "\n".join(b.text for b in reader.blocks).encode("utf-8")
        ).hexdigest()

        return ExtractedDocument(
            source_path=url,
            content_sha256=digest,
            page_count=1,
            blocks=tuple(reader.blocks),
            first_page_text="\n".join(b.text for b in reader.blocks[:20])[:4000],
            title=_page_title(reader.page_title, reader.title, url),
        )


def _page_title(document_title: str, heading: str, url: str) -> str:
    """A short, recognisable name for the page.

    Prefers <title> over the h1, which on a portfolio is often a tagline sentence.
    """
    if document_title:
        # Strip a site-name suffix: "CV | Site Name" becomes "CV".
        head = re.split(r"\s+[|\u00b7\u2014\u2013\u2022-]\s+", document_title)[0].strip()
        if head:
            return head
    if heading and len(heading) <= 70:
        return heading
    slug = url.rstrip("/").rsplit("/", 1)[-1] or "home"
    return slug.replace("-", " ").title()
