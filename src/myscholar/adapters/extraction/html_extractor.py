"""Extracting a web page into the same shape as a PDF.

The portfolio site is the other half of what a visitor might ask about — the
CV, the projects, the roles. None of it is in any paper, so without this the
assistant correctly but unhelpfully refuses every biographical question.

Fetched from the live site rather than read from the source repository, on the
grounds that the published page is what a visitor actually sees, needs no
credentials, and re-runs whenever the site changes. That only works because the
pages are server-rendered; a client-rendered site would need a browser.

Deliberately hand-rolled rather than pulling in a parser. The requirement is
narrow — headings, paragraphs and list items from one known site — and the
standard library's HTMLParser covers it in about a hundred lines.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
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

#: Elements whose text is chrome rather than content. Navigation and footers
#: repeat on every page, so indexing them makes every page match every query
#: about the site's own structure.
_SKIP = frozenset({"script", "style", "nav", "footer", "header", "noscript", "svg", "form"})

#: Only h1 and h2 begin a section. Deeper headings become content within the
#: one they sit under.
#:
#: Treating every heading as a section boundary shredded the CV: "Education"
#: became an empty section and each degree beneath it became its own, so
#: "MSc, Electrical Engineering: Control & Systems" was classified as a methods
#: section. Grouping by the level that actually delimits content keeps the
#: degrees inside Education, where a question about them will find them.
_SECTION_HEADINGS = frozenset({"h1", "h2"})
_SUB_HEADINGS = frozenset({"h3", "h4", "h5", "h6"})
_HEADINGS = _SECTION_HEADINGS | _SUB_HEADINGS
_BLOCKS = frozenset({"p", "li", "td", "dd", "dt", "blockquote", "figcaption"})

#: Boilerplate that survives the structural filter.
_NOISE = re.compile(
    # The copyright sign is matched separately: it is not a word character, so
    # the trailing \b never fired after it and "© 2026" was let through. Only
    # ever visible in a footer, which is skipped by tag — but the rule was
    # silently doing nothing.
    r"^\s*(©"
    r"|(skip to (main )?content|menu|close|toggle|back to top|copyright"
    r"|all rights reserved|cookie|accept|home)\b)",
    re.IGNORECASE,
)

#: Decoration that is genuinely contentless: the ordinals a styled list prints
#: beside each entry ("01", "02"), and bare link labels.
#:
#: This replaced a 24-character minimum, which was written for prose pages and
#: is actively wrong for a CV. Measured on /cv, the length rule discarded 99 of
#: 200 blocks — every date range, every funder, every award amount, and the
#: entire skills list down to "Python", "PyTorch" and "NSGA-II". The page
#: looked indexed and answered biographical questions plausibly, which is what
#: kept it hidden.
#:
#: Short is not the same as empty. On a CV the short strings *are* the data,
#: and the structural filters above — skipped tags, then this — are what
#: separate content from chrome.
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
    """From the <title> element — a real title, unlike an h1 that is often a
    tagline sentence."""

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
            # The first h1 is the page's own heading, not a section within it —
            # unless a section heading has already been seen, in which case the
            # page has an unusual structure and it is better treated as one.
            if tag == "h1" and not self.title and not self._seen_section_heading:
                self.title = text
                return
            self._seen_section_heading = True
            self.blocks.append(
                ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1, heading=text)
            )
        elif tag in _SUB_HEADINGS:
            # Content, not a boundary. Kept because it names what follows —
            # a degree, a role, a project — which the surrounding prose often
            # does not repeat.
            self.blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1))
        elif is_content(text):
            self.blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=1))


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

        if not reader.blocks:
            msg = (
                f"{url} produced no content. If the page renders client-side, "
                f"its text is not in the HTML and a browser would be needed."
            )
            raise ExtractionError(msg)

        # Hashed on the extracted text, not the raw HTML: build ids and asset
        # hashes change on every deploy, and hashing those would re-ingest an
        # unchanged page every time.
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

    The <title> element is preferred over the h1: a portfolio h1 is frequently
    a tagline sentence, and using one produced a document titled "I work on
    electric drives. Finding faults before they spread, and keeping control
    when the model is wrong." — which then appeared as the source of every
    citation to that page.
    """
    if document_title:
        # Strip a site-name suffix: "CV — Haseeb Arshad" becomes "CV".
        head = re.split(r"\s+[|\u00b7\u2014\u2013\u2022-]\s+", document_title)[0].strip()
        if head:
            return head
    if heading and len(heading) <= 70:
        return heading
    slug = url.rstrip("/").rsplit("/", 1)[-1] or "home"
    return slug.replace("-", " ").title()
