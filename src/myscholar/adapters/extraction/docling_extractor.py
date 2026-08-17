"""Docling-based PDF extraction.

Every configuration choice here was measured rather than assumed; the evidence
is in docs/learning/01b-extraction-findings.md. The two that matter:

- **The pypdfium backend is not optional.** Docling's default parser silently
  drops ``fl`` and ``fi`` ligatures, turning "flux" into " ux" — 51 corrupted
  occurrences against 10 intact ones in a single paper about motor flux
  control. Extraction still reports success, which is what makes it dangerous.

- **Formula enrichment is on**, so equations become LaTeX rather than
  ``<!-- formula-not-decoded -->`` placeholders. The LaTeX is stored but
  excluded from embeddings later, since it is token soup that matches no
  natural-language question.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

from myscholar.domain.enums import ChunkKind
from myscholar.ports.extractor import (
    DocumentExtractor,
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
)

logger = logging.getLogger(__name__)

#: Labels that start a new section.
#:
#: ``title`` is deliberately excluded. Docling labels the paper's own title as a
#: title item, and treating it as a section heading produces a section whose
#: heading is the entire paper title, swallowing the abstract and author block —
#: which then gets classified by content and filed as, say, methods. The title
#: is metadata, not a section, and is captured separately.
_HEADING_LABELS = frozenset({"section_header"})

#: Labels carrying no answerable content — running heads, page numbers, and
#: image placeholders.
_SKIP_LABELS = frozenset({"page_footer", "page_header", "picture"})

#: The document title, captured for metadata rather than treated as content.
_TITLE_LABELS = frozenset({"title"})

#: Shapes that mark a genuine section heading rather than a paper title.
#:
#: Length is not a usable discriminator: measured across the corpus, titles run
#: 79-108 characters while real headings such as "Q. Adaptive Forgetting Factor
#: with Generalized Regularization Online Sequential ..." run longer still.
#: Shape is reliable, because a paper title never opens with an enumerator or a
#: canonical section word.
_SECTION_HEADING_SHAPE = re.compile(
    r"""^\s*(
        # Numeric enumeration: "3.1 Results", "4. Methods". Trailing
        # punctuation is optional because both forms occur.
        \d+(?:\.\d+)*[.):\-]?\s+
        # Roman or single-letter enumeration: "III. METHODOLOGY", "B) Results".
        # Punctuation is required here, otherwise a title beginning "A Chaos
        # Based ..." or "I Propose ..." would match.
        |(?:[IVXLC]+|[A-Z])[.):\-]\s+
        # Canonical section words. No trailing \b, so plurals and derived
        # forms ("REFERENCES", "Methodology", "Experiments") match too.
        |(?:abstract|introduction|keyword|index\s+terms|background|
           related\s+work|literature\s+review|methodolog|method|
           experiment|result|discussion|conclusion|reference|
           bibliograph|acknowledg|appendix|nomenclature|
           impact\s+statement|ccs\s+concepts)
    )""",
    re.IGNORECASE | re.VERBOSE,
)


#: Publisher furniture that appears above the title on the first page.
#: Elsevier, Frontiers and IEEE templates all put a banner there, and taking
#: the first heading-like item produced five documents titled "ScienceDirect",
#: "ARTICLE", "ORIGINAL RESEARCH" and "Research paper".
_PUBLISHER_BANNER = re.compile(
    r"^\s*(sciencedirect|elsevier|springer|wiley|mdpi|frontiers|ieee|acm|"
    r"article|original\s+research|research\s+(paper|article)|review\s+article|"
    r"contents\s+lists|journal\s+homepage|open\s+access|"
    r"regular\s+paper|full\s+length\s+article)\b",
    re.IGNORECASE,
)

#: Shorter than this and it is furniture, not a title. Measured across the
#: corpus, real titles run 79-108 characters; the shortest plausible academic
#: title is still well above this floor.
_MIN_TITLE_CHARS = 30


def looks_like_section_heading(text: str) -> bool:
    """Whether a heading-like item is a section heading rather than a title."""
    return bool(_SECTION_HEADING_SHAPE.match(text))


def is_plausible_title(text: str) -> bool:
    """Whether a heading-like item could be the paper's title."""
    return (
        len(text) >= _MIN_TITLE_CHARS
        and not _PUBLISHER_BANNER.match(text)
        and not looks_like_section_heading(text)
    )


_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")

#: Shortest repeat worth removing.
#:
#: Long enough that ordinary English cannot reach it by accident. "had had" and
#: "that that" are seven characters; a duplicated title or affiliation line runs
#: to fifty.
_MIN_REPEAT_CHARS = 20

#: How far back to look for the original. The stacked layer interleaves with
#: the real one, so a duplicate follows its original within a few items.
_REPEAT_WINDOW = 6

_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _dedup_key(text: str) -> str:
    """Punctuation-free, lowercased. The duplicate layer differs only in these.

    IFAC's template stacks two text layers, and the second has its punctuation
    stripped: "Surveillance, reconnaissance, infrastructure inspection" is
    followed by "Surveillancereconnaissanceinfrastructure inspection". Compared
    literally the two are different strings; normalised, they are identical.
    """
    return _ALPHANUMERIC.sub("", text.casefold())


def collapse_repeated_lines(text: str, seen: set[str] | None = None) -> str:
    """Drop lines already seen, across the whole document by default.

    The damage is *inside* a block, not between blocks. One block in the UAV
    paper runs to 49,588 characters and is a single abstract accumulating line
    by line:

        Abstract: This paper proposes a linear quadratic Gaussian control ...
        Abstract: This paper proposes a linear quadratic Gaussian control ...
        a quadrotor UAV ...
        Abstract: This paper proposes a linear quadratic Gaussian control ...

    Deduplication spans the document rather than the block, because the copies
    are not confined to one: consecutive blocks are growing prefixes of the
    same text, so no two are identical and every one is mostly repeat.

    That is safe here, and it was measured rather than assumed. Across the
    twenty papers, nineteen contain 0-3% repeated lines — a figure caption
    restated, a running header — and the UAV paper contains 52%. A rule that
    removes three percent from healthy documents and half of a broken one is
    hitting what it aims at.

    Working at line granularity keeps the Phase 1 caution intact. No line is
    ever edited, so every surviving line is verbatim source and a quoted span
    still resolves; only whole duplicate lines are removed. English does not
    repeat a twenty-character line, whereas it repeats "had had" all the time —
    which is why the earlier attempt at a self-referential regex over the prose
    failed and this does not.
    """
    seen = set() if seen is None else seen
    kept: list[str] = []
    for line in text.split("\n"):
        key = _dedup_key(line)
        if len(key) >= _MIN_REPEAT_CHARS:
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)
    return "\n".join(kept)


def drop_repeated(blocks: list[ExtractedBlock]) -> tuple[list[ExtractedBlock], int]:
    """Remove blocks that restate a nearby one.

    Phase 1 deliberately left this alone, on the grounds that the artefact was
    cosmetic and that a self-referential regex over prose would be a
    correctness bug in the very text citations are verified against. The first
    half of that was wrong: measured in Phase 5, the worst-affected paper
    produced 796 chunks from 6 pages against ~12 per page elsewhere, and its
    garbled fragments were crowding out real content in unrelated searches.

    The second half still stands, which is why this compares *whole blocks*
    rather than rewriting inside one. A block is dropped only when another
    block nearby says the same thing — no prose is ever edited, so a citation
    can still be verified against text that appears exactly as extracted.
    """
    kept: list[ExtractedBlock] = []
    seen: dict[str, int] = {}  # key -> index in kept
    recent: list[tuple[str, int]] = []  # (key, index in kept), for containment
    seen_lines: set[str] = set()
    dropped = 0

    for original in blocks:
        # Tables are left alone: a column of repeated values is data, not an
        # artefact, and the table was exported as one unit already.
        block = original
        if block.kind is not ChunkKind.TABLE:
            collapsed = collapse_repeated_lines(block.text, seen_lines)
            if not collapsed.strip():
                # Everything in it had already been said elsewhere.
                dropped += 1
                continue
            if collapsed != block.text:
                block = replace(block, text=collapsed)

        key = _dedup_key(block.text)
        if len(key) < _MIN_REPEAT_CHARS:
            kept.append(block)
            continue

        # Exact repeats are matched across the whole document, not a window.
        # The stacked layer is not merely adjacent: the same paragraph
        # reappears dozens of times at every distance, and a six-block window
        # caught 94 of them out of several thousand.
        duplicate_of = seen.get(key)

        # Containment stays windowed. Globally it would drop a short block that
        # happens to be a substring of an unrelated one — a heading inside a
        # paragraph quoting it — which is a real loss rather than an artefact.
        if duplicate_of is None:
            for other_key, index in recent:
                if key in other_key or other_key in key:
                    duplicate_of = index
                    break

        if duplicate_of is None:
            seen[key] = len(kept)
            recent.append((key, len(kept)))
            kept.append(block)
        else:
            dropped += 1
            # Keep whichever copy is longer: the truncated one is the artefact.
            if len(block.text) > len(kept[duplicate_of].text):
                kept[duplicate_of] = block

        del recent[:-_REPEAT_WINDOW]

    return kept, dropped


def normalise(text: str) -> str:
    """Repair the text defects this corpus actually exhibits.

    NFKC decomposes ligature codepoints such as U+FB02 (``ﬂ``) into ``fl``.
    Different PDF backends disagree about whether to emit the ligature as one
    character or two, and an index containing both forms would fail to match
    either consistently.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphens, invisible but tokenised
    text = text.replace("​", "")  # zero-width spaces
    text = _WHITESPACE.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DoclingExtractor(DocumentExtractor):
    def __init__(self, *, enrich_formulas: bool = True, ocr: bool = False) -> None:
        # OCR defaults off: these are born-digital publisher PDFs with a real
        # text layer. Running OCR over them costs minutes per document and
        # introduces recognition errors into text that was already correct.
        self._enrich_formulas = enrich_formulas
        self._ocr = ocr
        self._converter: Any | None = None

    def _build_converter(self) -> Any:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = self._ocr
        options.do_table_structure = True
        options.generate_page_images = False
        options.do_formula_enrichment = self._enrich_formulas

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    @property
    def converter(self) -> Any:
        """Built once and reused — model loading dominates the first call."""
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    def extract(self, source: Path | str) -> ExtractedDocument:
        path = Path(source)
        if not path.is_file():
            msg = f"no such file: {path}"
            raise ExtractionError(msg)

        try:
            document = self.converter.convert(str(path)).document
        except Exception as exc:
            msg = f"docling could not parse {path.name}: {type(exc).__name__}: {exc}"
            raise ExtractionError(msg) from exc

        blocks: list[ExtractedBlock] = []
        warnings: list[str] = []
        title = ""
        seen_section_heading = False

        for item, _level in document.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            text = normalise(getattr(item, "text", "") or "")
            if not text or any(skip in label for skip in _SKIP_LABELS):
                continue

            page = self._page_of(item)

            is_title_label = any(t in label for t in _TITLE_LABELS)
            is_heading_label = any(head in label for head in _HEADING_LABELS)

            # Docling is inconsistent about the paper's own title: measured
            # across this corpus, only one paper in five carries a title item,
            # and the rest label the title as a section_header. Label alone is
            # therefore not enough.
            #
            # The reliable signal is position plus shape: the first heading-like
            # item, before any content, that does not look like a section
            # heading, is the paper title.
            # Keyed on "no section heading yet" rather than "no content yet":
            # several papers open with a journal banner or an author block
            # before the title, and requiring an empty block list missed those.
            looks_like_title = (
                not title
                and is_heading_label
                and page <= 1
                and not seen_section_heading
                and is_plausible_title(text)
            )

            # Furniture above the title — a banner, a correspondence line — is
            # neither a title nor a section. Treating it as a section heading
            # set seen_section_heading and blocked the real title that follows,
            # which is why five papers ended up titled "ScienceDirect".
            is_furniture = (
                is_heading_label
                and page <= 1
                and not seen_section_heading
                and not is_plausible_title(text)
                and not looks_like_section_heading(text)
            )

            if is_furniture:
                blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page))
            elif (is_title_label and is_plausible_title(text)) or looks_like_title:
                # The first such item is the paper's title; later ones are
                # figure or table titles, which belong with the surrounding
                # prose rather than replacing the document title.
                if not title:
                    title = text
                else:
                    blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page))
            elif is_heading_label:
                # Several papers repeat the title as a running heading. Once it
                # has been claimed as the title, later copies are not sections.
                if title and text.casefold() == title.casefold():
                    continue
                seen_section_heading = True
                blocks.append(
                    ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page, heading=text)
                )
            elif "formula" in label or "equation" in label:
                blocks.append(ExtractedBlock(kind=ChunkKind.EQUATION, text=text, page=page))
            elif "caption" in label:
                blocks.append(ExtractedBlock(kind=ChunkKind.CAPTION, text=text, page=page))
            else:
                blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page))

        for index, table in enumerate(getattr(document, "tables", []) or []):
            markdown = self._table_markdown(table, document)
            if not markdown:
                warnings.append(f"table {index + 1} produced no recoverable structure")
                continue
            blocks.append(
                ExtractedBlock(
                    kind=ChunkKind.TABLE,
                    text=normalise(markdown),
                    page=self._page_of(table),
                )
            )

        blocks, repeated = drop_repeated(blocks)
        if repeated:
            warnings.append(f"dropped {repeated} block(s) repeating a nearby one")

        if not blocks:
            msg = f"{path.name} parsed but yielded no content"
            raise ExtractionError(msg)

        first_page = "\n".join(b.text for b in blocks if b.page <= 1)[:4000]
        if not first_page:
            first_page = "\n".join(b.text for b in blocks[:15])[:4000]

        if not title:
            warnings.append("no title item found; falling back to the filename")

        return ExtractedDocument(
            source_path=path,
            content_sha256=_sha256(path),
            page_count=len(getattr(document, "pages", {}) or {}),
            blocks=tuple(blocks),
            first_page_text=first_page,
            title=title,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _page_of(item: Any) -> int:
        """Best-effort page number; 0 when provenance is missing."""
        provenance = getattr(item, "prov", None) or []
        for entry in provenance:
            page = getattr(entry, "page_no", None)
            if page is not None:
                return int(page)
        return 0

    @staticmethod
    def _table_markdown(table: Any, document: Any) -> str:
        try:
            return str(table.export_to_markdown(document))
        except (AttributeError, TypeError):
            try:
                return str(table.export_to_markdown())
            except Exception:  # noqa: BLE001 - table export is best-effort
                return ""
