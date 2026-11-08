"""Docling-based PDF extraction.

Uses the pypdfium backend: Docling's default parser drops ``fl`` and ``fi``
ligatures without reporting an error ("flux" became " ux" 51 times in one paper).
Formula enrichment turns equations into LaTeX instead of placeholders; the LaTeX is
stored but excluded from embeddings.
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

#: Labels that start a new section. ``title`` is excluded: the paper title is
#: captured as metadata, and as a heading it would open a section swallowing the
#: abstract and author block.
_HEADING_LABELS = frozenset({"section_header"})

#: Labels with no answerable content: running heads, page numbers and image placeholders.
_SKIP_LABELS = frozenset({"page_footer", "page_header", "picture"})

#: The document title, captured for metadata rather than treated as content.
_TITLE_LABELS = frozenset({"title"})

#: Shapes that mark a section heading rather than a paper title. Length cannot
#: separate them (titles in the corpus run 79-108 characters, some headings run
#: longer), but a title never opens with an enumerator or a canonical section word.
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


#: Publisher banners that Elsevier, Frontiers and IEEE templates put above the
#: title on the first page, such as "ScienceDirect" or "ORIGINAL RESEARCH".
_PUBLISHER_BANNER = re.compile(
    r"^\s*(sciencedirect|elsevier|springer|wiley|mdpi|frontiers|ieee|acm|"
    r"article|original\s+research|research\s+(paper|article)|review\s+article|"
    r"contents\s+lists|journal\s+homepage|open\s+access|"
    r"regular\s+paper|full\s+length\s+article)\b",
    re.IGNORECASE,
)

#: Shorter than this is furniture. Titles in the corpus run 79-108 characters.
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

#: Shortest repeat worth removing: longer than natural repetition such as
#: "had had", shorter than a duplicated title or affiliation line.
_MIN_REPEAT_CHARS = 20

#: How many recent blocks are checked for containment.
_REPEAT_WINDOW = 6

#: Fraction of repeated text above which a document is treated as damaged.
#: Nineteen papers in the corpus repeat 0-3% of their text and one repeats 52%.
#: Line deduplication is gated because it harms healthy papers: Docling emits
#: table captions twice, and removing every repeat moved the right document off
#: rank 1 for eight of thirty-seven retrieval questions.
_DAMAGED_ABOVE = 0.15

#: Kinds exempt from line deduplication: table rows repeat by nature, and
#: Docling emits each caption twice.
_DEDUP_EXEMPT = frozenset({ChunkKind.TABLE, ChunkKind.CAPTION})


def repetition_ratio(blocks: list[ExtractedBlock]) -> float:
    """Fraction of the document's characters that restate an earlier line."""
    seen: set[str] = set()
    repeated = total = 0
    for block in blocks:
        for line in block.text.split("\n"):
            total += len(line)
            key = _dedup_key(line)
            if len(key) < _MIN_REPEAT_CHARS:
                continue
            if key in seen:
                repeated += len(line)
            else:
                seen.add(key)
    return repeated / total if total else 0.0


_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _dedup_key(text: str) -> str:
    """Lowercase letters and digits only.

    IFAC's template stacks a second text layer with the punctuation stripped
    ("Surveillancereconnaissanceinfrastructure inspection"), so copies only
    match once normalised.
    """
    return _ALPHANUMERIC.sub("", text.casefold())


def collapse_repeated_lines(text: str, seen: set[str] | None = None) -> str:
    """Drop lines already seen, across the whole document when ``seen`` is shared.

    In a damaged paper one block can be an abstract accumulating line by line,
    and consecutive blocks are growing prefixes of the same text, so no two
    blocks are identical. Lines are removed whole and never edited, so every
    surviving line is verbatim source and quoted spans still resolve.
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
    """Remove blocks that restate another; returns the kept blocks and the number dropped.

    Repeats crowd out real content: the worst-affected paper produced 796 chunks
    from 6 pages, against about 12 per page elsewhere. Whole blocks and lines are
    removed but never edited, so citations still verify against the extracted text.
    """
    damaged = repetition_ratio(blocks) > _DAMAGED_ABOVE

    kept: list[ExtractedBlock] = []
    seen: dict[str, int] = {}  # key -> index in kept
    recent: list[tuple[str, int]] = []  # (key, index in kept), for containment
    seen_lines: set[str] = set()
    dropped = 0

    for original in blocks:
        block = original
        if damaged and block.kind not in _DEDUP_EXEMPT:
            collapsed = collapse_repeated_lines(block.text, seen_lines)
            if not collapsed.strip():
                # Every line repeats an earlier one.
                dropped += 1
                continue
            if collapsed != block.text:
                block = replace(block, text=collapsed)

        key = _dedup_key(block.text)
        if len(key) < _MIN_REPEAT_CHARS:
            kept.append(block)
            continue

        # Exact repeats are matched document-wide: the same paragraph can
        # reappear at any distance.
        duplicate_of = seen.get(key)

        # Containment is windowed: globally it would drop a short block that
        # appears inside an unrelated one, such as a heading quoted in a paragraph.
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
    """Repair the text defects found in this corpus.

    NFKC turns ligature codepoints such as U+FB02 into ``fl``, so backends that
    emit the ligature and backends that emit two letters index the same way.
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
        # OCR is off by default: the corpus is born-digital PDFs with a real text
        # layer, and OCR costs minutes per document and adds recognition errors.
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
        """Built on first use and reused, since model loading is slow."""
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

            # Only one paper in five has a title item; the rest label the title
            # a section_header. So the title is the first plausible heading on
            # page 1 before any section heading. Other content may precede it:
            # several papers open with a journal banner or an author block.
            looks_like_title = (
                not title
                and is_heading_label
                and page <= 1
                and not seen_section_heading
                and is_plausible_title(text)
            )

            # Furniture above the title (a banner, a correspondence line) is kept
            # as prose, so it cannot block the title as a section heading.
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
                # Later title items are figure or table titles and stay with the prose.
                if not title:
                    title = text
                else:
                    blocks.append(ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page))
            elif is_heading_label:
                # Some papers repeat the title as a running heading.
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
