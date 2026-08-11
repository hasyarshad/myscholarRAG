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


def looks_like_section_heading(text: str) -> bool:
    """Whether a heading-like item is a section heading rather than a title."""
    return bool(_SECTION_HEADING_SHAPE.match(text))


_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    """Repair the text defects this corpus actually exhibits.

    NFKC decomposes ligature codepoints such as U+FB02 (``ﬂ``) into ``fl``.
    Different PDF backends disagree about whether to emit the ligature as one
    character or two, and an index containing both forms would fail to match
    either consistently.
    """
    text = unicodedata.normalize("NFKC", text)
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

    def extract(self, path: Path) -> ExtractedDocument:
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
                and not looks_like_section_heading(text)
            )

            if is_title_label or looks_like_title:
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
