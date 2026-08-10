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

#: Docling item labels that represent a heading rather than content.
_HEADING_LABELS = frozenset({"section_header", "title", "page_header"})

#: Labels carrying no answerable content — running headers, page numbers,
#: footnotes referencing the publisher, and so on.
_SKIP_LABELS = frozenset({"page_footer", "page_header", "picture"})

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

        for item, _level in document.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            text = normalise(getattr(item, "text", "") or "")
            if not text or any(skip in label for skip in _SKIP_LABELS):
                continue

            page = self._page_of(item)

            if any(head in label for head in _HEADING_LABELS):
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

        return ExtractedDocument(
            source_path=path,
            content_sha256=_sha256(path),
            page_count=len(getattr(document, "pages", {}) or {}),
            blocks=tuple(blocks),
            first_page_text=first_page,
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
