"""The ingestion pipeline: PDFs in, indexed chunks out."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from myscholar.adapters.persistence.orm import Chunk as ChunkRow
from myscholar.adapters.persistence.orm import Document as DocumentRow
from myscholar.adapters.persistence.orm import Section as SectionRow
from myscholar.adapters.persistence.repository import PostgresStore
from myscholar.domain.documents import PaperMetadata
from myscholar.domain.enums import NON_INDEXED_SECTIONS, DocumentKind
from myscholar.ingest.chunker import (
    ChunkingPolicy,
    build_context_header,
    chunk_section,
    strip_mathematics,
)
from myscholar.ingest.grouping import group_into_sections
from myscholar.ingest.metadata import MetadataExtractor, MetadataProposal, slugify
from myscholar.ingest.sectionizer import Sectionizer
from myscholar.ports.embedding import Embedder
from myscholar.ports.extractor import DocumentExtractor, ExtractionError

logger = logging.getLogger(__name__)


def _is_url(source: Path | str) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def _label(source: Path | str) -> str:
    """A short name for reports, whether the source is a file or a URL."""
    return str(source) if _is_url(source) else Path(source).name


def _stem(source: Path | str) -> str:
    """A slug fallback, whether the source is a file or a URL."""
    if _is_url(source):
        return str(source).rstrip("/").rsplit("/", 1)[-1] or "home"
    return Path(source).stem


@dataclass(slots=True)
class IngestReport:
    documents: int = 0
    sections: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    section_stats: dict[str, int] = field(default_factory=dict)
    proposals: list[MetadataProposal] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        *,
        extractor: DocumentExtractor,
        embedder: Embedder,
        store: PostgresStore,
        sectionizer: Sectionizer,
        metadata_extractor: MetadataExtractor,
        policy: ChunkingPolicy | None = None,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._store = store
        self._sectionizer = sectionizer
        self._metadata = metadata_extractor
        self._policy = policy or ChunkingPolicy()

    async def run(
        self,
        paths: Sequence[Path | str],
        *,
        kind: DocumentKind = DocumentKind.PAPER,
        known_metadata: dict[str, PaperMetadata] | None = None,
        force: bool = False,
    ) -> IngestReport:
        report = IngestReport()
        known_metadata = known_metadata or {}

        for source in paths:
            path = Path(source) if not _is_url(source) else source
            try:
                await self._ingest_one(path, kind, known_metadata, force, report)
            except ExtractionError as exc:
                report.failed.append((_label(path), str(exc)))
                logger.warning("extraction failed for %s: %s", _label(path), exc)
            except Exception as exc:
                report.failed.append((_label(path), f"{type(exc).__name__}: {exc}"))
                logger.exception("ingestion failed for %s", _label(path))

        report.section_stats = dict(self._sectionizer.stats)
        return report

    async def _ingest_one(
        self,
        path: Path | str,
        kind: DocumentKind,
        known_metadata: dict[str, PaperMetadata],
        force: bool,
        report: IngestReport,
    ) -> None:
        extracted = self._extractor.extract(path)

        # Content-hash skip: extraction costs ~40s per paper, and a chunker
        # change should not pay that again for unchanged files.
        if not force and await self._store.document_exists(extracted.content_sha256):
            report.skipped.append(f"{_label(path)} (unchanged)")
            return

        if reviewed := known_metadata.get(_label(path)):
            metadata = reviewed
        else:
            proposal = await self._metadata.propose(
                first_page=extracted.first_page_text,
                source_path=path,
                extracted_title=extracted.title,
            )
            metadata = proposal.metadata
            report.proposals.append(proposal)

        sections = await group_into_sections(
            extracted.blocks,
            sectionizer=self._sectionizer,
            title=metadata.title,
            kind=kind,
        )
        indexed = [s for s in sections if s.kind not in NON_INDEXED_SECTIONS]
        if dropped := len(sections) - len(indexed):
            report.warnings.append(f"{_label(path)}: {dropped} section(s) excluded")

        if force:
            # Keyed on content, not on the slug. Correcting a title changes the
            # slug, so deleting by slug removes nothing and the insert then
            # collides on the content hash — which failed every document on a
            # forced re-run after a metadata review.
            await self._store.delete_by_content_hash(extracted.content_sha256)

        slug = slugify(metadata.title, _stem(path), taken=await self._store.taken_slugs())

        document = DocumentRow(
            id=uuid4(),
            slug=slug,
            kind=kind.value,
            title=metadata.title,
            authors=list(metadata.authors),
            year=metadata.year,
            venue=metadata.venue,
            doi=metadata.doi,
            metadata_verified=metadata.verified,
            source_path=str(path),
            content_sha256=extracted.content_sha256,
            page_count=extracted.page_count,
        )

        payload: list[tuple[SectionRow, list[tuple[ChunkRow, list[float]]]]] = []
        for ordinal, pending in enumerate(indexed):
            page_start, page_end = pending.pages
            section_row = SectionRow(
                id=uuid4(),
                document_id=document.id,
                ordinal=ordinal,
                kind=pending.kind.value,
                heading=pending.heading,
                page_start=page_start,
                page_end=page_end,
            )

            chunks = chunk_section([(b.kind, b.text, b.page) for b in pending.blocks], self._policy)
            if not chunks:
                continue

            header = build_context_header(
                title=metadata.title,
                year=metadata.year,
                venue=metadata.venue,
                section_heading=pending.heading,
                section_kind=pending.kind.value,
            )

            # LaTeX is stripped for embedding only. The stored text stays
            # verbatim because citation verification checks quoted spans
            # against it.
            embed_texts = [f"{header}\n\n{strip_mathematics(c.text)}" for c in chunks]
            vectors = await self._embedder.embed_documents(embed_texts)

            rows = [
                (
                    ChunkRow(
                        id=uuid4(),
                        document_id=document.id,
                        section_id=section_row.id,
                        ordinal_in_section=chunk.ordinal,
                        kind=chunk.kind.value,
                        text=chunk.text,
                        context_header=header,
                        token_count=chunk.token_count,
                        page=chunk.page,
                    ),
                    vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            payload.append((section_row, rows))
            report.chunks += len(rows)

        if not payload:
            report.failed.append((_label(path), "no indexable content after sectioning"))
            return

        await self._store.add_document(document, payload, model_id=self._embedder.model_id)
        report.documents += 1
        report.sections += len(payload)
        report.warnings.extend(f"{_label(path)}: {w}" for w in extracted.warnings)
