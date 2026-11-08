"""Database access: writes during ingestion, searches during retrieval."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from myscholar.adapters.persistence.orm import (
    HNSW_STATEMENTS,
    Base,
    Chunk,
    ChunkEmbedding,
    ConversationTurn,
    Document,
    DocumentSummary,
    Section,
)
from myscholar.domain.enums import DocumentKind, SectionKind
from myscholar.ingest.health import DocumentStats
from myscholar.ingest.sync import IndexedDocument
from myscholar.ports.embedding import Vector
from myscholar.ports.store import SearchFilters, SearchHit, VectorStore
from myscholar.query.history import MAX_TURN_CHARS, Turn


class IndexMismatchError(RuntimeError):
    """The configured embedder is not the one the index was built with.

    Searching an index built by a different model returns ranked but meaningless results.
    """


def _source_url(doi: str | None, source_path: str | None) -> str:
    """Where a citation should send a reader, or empty if nowhere.

    A paper links to its DOI, which survives publisher site changes; other
    documents link to their page. ``source_path`` can also be a local path for a
    PDF ingested from disk (``/home/user/papers/foo.pdf``), so only http(s) URLs
    are used.
    """
    if doi:
        return f"https://doi.org/{doi}"
    if source_path and source_path.startswith(("http://", "https://")):
        return source_path
    return ""


@dataclass(frozen=True, slots=True)
class SpanContext:
    """What a retrieved span needs beyond its own text."""

    document_title: str
    section_heading: str
    page_start: int
    page_end: int
    document_kind: str
    is_review: bool
    authors: list[str]
    source_url: str


@dataclass(frozen=True, slots=True)
class DocumentRow:
    """A document as the CLI and the bibliographic query path see it."""

    id: UUID
    slug: str
    kind: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    verified: bool
    cited_by: int | None
    sections: int
    chunks: int


def _to_hit(row: Any, *, score: float | None = None) -> SearchHit:
    """Build a hit from a result row by column name, so column order does not matter."""
    return SearchHit(
        chunk_id=row.id,
        document_id=row.document_id,
        section_id=row.section_id,
        ordinal_in_section=row.ordinal_in_section,
        text=row.text,
        score=float(row.score) if score is None else score,
    )


def create_engine(url: str, *, echo: bool = False) -> AsyncEngine:
    """Create an engine whose connections search ``public`` and then ``extensions``.

    pgvector lives in ``extensions``, and neither SQLAlchemy's ``VECTOR(768)`` nor
    the ``<=>`` operator is schema-qualified, so without this schema creation and
    similarity queries fail. It is set per connection, not with ``ALTER DATABASE``,
    so a change in the Supabase dashboard cannot revert it.
    """
    return create_async_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "public,extensions"}},
    )


class PostgresStore(VectorStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        async with self._session() as session:
            yield session

    # -- schema ------------------------------------------------------------

    #: Columns added after the first schema. ``create_all`` creates missing tables
    #: but not missing columns, and there is no migration tool: the index can be
    #: rebuilt from the papers, but at about 40 s per document.
    _ADDED_COLUMNS = (
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS cited_by INTEGER",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_review BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_source VARCHAR(40) "
        "NOT NULL DEFAULT ''",
    )

    #: Extensions go in their own schema. Objects in ``public`` are reachable by
    #: every role and exposed through PostgREST, which Supabase's linter flags.
    #: The move is conditional: ``CREATE EXTENSION IF NOT EXISTS`` does not
    #: relocate an installed extension, and an unconditional ``ALTER`` fails once
    #: it has moved.
    _EXTENSIONS = (
        "CREATE SCHEMA IF NOT EXISTS extensions",
        "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions",
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_extension e
                  JOIN pg_namespace n ON n.oid = e.extnamespace
                 WHERE e.extname = 'vector' AND n.nspname <> 'extensions'
            ) THEN
                ALTER EXTENSION vector SET SCHEMA extensions;
            END IF;
        END $$;
        """,
    )

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            for statement in self._EXTENSIONS:
                await connection.execute(text(statement))
            await connection.run_sync(Base.metadata.create_all)
            for statement in self._ADDED_COLUMNS:
                await connection.execute(text(statement))

    async def create_vector_indexes(self) -> None:
        """Built after bulk loading, which is much faster than inserting through them."""
        async with self._engine.begin() as connection:
            for statement in HNSW_STATEMENTS:
                await connection.execute(text(statement))

    # -- integrity ---------------------------------------------------------

    async def indexed_model_ids(self) -> set[str]:
        async with self.session() as session:
            rows = await session.execute(select(ChunkEmbedding.model_id).distinct())
            return set(rows.scalars().all())

    async def assert_embedder_matches(self, model_id: str) -> None:
        """Raise ``IndexMismatchError`` if the index has vectors, but none from ``model_id``."""
        indexed = await self.indexed_model_ids()
        if indexed and model_id not in indexed:
            msg = (
                f"The index was built with {sorted(indexed)} but the configured "
                f"embedder is {model_id!r}. Searching across models returns "
                f"meaningless results. Re-index, or change the configuration."
            )
            raise IndexMismatchError(msg)

    async def health(self) -> bool:
        try:
            async with self.session() as session:
                await session.execute(select(1))
        except Exception:  # noqa: BLE001 - any failure means unhealthy
            return False
        return True

    # -- ingestion ---------------------------------------------------------

    async def taken_slugs(self) -> set[str]:
        """Slugs already in use, so a new document can avoid colliding."""
        async with self.session() as session:
            rows = await session.execute(select(Document.slug))
            return set(rows.scalars().all())

    async def document_exists(self, content_sha256: str) -> bool:
        async with self.session() as session:
            found = await session.scalar(
                select(Document.id).where(Document.content_sha256 == content_sha256)
            )
            return found is not None

    async def delete_for_reingest(self, *, content_sha256: str, source_path: str) -> None:
        """Remove any prior copy of a document, matched by content hash or source path.

        The hash catches a renamed file. The source path catches a re-extracted
        web page, whose hash is computed over the extracted text and so changes
        whenever the extractor does.
        """
        async with self.session() as session, session.begin():
            await session.execute(
                delete(Document).where(
                    (Document.content_sha256 == content_sha256)
                    | (Document.source_path == source_path)
                )
            )

    # -- document summaries ------------------------------------------------

    async def documents_needing_summary(self, model_id: str) -> list[tuple[UUID, str, str]]:
        """Documents with no summary for this embedder, with text to summarise from.

        Keyed on the embedder because a summary is stored with its vector.
        """
        async with self.session() as session:
            existing = await session.scalars(
                select(DocumentSummary.document_id).where(DocumentSummary.model_id == model_id)
            )
            done = set(existing.all())

            # Summaries use these sections only: reading in document order pulled
            # in related work (other researchers' methods) and often missed the
            # paper's own findings.
            summary_sections = (
                SectionKind.ABSTRACT.value,
                SectionKind.METHODS.value,
                SectionKind.RESULTS.value,
                SectionKind.CONCLUSION.value,
                SectionKind.DISCUSSION.value,
            )
            # The section filter applies to papers only: other documents have
            # none of these kinds, and a document with no summary cannot be selected.
            rows = await session.execute(
                select(Document.id, Document.title, Chunk.text)
                .join(Chunk, Chunk.document_id == Document.id)
                .join(Section, Section.id == Chunk.section_id)
                .where(
                    or_(
                        Document.kind != DocumentKind.PAPER.value,
                        Section.kind.in_(summary_sections),
                    )
                )
                .order_by(Document.id, Section.ordinal, Chunk.ordinal_in_section)
            )

        by_document: dict[UUID, list[str]] = {}
        titles: dict[UUID, str] = {}
        for document_id, title, chunk_text in rows:
            if document_id in done:
                continue
            titles[document_id] = title
            by_document.setdefault(document_id, []).append(chunk_text)

        return [(did, titles[did], "\n\n".join(texts)) for did, texts in by_document.items()]

    async def store_summaries(
        self, summaries: Sequence[tuple[UUID, str, Vector]], *, model_id: str
    ) -> None:
        async with self.session() as session, session.begin():
            for document_id, text, embedding in summaries:
                await session.execute(
                    delete(DocumentSummary).where(
                        (DocumentSummary.document_id == document_id)
                        & (DocumentSummary.model_id == model_id)
                    )
                )
                session.add(
                    DocumentSummary(
                        document_id=document_id,
                        model_id=model_id,
                        text=text,
                        embedding=embedding,
                    )
                )

    async def mark_reviews(self, document_ids: Sequence[UUID]) -> None:
        """Flag papers that survey other researchers' work."""
        async with self.session() as session, session.begin():
            await session.execute(
                update(Document).where(Document.id.in_(list(document_ids))).values(is_review=True)
            )

    async def search_summaries(
        self, embedding: Vector, *, model_id: str, limit: int = 6
    ) -> list[tuple[UUID, float, bool, str]]:
        """The documents whose summaries are closest to a query.

        Returns ``(document_id, similarity, is_review, kind)``, best first. Used to
        select documents before chunk retrieval on cross-paper questions.
        """
        async with self.session() as session:
            rows = await session.execute(
                select(
                    DocumentSummary.document_id,
                    DocumentSummary.embedding.cosine_distance(embedding).label("distance"),
                    Document.is_review,
                    Document.kind,
                )
                .join(Document, Document.id == DocumentSummary.document_id)
                .where(DocumentSummary.model_id == model_id)
                .order_by("distance")
                .limit(limit)
            )
            return [
                (document_id, 1.0 - float(distance), bool(is_review), str(kind))
                for document_id, distance, is_review, kind in rows
            ]

    async def summary_count(self, model_id: str) -> int:
        async with self.session() as session:
            return int(
                await session.scalar(
                    select(func.count(DocumentSummary.document_id)).where(
                        DocumentSummary.model_id == model_id
                    )
                )
                or 0
            )

    async def corpus_stats(self) -> list[DocumentStats]:
        """Per-document index contents, for the health check."""
        async with self.session() as session:
            rows = await session.execute(
                text("""
                select d.title, d.kind, d.page_count, d.metadata_verified, d.doi,
                       count(distinct s.id)                                    as sections,
                       count(distinct s.id) filter (where s.kind = 'other')    as unclassified,
                       count(distinct s.id) filter (
                           where s.kind in ('references', 'acknowledgements'))  as excluded,
                       count(c.id)                                             as chunks,
                       count(c.id) filter (where length(trim(c.text)) = 0)     as empty_chunks,
                       count(distinct ds.document_id)                          as summaries
                from documents d
                left join sections s on s.document_id = d.id
                left join chunks c on c.section_id = s.id
                left join document_summaries ds on ds.document_id = d.id
                group by d.id
                order by d.title
                """)
            )
            return [
                DocumentStats(
                    title=row.title,
                    kind=row.kind,
                    pages=int(row.page_count or 0),
                    chunks=int(row.chunks or 0),
                    sections=int(row.sections or 0),
                    unclassified_sections=int(row.unclassified or 0),
                    excluded_sections=int(row.excluded or 0),
                    empty_chunks=int(row.empty_chunks or 0),
                    has_summary=bool(row.summaries),
                    verified=bool(row.metadata_verified),
                    doi=row.doi,
                )
                for row in rows
            ]

    # -- conversations -----------------------------------------------------

    async def recent_turns(self, conversation_id: str, *, limit: int = 4) -> list[Turn]:
        """The last ``limit`` exchanges, oldest first.

        A follow-up usually refers to the previous turn; more history costs tokens
        and offers the resolver more wrong antecedents.
        """
        async with self.session() as session:
            rows = await session.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.conversation_id == conversation_id)
                .order_by(ConversationTurn.ordinal.desc())
                .limit(limit)
            )
            return [
                Turn(question=row.question, answer=row.answer, refused=row.refused)
                for row in reversed(list(rows))
            ]

    async def append_turn(
        self, conversation_id: str, *, question: str, answer: str, refused: bool
    ) -> None:
        """Record an exchange, truncated: resolving a follow-up only needs the gist."""
        async with self.session() as session, session.begin():
            last = await session.scalar(
                select(func.max(ConversationTurn.ordinal)).where(
                    ConversationTurn.conversation_id == conversation_id
                )
            )
            session.add(
                ConversationTurn(
                    id=uuid4(),
                    conversation_id=conversation_id,
                    ordinal=(last or 0) + 1,
                    question=question[:MAX_TURN_CHARS],
                    answer=answer[:MAX_TURN_CHARS],
                    refused=refused,
                )
            )

    async def prune_conversations(self, *, older_than_days: int = 30) -> int:
        """Delete turns older than ``older_than_days``; returns how many were deleted."""
        async with self.session() as session, session.begin():
            result = await session.execute(
                delete(ConversationTurn).where(
                    ConversationTurn.created_at
                    < func.now() - text(f"interval '{int(older_than_days)} days'")
                )
            )
            return int(getattr(result, "rowcount", 0) or 0)

    async def add_document(
        self,
        document: Document,
        sections: Sequence[tuple[Section, Sequence[tuple[Chunk, Vector]]]],
        *,
        model_id: str,
    ) -> None:
        """Insert a document with its sections, chunks and vectors in one transaction.

        A partial insert would leave gaps in chunk ordinals, and parent expansion
        would return truncated spans.
        """
        async with self.session() as session, session.begin():
            session.add(document)
            for section, chunks in sections:
                session.add(section)
                for chunk, vector in chunks:
                    session.add(chunk)
                    session.add(
                        ChunkEmbedding(chunk_id=chunk.id, model_id=model_id, embedding=vector)
                    )

    async def chunks_for_reheadering(self) -> list[tuple[UUID, UUID, str, str, str]]:
        """Every chunk with what is needed to rebuild its contextual header.

        Returns ``(chunk_id, document_id, chunk_text, section_heading,
        section_kind)``.
        """
        async with self.session() as session:
            rows = await session.execute(
                select(
                    Chunk.id,
                    Chunk.document_id,
                    Chunk.text,
                    Section.heading,
                    Section.kind,
                ).join(Section, Section.id == Chunk.section_id)
            )
            return [
                (row.id, row.document_id, row.text, row.heading, row.kind) for row in rows.all()
            ]

    async def slugs_by_filename(self) -> dict[str, str]:
        """Current slug for each document, keyed by source filename.

        Lets a caller exclude a document's own slug when regenerating it, so it
        is not treated as a collision.
        """
        async with self.session() as session:
            rows = await session.execute(select(Document.source_path, Document.slug))
            return {Path(source).name: slug for source, slug in rows.all()}

    async def update_metadata(self, filename: str, metadata: dict[str, Any]) -> UUID | None:
        """Apply reviewed bibliographic fields to one document.

        Matched on source path: the review file is keyed by filename, and
        correcting a title changes the slug.
        """
        async with self.session() as session, session.begin():
            document = await session.scalar(
                select(Document).where(Document.source_path.like(f"%{filename}"))
            )
            if document is None:
                return None
            for field_name, value in metadata.items():
                setattr(document, field_name, value)
            return document.id

    async def bibliographic_view(self) -> list[IndexedDocument]:
        """Every paper's bibliographic fields, for reconciliation.

        Pages and the CV are excluded, as they have no DOI or registry record.
        """
        async with self.session() as session:
            rows = await session.scalars(
                select(Document)
                .where(Document.kind == DocumentKind.PAPER.value)
                .order_by(Document.year.desc().nulls_last(), Document.title)
            )
            return [
                IndexedDocument(
                    slug=row.slug,
                    title=row.title,
                    authors=tuple(row.authors or ()),
                    year=row.year,
                    venue=row.venue,
                    doi=row.doi,
                    cited_by=row.cited_by,
                )
                for row in rows
            ]

    async def apply_correction(self, slug: str, fields: dict[str, Any], *, source: str) -> bool:
        """Write registry-resolved fields onto one document and mark it verified.

        The slug stays unchanged even when the title changes, because links are
        built from it.
        """
        async with self.session() as session, session.begin():
            document = await session.scalar(select(Document).where(Document.slug == slug))
            if document is None:
                return False
            for name, value in fields.items():
                setattr(document, name, value)
            document.metadata_source = source
            document.metadata_verified = True
            return True

    async def replace_chunk_headers(
        self,
        updates: Sequence[tuple[UUID, str, Vector]],
        *,
        model_id: str,
    ) -> None:
        """Rewrite contextual headers and their embeddings.

        The header names the paper and section, so a title correction requires
        re-embedding but not re-extraction.
        """
        async with self.session() as session, session.begin():
            for chunk_id, header, vector in updates:
                await session.execute(
                    update(Chunk).where(Chunk.id == chunk_id).values(context_header=header)
                )
                await session.execute(
                    update(ChunkEmbedding)
                    .where(
                        ChunkEmbedding.chunk_id == chunk_id,
                        ChunkEmbedding.model_id == model_id,
                    )
                    .values(embedding=vector)
                )

    # -- span context ------------------------------------------------------

    async def section_chunk_counts(self, section_ids: set[UUID]) -> dict[UUID, int]:
        """How many chunks each section holds, used by parent expansion to clamp spans."""
        if not section_ids:
            return {}
        async with self.session() as session:
            rows = await session.execute(
                select(Chunk.section_id, func.count(Chunk.id))
                .where(Chunk.section_id.in_(section_ids))
                .group_by(Chunk.section_id)
            )
            return {section_id: int(count) for section_id, count in rows.all()}

    async def span_context(self, section_id: UUID) -> SpanContext | None:
        """Title, heading, page range, document kind, review flag, authors and link.

        Authors are included because passages rarely state authorship, and a
        related-work section is full of other researchers' names; without them
        the generator cannot tell whose work it is reading.
        """
        async with self.session() as session:
            row = (
                await session.execute(
                    select(
                        Document.title,
                        Section.heading,
                        Section.page_start,
                        Section.page_end,
                        Document.kind,
                        Document.is_review,
                        Document.authors,
                        Document.doi,
                        Document.source_path,
                    )
                    .join(Document, Document.id == Section.document_id)
                    .where(Section.id == section_id)
                )
            ).first()
        if row is None:
            return None
        return SpanContext(
            document_title=row.title,
            section_heading=row.heading,
            page_start=int(row.page_start),
            page_end=int(row.page_end),
            document_kind=row.kind,
            is_review=bool(row.is_review),
            authors=list(row.authors or []),
            source_url=_source_url(row.doi, row.source_path),
        )

    # -- statistics --------------------------------------------------------

    async def list_documents(self) -> list[DocumentRow]:
        async with self.session() as session:
            section_counts = (
                select(Section.document_id, func.count(Section.id).label("n"))
                .group_by(Section.document_id)
                .subquery()
            )
            chunk_counts = (
                select(Chunk.document_id, func.count(Chunk.id).label("n"))
                .group_by(Chunk.document_id)
                .subquery()
            )
            rows = await session.execute(
                select(
                    Document,
                    func.coalesce(section_counts.c.n, 0),
                    func.coalesce(chunk_counts.c.n, 0),
                )
                .outerjoin(section_counts, section_counts.c.document_id == Document.id)
                .outerjoin(chunk_counts, chunk_counts.c.document_id == Document.id)
                .order_by(Document.year.desc().nullslast(), Document.title)
            )
            return [
                DocumentRow(
                    id=doc.id,
                    slug=doc.slug,
                    kind=doc.kind,
                    title=doc.title,
                    authors=list(doc.authors or []),
                    year=doc.year,
                    venue=doc.venue,
                    verified=doc.metadata_verified,
                    cited_by=doc.cited_by,
                    sections=int(n_sections),
                    chunks=int(n_chunks),
                )
                for doc, n_sections, n_chunks in rows.all()
            ]

    async def section_kind_counts(self) -> dict[str, int]:
        async with self.session() as session:
            rows = await session.execute(
                select(Section.kind, func.count(Section.id)).group_by(Section.kind)
            )
            return {kind: int(count) for kind, count in rows.all()}

    async def counts(self) -> dict[str, int]:
        async with self.session() as session:
            return {
                "documents": int(await session.scalar(select(func.count(Document.id))) or 0),
                "sections": int(await session.scalar(select(func.count(Section.id))) or 0),
                "chunks": int(await session.scalar(select(func.count(Chunk.id))) or 0),
                "embeddings": int(
                    await session.scalar(select(func.count(ChunkEmbedding.chunk_id))) or 0
                ),
            }

    # -- search ------------------------------------------------------------

    def _apply_filters(self, statement: Any, filters: SearchFilters | None) -> Any:
        if filters is None or filters.is_empty:
            return statement
        if filters.document_ids:
            statement = statement.where(Chunk.document_id.in_(filters.document_ids))
        if filters.years:
            statement = statement.where(Document.year.in_(filters.years))
        if filters.section_kinds:
            statement = statement.where(Section.kind.in_([k.value for k in filters.section_kinds]))
        if filters.exclude_section_kinds:
            statement = statement.where(
                Section.kind.notin_([k.value for k in filters.exclude_section_kinds])
            )
        return statement

    async def search_dense(
        self,
        embedding: Vector,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        distance = ChunkEmbedding.embedding.cosine_distance(embedding)
        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.section_id,
                Chunk.ordinal_in_section,
                Chunk.text,
                (1 - distance).label("score"),
            )
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .join(Section, Section.id == Chunk.section_id)
            .join(Document, Document.id == Chunk.document_id)
            .order_by(distance)
            .limit(limit)
        )
        async with self.session() as session:
            rows = await session.execute(self._apply_filters(statement, filters))
            return [_to_hit(row) for row in rows.all()]

    async def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        # The built-in parsers (websearch_to_tsquery, plainto_tsquery,
        # phraseto_tsquery) AND their terms, so a full question matched nothing:
        # "NSGA-II" returned 5 hits, a question about NSGA-II returned 0. The
        # question's lexemes are OR-ed instead (to_tsvector also drops stopwords
        # and stems), and ts_rank_cd ranks by how many terms match and how close
        # together they are.
        lexemes = func.tsvector_to_array(func.to_tsvector("english", query))
        tsquery = func.to_tsquery("english", func.array_to_string(lexemes, " | "))
        rank = func.ts_rank_cd(Chunk.tsv, tsquery)
        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.section_id,
                Chunk.ordinal_in_section,
                Chunk.text,
                rank.label("score"),
            )
            .join(Section, Section.id == Chunk.section_id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )
        async with self.session() as session:
            rows = await session.execute(self._apply_filters(statement, filters))
            return [_to_hit(row) for row in rows.all()]

    async def fetch_span(
        self,
        section_id: UUID,
        start_ordinal: int,
        end_ordinal: int,
    ) -> list[SearchHit]:
        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.section_id,
                Chunk.ordinal_in_section,
                Chunk.text,
            )
            .where(
                Chunk.section_id == section_id,
                Chunk.ordinal_in_section >= start_ordinal,
                Chunk.ordinal_in_section <= end_ordinal,
            )
            .order_by(Chunk.ordinal_in_section)
        )
        async with self.session() as session:
            rows = await session.execute(statement)
            return [_to_hit(row, score=0.0) for row in rows.all()]
