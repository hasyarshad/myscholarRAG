"""Database access: writes during ingestion, searches during retrieval."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text, update
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
    Document,
    DocumentSummary,
    Section,
)
from myscholar.ports.embedding import Vector
from myscholar.ports.store import SearchFilters, SearchHit, VectorStore


class IndexMismatchError(RuntimeError):
    """The configured embedder is not the one the index was built with.

    Fatal by design. Searching an index built by a different model returns
    plausible-looking, meaningless results — the exact failure mode of the code
    this project replaces, which went unnoticed because nothing checked.
    """


@dataclass(frozen=True, slots=True)
class DocumentRow:
    """A document as the CLI and the bibliographic query path see it."""

    id: UUID
    slug: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    verified: bool
    sections: int
    chunks: int


def _to_hit(row: Any, *, score: float | None = None) -> SearchHit:
    """Build a hit from a result row.

    Named rather than positional: the column list and the dataclass field order
    would otherwise have to be kept in step by hand, and a silent reordering
    would mix up chunk and section identifiers.
    """
    return SearchHit(
        chunk_id=row.id,
        document_id=row.document_id,
        section_id=row.section_id,
        ordinal_in_section=row.ordinal_in_section,
        text=row.text,
        score=float(row.score) if score is None else score,
    )


def create_engine(url: str, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(url, echo=echo, pool_pre_ping=True)


class PostgresStore(VectorStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        async with self._session() as session:
            yield session

    # -- schema ------------------------------------------------------------

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await connection.run_sync(Base.metadata.create_all)

    async def create_vector_indexes(self) -> None:
        """Built after bulk loading, which is much faster than inserting through them."""
        async with self._engine.begin() as connection:
            for statement in HNSW_STATEMENTS:
                await connection.execute(text(statement))

    async def drop_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    # -- integrity ---------------------------------------------------------

    async def indexed_model_ids(self) -> set[str]:
        async with self.session() as session:
            rows = await session.execute(select(ChunkEmbedding.model_id).distinct())
            return set(rows.scalars().all())

    async def assert_embedder_matches(self, model_id: str) -> None:
        """Fail loudly at startup rather than quietly returning noise."""
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

    async def delete_document(self, slug: str) -> None:
        async with self.session() as session, session.begin():
            await session.execute(delete(Document).where(Document.slug == slug))

    async def delete_by_content_hash(self, content_sha256: str) -> None:
        """Remove a document by source content.

        Re-ingestion has to key on the file, not the slug. Correcting a title
        changes the slug, so deleting by slug removes nothing and the insert
        then collides on the content hash — which failed every document on a
        forced re-run after a metadata review.
        """
        async with self.session() as session, session.begin():
            await session.execute(delete(Document).where(Document.content_sha256 == content_sha256))

    async def add_document(
        self,
        document: Document,
        sections: Sequence[tuple[Section, Sequence[tuple[Chunk, Vector]]]],
        *,
        model_id: str,
        summaries: Sequence[DocumentSummary] = (),
    ) -> None:
        """Insert a document with its sections, chunks and vectors in one transaction.

        All or nothing. A partially ingested document leaves a section whose
        chunk ordinals are not dense, and parent expansion would then silently
        return a truncated span rather than failing.

        ``model_id`` is required rather than inferred: it is what makes a
        later index/query mismatch detectable.
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
            for summary in summaries:
                session.add(summary)

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

    async def update_metadata(self, slug_or_path: str, metadata: dict[str, Any]) -> UUID | None:
        """Apply reviewed bibliographic fields to one document.

        Matched on source path rather than slug, because correcting a title
        changes the slug and the review file is keyed by filename.
        """
        async with self.session() as session, session.begin():
            document = await session.scalar(
                select(Document).where(Document.source_path.like(f"%{slug_or_path}"))
            )
            if document is None:
                return None
            for field_name, value in metadata.items():
                setattr(document, field_name, value)
            return document.id

    async def replace_chunk_headers(
        self,
        updates: Sequence[tuple[UUID, str, Vector]],
        *,
        model_id: str,
    ) -> None:
        """Rewrite contextual headers and their embeddings.

        The header names the paper and section, so correcting a title changes
        what every chunk of that paper embeds as. Re-embedding is required;
        re-extraction is not.
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
                    title=doc.title,
                    authors=list(doc.authors or []),
                    year=doc.year,
                    venue=doc.venue,
                    verified=doc.metadata_verified,
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
        # websearch_to_tsquery tolerates arbitrary user input — quotes, OR, and
        # minus signs — without the syntax errors plainto_tsquery raises.
        tsquery = func.websearch_to_tsquery("english", query)
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
