"""Database schema.

Five tables rather than the single flat one the previous code used. The shape
is driven by three requirements:

**Bibliographic questions must be answered by SQL.** "How many papers in 2023?"
is a ``COUNT`` over ``documents``, not something a language model recalls. That
needs real columns, human-verified.

**Embeddings are keyed by ``(chunk_id, model_id)``.** The previous code indexed
with one model and queried with another — arithmetically valid, semantically
meaningless, and completely silent. Recording which model produced each vector
makes the mismatch detectable, and allows two models to be indexed side by side
for comparison without a rebuild.

**Chunks belong to sections, and ``(section_id, ordinal_in_section)`` is
unique.** Parent expansion fetches contiguous ordinal ranges within one
section, so a span cannot bleed across sections or documents. The previous
implementation expanded by global row id and dropped the filename filter, which
allowed exactly that.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Must match the configured embedder. Enforced at startup, not assumed.
EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)

    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSONB, default=list)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    venue: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(Text)

    #: Citations recorded by the registry. NULL when unknown, which is a
    #: different claim from zero and is reported differently.
    cited_by: Mapped[int | None] = mapped_column(Integer, default=None)

    #: Which registry the bibliographic fields came from, empty when they were
    #: read off the page. Stored so an answer's provenance is inspectable
    #: rather than assumed.
    metadata_source: Mapped[str] = mapped_column(String(40), default="")

    #: False until the bibliographic fields are trustworthy. Bibliographic
    #: questions are only answered from verified rows.
    #:
    #: Originally that meant a human had confirmed them. It now also covers a
    #: resolved DOI, which measured *more* accurate than the review: the
    #: review file was pre-filled with extraction guesses, and three of twenty
    #: confirmed years were wrong because spotting an error is harder than
    #: supplying the answer.
    metadata_verified: Mapped[bool] = mapped_column(default=False)

    source_path: Mapped[str] = mapped_column(Text)
    #: Content hash, so re-ingesting an unchanged file is a no-op. Extraction
    #: costs ~40s per paper, and chunker changes should not pay it again.
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    page_count: Mapped[int] = mapped_column(default=0)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sections: Mapped[list[Section]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_section_ordinal"),
        Index("ix_sections_kind", "kind"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(30))
    heading: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(default=0)
    page_end: Mapped[int] = mapped_column(default=0)

    document: Mapped[Document] = relationship(back_populates="sections")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # Parent expansion depends on this being dense and unique.
        UniqueConstraint("section_id", "ordinal_in_section", name="uq_chunk_ordinal"),
        Index("ix_chunks_document", "document_id"),
        Index("ix_chunks_span", "section_id", "ordinal_in_section"),
        # Full-text index backing the lexical retrieval branch. Dense vectors
        # smear rare identifiers like FCS-MPCC and NSGA-II; this recovers them.
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        CheckConstraint("length(text) > 0", name="ck_chunk_not_empty"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    section_id: Mapped[UUID] = mapped_column(ForeignKey("sections.id", ondelete="CASCADE"))
    ordinal_in_section: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20), default="prose")

    #: Verbatim source. Citation verification checks that quoted spans appear
    #: here, so this must never contain generated text.
    text: Mapped[str] = mapped_column(Text)
    #: Prepended to the text before embedding, never stored inside it.
    context_header: Mapped[str] = mapped_column(Text, default="")

    token_count: Mapped[int] = mapped_column(default=0)
    page: Mapped[int] = mapped_column(default=0)

    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    section: Mapped[Section] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class ChunkEmbedding(Base):
    """One vector per (chunk, model).

    The composite primary key is the structural fix for the defect in the code
    this replaces: a vector cannot exist without recording which model produced
    it, so index-time and query-time models can be compared rather than assumed
    equal.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")


#: HNSW indexes are created after bulk loading rather than declared here.
#: Building them on an empty table and inserting through them is markedly
#: slower than building once over the finished data.
HNSW_STATEMENTS = (
    """
    CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    """,
)
