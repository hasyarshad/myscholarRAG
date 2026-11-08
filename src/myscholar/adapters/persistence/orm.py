"""Database schema.

- Bibliographic questions ("How many papers in 2023?") are answered by SQL over
  verified ``documents`` columns.
- Embeddings are keyed by ``(chunk_id, model_id)``, and the configured embedder is
  checked against the index at startup. Only one model should be indexed: the HNSW
  index covers the whole column, and a second model's vectors cost 0.07 MRR.
  ``myscholar papers check`` fails when more than one is present.
- ``(section_id, ordinal_in_section)`` is unique, so parent expansion fetches
  contiguous ranges within one section and never crosses sections or documents.
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

#: Must match the configured embedder; checked at startup.
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

    #: Citations recorded by the registry; NULL when unknown, which is distinct from zero.
    cited_by: Mapped[int | None] = mapped_column(Integer, default=None)

    #: True when the paper surveys others' work. A passage cannot show this, and
    #: without the flag the assistant credited the author with an algorithm the
    #: survey only described.
    is_review: Mapped[bool] = mapped_column(default=False)

    #: The registry the bibliographic fields came from; empty when read off the page.
    metadata_source: Mapped[str] = mapped_column(String(40), default="")

    #: True once a person or a resolved DOI has confirmed the bibliographic
    #: fields. Bibliographic questions are answered only from verified rows.
    metadata_verified: Mapped[bool] = mapped_column(default=False)

    source_path: Mapped[str] = mapped_column(Text)
    #: Content hash, so re-ingesting an unchanged file is a no-op (extraction
    #: takes about 40 s per paper).
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
        # Full-text index for lexical retrieval, which finds rare identifiers
        # such as FCS-MPCC that dense vectors blur.
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
    """One vector per (chunk, model), so every vector records the model that produced it."""

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")


class DocumentSummary(Base):
    """A paper described in one paragraph, embedded separately.

    A paper rarely describes itself in a reader's words: the fault-detection work
    names LFBLSTM and AFL-1DCNN but never says "machine learning", while a survey
    of learning algorithms says it on every page. Summaries use the reader's
    vocabulary, and cross-paper synthesis selects documents here before
    searching within them.
    """

    __tablename__ = "document_summaries"

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String(120), primary_key=True)

    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ConversationTurn(Base):
    """One exchange, held server-side.

    History is never accepted from a client, which could forge an assistant turn
    the model cannot tell from a real one. Only the question and answer are
    stored, with no IP, user agent or other identifier; the random conversation
    id is the only link between turns.
    """

    __tablename__ = "conversation_turns"
    __table_args__ = (
        Index("ix_turns_conversation", "conversation_id", "ordinal"),
        Index("ix_turns_created", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer)

    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    refused: Mapped[bool] = mapped_column(default=False)
    """Recorded so a refusal is not used to resolve a follow-up."""

    # No index=True: ix_turns_created already indexes this column.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


#: HNSW indexes are built after bulk loading, which is much faster than
#: inserting through an existing index.
HNSW_STATEMENTS = (
    """
    CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    """,
)
