"""The vector store interface.

Two implementations matter: Postgres with pgvector (local Docker in development,
hosted Supabase in production) and an in-memory store used by tests.

Search is split into dense and lexical methods rather than hidden behind one
call, because the caller fuses their rankings and needs them separately. Dense
search finds passages that mean the same thing; lexical search finds passages
containing the same rare terms. On a corpus full of identifiers like FCS-MPCC
and NSGA-II, the second is not optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID

from myscholar.domain.enums import SectionKind
from myscholar.ports.embedding import Vector


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Metadata constraints applied inside the search query.

    Callers should always issue at least one unfiltered branch alongside any
    filtered one, so an over-confident filter cannot empty the result set.
    """

    document_ids: frozenset[UUID] = field(default_factory=frozenset)
    years: frozenset[int] = field(default_factory=frozenset)
    section_kinds: frozenset[SectionKind] = field(default_factory=frozenset)
    exclude_section_kinds: frozenset[SectionKind] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not (
            self.document_ids or self.years or self.section_kinds or self.exclude_section_kinds
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    section_id: UUID
    ordinal_in_section: int
    text: str
    score: float


class VectorStore(ABC):
    """Persistence and search over chunks and their embeddings."""

    @abstractmethod
    async def indexed_model_ids(self) -> set[str]:
        """Embedding models present in the index.

        Checked at startup against the configured embedder. A mismatch is a
        fatal configuration error, not a silent quality regression.
        """

    @abstractmethod
    async def search_dense(
        self,
        embedding: Vector,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours by cosine distance."""

    @abstractmethod
    async def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Full-text ranking over the same chunks."""

    @abstractmethod
    async def fetch_span(
        self,
        section_id: UUID,
        start_ordinal: int,
        end_ordinal: int,
    ) -> list[SearchHit]:
        """Fetch a contiguous run of chunks within one section.

        Scoped to a section by design: parent expansion must never stitch
        together text from different sections or different documents.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Whether the store is reachable and usable."""
