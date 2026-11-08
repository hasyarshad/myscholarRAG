"""The vector store interface: Postgres with pgvector, and an in-memory store for tests.

Dense and lexical search are separate methods because the caller fuses their
rankings. Lexical search catches rare identifiers such as FCS-MPCC and NSGA-II.
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

    Callers should also issue an unfiltered branch, so an over-confident filter
    cannot empty the result set.
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
        """Embedding models present in the index, checked at startup against the embedder."""

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

        Parent expansion must never join text across sections or documents.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Whether the store is reachable and usable."""
