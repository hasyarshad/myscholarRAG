"""An in-memory vector store."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from myscholar.domain.enums import SectionKind
from myscholar.ports.embedding import Vector
from myscholar.ports.store import SearchFilters, SearchHit, VectorStore

_WORD = re.compile(r"[a-z0-9][a-z0-9\-_]*")


@dataclass(slots=True)
class _Record:
    chunk_id: UUID
    document_id: UUID
    section_id: UUID
    ordinal_in_section: int
    text: str
    embedding: Vector
    section_kind: SectionKind = SectionKind.OTHER
    year: int | None = None


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Exact search over a list, for tests and small fixtures.

    Behaviourally equivalent to the Postgres adapter for everything the pipeline
    relies on: the same filters apply, lexical scoring ranks by term overlap, and
    span fetching is scoped to a section. It is exact rather than approximate,
    which is a feature in tests — no HNSW recall noise to explain away.
    """

    def __init__(self, model_id: str = "fake/hash-embedder") -> None:
        self._records: list[_Record] = []
        self._model_id = model_id

    def add(self, record: _Record) -> None:
        self._records.append(record)

    async def indexed_model_ids(self) -> set[str]:
        return {self._model_id} if self._records else set()

    def _matches(self, rec: _Record, filters: SearchFilters | None) -> bool:
        if filters is None or filters.is_empty:
            return True
        if filters.document_ids and rec.document_id not in filters.document_ids:
            return False
        if filters.years and rec.year not in filters.years:
            return False
        if filters.section_kinds and rec.section_kind not in filters.section_kinds:
            return False
        return not (
            filters.exclude_section_kinds and rec.section_kind in filters.exclude_section_kinds
        )

    async def search_dense(
        self,
        embedding: Vector,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        scored = [
            (rec, _cosine(embedding, rec.embedding))
            for rec in self._records
            if self._matches(rec, filters)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [self._to_hit(rec, score) for rec, score in scored[:limit]]

    async def search_lexical(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        terms = set(_WORD.findall(query.lower()))
        if not terms:
            return []

        # Inverse document frequency, so a rare identifier such as "fcs-mpcc"
        # counts for far more than a common word. This mirrors what ts_rank_cd
        # does in Postgres closely enough for tests to be meaningful.
        doc_freq: dict[str, int] = defaultdict(int)
        tokenised = [set(_WORD.findall(rec.text.lower())) for rec in self._records]
        for tokens in tokenised:
            for term in terms & tokens:
                doc_freq[term] += 1

        total = max(len(self._records), 1)
        scored: list[tuple[_Record, float]] = []
        for rec, tokens in zip(self._records, tokenised, strict=True):
            if not self._matches(rec, filters):
                continue
            score = sum(math.log(1 + total / (1 + doc_freq[term])) for term in terms & tokens)
            if score > 0.0:
                scored.append((rec, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [self._to_hit(rec, score) for rec, score in scored[:limit]]

    async def fetch_span(
        self,
        section_id: UUID,
        start_ordinal: int,
        end_ordinal: int,
    ) -> list[SearchHit]:
        span = [
            rec
            for rec in self._records
            if rec.section_id == section_id
            and start_ordinal <= rec.ordinal_in_section <= end_ordinal
        ]
        span.sort(key=lambda rec: rec.ordinal_in_section)
        return [self._to_hit(rec, 0.0) for rec in span]

    async def health(self) -> bool:
        return True

    @staticmethod
    def _to_hit(rec: _Record, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=rec.chunk_id,
            document_id=rec.document_id,
            section_id=rec.section_id,
            ordinal_in_section=rec.ordinal_in_section,
            text=rec.text,
            score=score,
        )
