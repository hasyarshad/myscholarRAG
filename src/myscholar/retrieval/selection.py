"""Choosing which documents to search, before searching passages.

Chunk search serves questions about a body of work badly: the applied papers say
"LFBLSTM" or "feature fusion", while a survey says "machine learning" on every page.
Selection matches document summaries instead and scopes retrieval to the result.

Order of authority: a named paper, then summary similarity (reviews last), then the
whole corpus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from uuid import UUID

from myscholar.adapters.persistence.repository import PostgresStore
from myscholar.domain.enums import DocumentKind
from myscholar.ports.embedding import Embedder

logger = logging.getLogger(__name__)

#: Slots reserved for the CV and the site pages, so a research question keeps
#: its papers and a biographical one still reaches the CV.
_PAGE_SLOTS = 2

#: Questions asking for a review, where demoting reviews would drop the right document.
_WANTS_A_REVIEW = re.compile(
    r"\b(survey|review|overview|taxonomy|state[\s-]of[\s-]the[\s-]art|literature)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Selection:
    """Which documents are in scope, and why."""

    document_ids: frozenset[UUID] = frozenset()
    reason: str = "whole corpus"


class DocumentSelector:
    """Decides which documents a question should be answered from."""

    def __init__(
        self,
        *,
        store: PostgresStore,
        embedder: Embedder,
        demote_reviews: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._demote_reviews = demote_reviews

    async def select(
        self,
        question: str,
        *,
        titles: tuple[str, ...],
        title_to_id: dict[str, UUID],
        limit: int,
    ) -> Selection:
        """Scope for one question.

        A ``limit`` of zero means no selection unless a title was resolved.
        """
        if named := frozenset(title_to_id[t] for t in titles if t in title_to_id):
            return Selection(document_ids=named, reason=f"{len(named)} named paper(s)")

        if limit <= 0:
            return Selection()

        try:
            matches = await self._store.search_summaries(
                await self._embedder.embed_query(question),
                model_id=self._embedder.model_id,
                # Over-fetched because reviews are moved to the back below.
                limit=limit * 2,
            )
        except Exception:  # selection is an optimisation, not a requirement
            logger.exception("document selection failed; searching the whole corpus")
            return Selection()

        if not matches:
            # No summaries indexed: search the whole corpus.
            return Selection()

        if self._demote_reviews and not _WANTS_A_REVIEW.search(question):
            # Stable, so similarity order survives within each group.
            matches = sorted(matches, key=lambda match: match[2])

        chosen = self._quota(matches, limit)
        return Selection(document_ids=chosen, reason=f"{len(chosen)} document(s) by summary")

    @staticmethod
    def _quota(matches: list[tuple[UUID, float, bool, str]], limit: int) -> frozenset[UUID]:
        """Fill paper and page slots separately so pages cannot crowd out papers.

        The CV and site pages describe the work secondhand but match broad questions
        strongly; on similarity alone, selection recall fell from 88% to 76%. Up to
        ``_PAGE_SLOTS`` go to pages, and unused slots on either side go to the other.
        """
        papers = [m for m in matches if m[3] == DocumentKind.PAPER]
        others = [m for m in matches if m[3] != DocumentKind.PAPER]

        reserved = min(_PAGE_SLOTS, len(others), max(limit - 1, 0))
        chosen = [m[0] for m in papers[: limit - reserved]]
        chosen += [m[0] for m in others[:reserved]]

        # Backfill from whichever side still has candidates.
        if len(chosen) < limit:
            picked = set(chosen)
            chosen += [m[0] for m in matches if m[0] not in picked][: limit - len(chosen)]
        return frozenset(chosen)
