"""Choosing which documents to search, before searching them.

Retrieval over a personal corpus asks two different questions and only one of
them is about passages:

    "What does the MMC paper say about capacitor balancing?"   → a passage
    "What has he contributed to machine learning?"             → a set of papers

Chunk search answers the first well and the second badly, because a paper
rarely describes itself in the words a reader would use. Six papers here apply
machine learning; none says so. They say LFBLSTM, AFL-1DCNN, unified feature
fusion. The one document that says "machine learning" on every page is a
*survey* of learning algorithms, so it won every slot and the applied work was
never retrieved.

So selection is a stage, not a filter bolted onto search. It runs once, decides
which documents are in scope, and hands that to retrieval. Everything about
"which paper" lives here; everything about "which passage" lives in the
retriever.

That separation is the point, and it is a correction. Selection was previously
spread across the router, the planner and the store — a per-document cap in one
place, a review demotion in another, a summary lookup in a third — and the
result was three partial policies that interacted by accident. This module is
the only thing that decides scope.

The order of authority, highest first:

1. **A named paper.** If the classifier resolved a title, that is the most
   specific signal anyone has given us and nothing should widen it.
2. **Summary similarity.** What each document *is*, matched against what was
   asked. Reviews rank last: a survey matches "his contribution to machine
   learning" strongly and contributes nothing to it, because its methods are
   other people's.
3. **Nothing.** Search the whole corpus, which is correct for a narrow factual
   question where one passage is the answer wherever it lives.
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

#: Questions that are asking for a review, in which case demoting reviews is
#: exactly backwards.
#:
#: Demotion exists because a survey matches "his contribution to machine
#: learning" strongly and contributes nothing to it. But "what survey reviews
#: algorithms that learn incrementally?" wants the survey, and demoting it
#: pushed the only correct document out of scope — a fix creating the failure
#: it was written to prevent, in the one case where its premise is false.
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

        ``limit`` of zero means the class does not want document selection —
        a single-paper factual question should search everywhere unless a title
        was resolved, because the answer is one passage and it does not matter
        which paper holds it.
        """
        if named := frozenset(title_to_id[t] for t in titles if t in title_to_id):
            return Selection(document_ids=named, reason=f"{len(named)} named paper(s)")

        if limit <= 0:
            return Selection()

        try:
            matches = await self._store.search_summaries(
                await self._embedder.embed_query(question),
                model_id=self._embedder.model_id,
                # Over-fetched because reviews are about to be moved to the
                # back and would otherwise consume slots on the way.
                limit=limit * 2,
            )
        except Exception:  # selection is an optimisation, not a requirement
            logger.exception("document selection failed; searching the whole corpus")
            return Selection()

        if not matches:
            # No summaries yet. The corpus behaves exactly as it did before
            # they existed, which is what makes this safe to add.
            return Selection()

        if self._demote_reviews and not _WANTS_A_REVIEW.search(question):
            # Stable, so similarity order survives within each group.
            matches = sorted(matches, key=lambda match: match[2])

        chosen = self._quota(matches, limit)
        return Selection(document_ids=chosen, reason=f"{len(chosen)} document(s) by summary")

    @staticmethod
    def _quota(matches: list[tuple[UUID, float, bool, str]], limit: int) -> frozenset[UUID]:
        """Fill from papers and pages separately, so neither crowds the other.

        The corpus holds two populations with different roles: twenty papers,
        which are the primary sources, and five site pages that describe the
        same work secondhand. A page matches a broad question about the person
        very strongly — the CV and the projects page are *about* him — so on
        similarity alone they take slots from the papers that contain the
        answer.

        Measured: giving the pages summaries, which was itself a fix, dropped
        selection recall from 88% to 76% and complete answers from 80% to 50%.
        Papers were displaced by pages describing them.

        A page is still worth having in scope — biographical questions are
        answered from the CV — so this reserves a couple of slots rather than
        excluding them. Either side gives up its unused slots to the other, so
        a corpus of only papers behaves exactly as before.
        """
        papers = [m for m in matches if m[3] == DocumentKind.PAPER]
        others = [m for m in matches if m[3] != DocumentKind.PAPER]

        reserved = min(_PAGE_SLOTS, len(others), max(limit - 1, 0))
        chosen = [m[0] for m in papers[: limit - reserved]]
        chosen += [m[0] for m in others[:reserved]]

        # Backfill from whichever side still has candidates, so a short list on
        # one side never leaves the selection smaller than it should be.
        if len(chosen) < limit:
            picked = set(chosen)
            chosen += [m[0] for m in matches if m[0] not in picked][: limit - len(chosen)]
        return frozenset(chosen)
