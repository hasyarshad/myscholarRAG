"""Per-claim faithfulness: is every factual sentence supported by the context?

The headline number for a RAG system, and the one most demos cannot produce.
"It gave a good answer" is an impression; "4% of its factual sentences were
unsupported by the passages it was given, against 34% for the pipeline it
replaces" is a measurement.

Deliberately separate from the grounding check inside the loop, for one
reason. That check treats an unavailable model as *supported*, because an
outage in a safety check must not manufacture refusals for visitors. That is
right for serving and wrong for measuring: it turns every failed call into
evidence of faithfulness. Here an unjudged sentence is recorded as unjudged
and excluded from the denominator, and the count is reported so a run with
many failures is visibly untrustworthy rather than quietly flattering.

Scope: this measures whether an answer is grounded *in the passages retrieved
for it*, not whether it is true. A claim faithfully drawn from a passage that
does not answer the question is faithful and unhelpful, which is what the
retrieval metrics are for.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from myscholar.domain.documents import ContextBlock
from myscholar.generation import citations
from myscholar.ports.llm import ChatModel, Message, Role

logger = logging.getLogger(__name__)


class _Support(BaseModel):
    supported: bool = Field(description="True only if the excerpts state or directly imply this.")


_PROMPT = """Does the excerpt support the statement?

Answer true only if the excerpt states the statement or directly implies it.
Answer false if the statement adds any specific detail the excerpt does not \
contain — a number, a name, a comparison, a cause — even if the statement is \
plausible and the excerpt is about the same subject.

Judge only what is written. Do not use anything you know about the subject.

Excerpt:
{excerpt}

Statement: {statement}
/no_think"""


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    sentence: str
    supported: bool | None
    """None when the judge could not be reached, which is not a verdict."""


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    verdicts: tuple[ClaimVerdict, ...]

    @property
    def judged(self) -> int:
        return sum(1 for v in self.verdicts if v.supported is not None)

    @property
    def supported(self) -> int:
        return sum(1 for v in self.verdicts if v.supported is True)

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(v.sentence for v in self.verdicts if v.supported is False)

    @property
    def unjudged(self) -> int:
        return sum(1 for v in self.verdicts if v.supported is None)

    @property
    def rate(self) -> float | None:
        """Fraction of judged claims that were unsupported.

        None when nothing was judged — an answer with no factual sentences, or
        a judge that could not be reached. Reported as absent rather than as
        zero, which would read as perfect.
        """
        return None if self.judged == 0 else len(self.unsupported) / self.judged


class FaithfulnessJudge:
    """Scores each factual sentence against the context the answer was given."""

    def __init__(self, model: ChatModel, *, concurrency: int = 4) -> None:
        self._model = model
        self._gate = asyncio.Semaphore(concurrency)

    async def score(self, answer: str, blocks: tuple[ContextBlock, ...]) -> FaithfulnessResult:
        # Markers are stripped first. "[C3]" appears in no excerpt, so leaving
        # them in asks the judge whether the context supports a string it
        # cannot contain, and the cited configurations would score worse than
        # the uncited ones for having citations at all.
        sentences = [
            citations.strip_markers(s).strip()
            for s in citations.split_sentences(answer)
            if citations.is_factual(s)
        ]
        sentences = [s for s in sentences if s]
        if not sentences or not blocks:
            return FaithfulnessResult(verdicts=())

        # Judged against the whole context rather than against the block a
        # sentence cites. A claim legitimately drawn from two excerpts cites
        # one of them, and scoring it against that one alone would mark
        # correct synthesis as unfaithful.
        excerpt = "\n\n".join(b.text for b in blocks)[:6000]

        verdicts = await asyncio.gather(*(self._judge(s, excerpt) for s in sentences))
        return FaithfulnessResult(verdicts=tuple(verdicts))

    async def _judge(self, sentence: str, excerpt: str) -> ClaimVerdict:
        async with self._gate:
            try:
                result = await self._model.structured(
                    [
                        Message(
                            role=Role.USER,
                            content=_PROMPT.format(excerpt=excerpt, statement=sentence[:600]),
                        )
                    ],
                    _Support,
                )
            except Exception:  # noqa: BLE001 - an unreachable judge is not a verdict
                logger.warning("could not judge a claim; recording it as unjudged")
                return ClaimVerdict(sentence=sentence, supported=None)
        return ClaimVerdict(sentence=sentence, supported=result.supported)
