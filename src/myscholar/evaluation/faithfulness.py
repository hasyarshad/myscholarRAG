"""Per-claim faithfulness: is every factual sentence supported by the retrieved context?

Separate from the grounding check in the loop, which treats an unreachable model as
supported so an outage causes no refusals. Here an unjudged sentence is counted and left
out of the denominator, so a run with many failures is visibly unreliable.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from myscholar.domain.documents import ContextBlock
from myscholar.generation import citations
from myscholar.ports.llm import ChatModel, Message, Role

logger = logging.getLogger(__name__)


class _Support(BaseModel):
    supported: bool = Field(description="True only if the excerpts state or directly imply this.")


#: The judge is told whose corpus this is, as the generator is. Without that, a bare CV
#: line ("Jun 2025 Best Paper Presentation Award IEEE ISIE") does not support "he
#: received the award", and correct CV-derived claims scored as unsupported.
_PROMPT = """The excerpt is from {person}'s own papers, CV or website. Each is \
labelled with the document and section it came from.

Does the excerpt support the statement?

Answer true if the excerpt states the statement or directly implies it, \
including where the excerpt is a list or a table and the statement reads it \
back in a sentence.

Answer false if the statement adds a specific detail the excerpt does not \
contain — a number, a name, a comparison, a cause — even if it is plausible \
and the excerpt is about the same subject. Answer false if the statement \
reverses or contradicts what the excerpt says.

Judge only what is written. Do not use anything you know about the subject.

Excerpt:
{excerpt}

Statement: {statement}
/no_think"""


#: Roughly a paragraph: small enough for a small judge model to take in, large enough
#: that a claim's evidence is rarely split across two.
_PASSAGE_CHARS = 700

#: Passages each verdict sees, ranked by overlap with the claim.
_PASSAGES_PER_CLAIM = 3

_WORD = re.compile(r"[a-z0-9]+")

#: Words that appear in every passage and so distinguish none of them.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
        "this",
        "these",
        "those",
        "which",
        "used",
        "using",
        "use",
        "based",
        "can",
        "may",
        "also",
    ]
)


def _passages(blocks: tuple[ContextBlock, ...]) -> list[str]:
    """Split the context into paragraph-sized pieces, on sentence boundaries.

    Each piece keeps its document and section label, as the generator saw it; a bare list
    means little without its heading.
    """
    pieces: list[str] = []
    for block in blocks:
        label = f"[{block.document_title} — {block.section_heading}]"
        current = ""
        for sentence in citations.split_sentences(block.text):
            if len(current) + len(sentence) > _PASSAGE_CHARS and current:
                pieces.append(f"{label}\n{current.strip()}")
                current = ""
            current += sentence + " "
        if current.strip():
            pieces.append(f"{label}\n{current.strip()}")
    return pieces


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if w not in _STOPWORDS and len(w) > 2}


def _relevant(sentence: str, passages: list[str]) -> str:
    """The passages sharing the most content words with the sentence."""
    if len(passages) <= _PASSAGES_PER_CLAIM:
        return "\n\n".join(passages)

    terms = _content_words(sentence)
    ranked = sorted(passages, key=lambda p: len(terms & _content_words(p)), reverse=True)
    return "\n\n".join(ranked[:_PASSAGES_PER_CLAIM])


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    sentence: str
    supported: bool | None
    """None when the judge could not be reached."""


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

        None when nothing was judged (no factual sentences, or the judge was unreachable),
        since zero would read as perfect.
        """
        return None if self.judged == 0 else len(self.unsupported) / self.judged


class FaithfulnessJudge:
    """Scores each factual sentence against the context the answer was given."""

    def __init__(
        self, model: ChatModel, *, concurrency: int = 4, person: str = "the author"
    ) -> None:
        self._model = model
        self._gate = asyncio.Semaphore(concurrency)
        self._person = person

    async def score(self, answer: str, blocks: tuple[ContextBlock, ...]) -> FaithfulnessResult:
        # Markers such as "[C3]" appear in no excerpt and would penalise cited answers.
        sentences = [
            citations.strip_markers(s).strip()
            for s in citations.split_sentences(answer)
            if citations.is_factual(s)
        ]
        sentences = [s for s in sentences if s]
        if not sentences or not blocks:
            return FaithfulnessResult(verdicts=())

        # Each sentence is judged against its most relevant passages. Judging against
        # the whole context (often 13,000 characters) reported 57% of claims unsupported
        # when the sampled failures were correct. Passages are ranked instead of taken
        # from citations, so uncited baselines are measured the same way.
        passages = _passages(blocks)
        verdicts = await asyncio.gather(
            *(self._judge(s, _relevant(s, passages)) for s in sentences)
        )
        return FaithfulnessResult(verdicts=tuple(verdicts))

    async def judge_one(self, statement: str, excerpt: str) -> bool | None:
        """One verdict, for calibrating the judge against known cases."""
        return (await self._judge(statement, excerpt)).supported

    async def _judge(self, sentence: str, excerpt: str) -> ClaimVerdict:
        async with self._gate:
            try:
                result = await self._model.structured(
                    [
                        Message(
                            role=Role.USER,
                            content=_PROMPT.format(
                                person=self._person,
                                excerpt=excerpt,
                                statement=sentence[:600],
                            ),
                        )
                    ],
                    _Support,
                )
            except Exception:  # noqa: BLE001 - an unreachable judge is not a verdict
                logger.warning("could not judge a claim; recording it as unjudged")
                return ClaimVerdict(sentence=sentence, supported=None)
        return ClaimVerdict(sentence=sentence, supported=result.supported)
