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
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from myscholar.domain.documents import ContextBlock
from myscholar.generation import citations
from myscholar.ports.llm import ChatModel, Message, Role

logger = logging.getLogger(__name__)


class _Support(BaseModel):
    supported: bool = Field(description="True only if the excerpts state or directly imply this.")


#: The judge is given the same premise the generator had.
#:
#: Without it the judge is strictly *harsher* than the system it is measuring.
#: The CV's awards section is a bare list — "Jun 2025 Best Paper Presentation
#: Award IEEE ISIE" — with no subject, so "he received the Best Paper
#: Presentation Award" is unsupported by the literal text and every correct
#: CV-derived claim scored as a hallucination. The generator was told whose
#: corpus this is, in its system prompt and in each block's header. Measuring
#: it against a stricter premise measures the wrong thing.
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


#: Roughly a paragraph. Small enough that a 4B model can hold all of it in
#: view, large enough that a claim's evidence is rarely split across two.
_PASSAGE_CHARS = 700

#: How much context one verdict sees. Three passages: the best match plus two
#: neighbours, so a claim supported by an adjacent sentence is not missed.
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

    Each piece keeps its document and section label. The generator sees that
    header on every block, and a passage stripped of it loses the attribution
    that makes a bare list readable — "Awards" is what tells you the list is of
    awards and whose they are.
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
    """The passages most likely to contain this sentence's evidence.

    Ranked by shared content words. Crude next to an embedding, and adequate:
    a claim and the text supporting it overlap heavily in the terms that
    matter, which is exactly what the lexical retrieval branch relies on.
    """
    if len(passages) <= _PASSAGES_PER_CLAIM:
        return "\n\n".join(passages)

    terms = _content_words(sentence)
    ranked = sorted(passages, key=lambda p: len(terms & _content_words(p)), reverse=True)
    return "\n\n".join(ranked[:_PASSAGES_PER_CLAIM])


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

    def __init__(
        self, model: ChatModel, *, concurrency: int = 4, person: str = "the author"
    ) -> None:
        self._model = model
        self._gate = asyncio.Semaphore(concurrency)
        self._person = person

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

        # Each sentence is judged against the passages most likely to contain
        # its evidence, not against the whole context.
        #
        # The first version concatenated every block and truncated to 6000
        # characters. Parent expansion routinely produces 13,000, so the
        # evidence was often simply cut — and even when it survived, a 4B judge
        # asked to find one supporting clause in six thousand characters of
        # dense engineering prose says "no". Measured, that reported 57% of the
        # adaptive system's claims as unsupported while the sampled failures
        # were all correct, cited and verbatim from the CV. The instrument was
        # broken, not the system.
        #
        # Passages are ranked rather than cited blocks used, so this works
        # identically for the baseline, which emits no citations. Scoring a
        # cited configuration against its own citations and an uncited one
        # against everything would not be the same measurement.
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
