"""A naive RAG baseline for comparison with the full pipeline.

It is a fair baseline, not a straw man:

- dense similarity only, top 5 matches
- neighbouring rows pulled in around each hit
- every passage concatenated into one string
- one model call, with the prompt built as
  ``f"<|content_start>{context} <|content_end> {query}"``
- no system prompt, no citation instruction, no grading, no verification, and
  no way to decline

Three things are deliberately *not* reproduced, and each makes the baseline
stronger rather than weaker:

**The same embedder is used.** The original indexed with one model and queried
with another, which is arithmetically valid and semantically meaningless. Left
in, the comparison would measure that bug rather than the pipeline.

**The same generator is used.** The original ran ``qwen3:0.6b``; this runs
whatever the full system runs. A comparison against a much smaller model would
be a comparison of models.

**The same corpus is used** — properly extracted, sectioned and chunked, with
ligatures repaired. The original's sentence-level rows over raw text would
lose on ingestion quality alone.

What remains is a comparison of *pipeline design*, which is the only
interesting one. Every difference the report shows is attributable to hybrid
retrieval, reranking, parent expansion, grading, citation verification,
grounding and refusal — not to a bigger model or a cleaner corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from myscholar.domain.documents import ContextBlock
from myscholar.ports.llm import ChatModel, Message, Role
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode

#: The naive prompt. No system message, no instructions:
#: the context and the question, concatenated.
NAIVE_PROMPT = "<|content_start>{context} <|content_end> {query}"

#: Phrases a model reaches for when the context does not contain the answer.
#:
#: The naive pipeline has no refusal mechanism, so structurally it always
#: answers. Its *text* sometimes declines anyway, and scoring that as a
#: hallucination would overstate the difference. Detected by pattern so the
#: same rule applies to every configuration.
_TEXTUAL_REFUSAL = re.compile(
    r"\b(do(es)? not (mention|contain|provide|discuss|include|specify|address)"
    r"|no (information|mention|reference|evidence|indication) (of|about|that|regarding)"
    r"|not (mentioned|provided|discussed|specified|found|available|present) in"
    r"|cannot (answer|determine|find|be determined)"
    r"|unable to (answer|determine|find)"
    r"|i (don'?t|do not) (have|know)"
    r"|there is no (mention|information|reference|evidence)"
    r"|the (excerpts?|context|provided text|passages?) (do|does) not"
    # "None of the cited papers reference photovoltaic systems" is as explicit
    # a refusal as exists, and was being scored as a hallucination. Adopted
    # because it is balanced — one answer each for the naive pipeline and the
    # adaptive one — rather than because it improved the favourite.
    r"|none of (the|his|these)\b[^.?!]{0,60}"
    r"\b(reference|mention|discuss|cover|address|study|examine)"
    r"|no (paper|study|work|excerpt)s? (of his )?(reference|mention|discuss|cover|address))",
    re.IGNORECASE,
)

# Deliberately *not* matched: "the excerpts focus on [other topics]".
#
# It is arguably a refusal — listing what the corpus does cover, in place of
# what was asked — and it is the behaviour the nearest-topics design aims for.
# It is left out because it reclassifies one answer, and that answer is the
# adaptive system's: a scoring change that only ever helps the configuration
# being advocated for is one to decline, however good the argument for it.
# The reported refusal rate is therefore conservative.


#: Markdown emphasis, stripped before matching.
#:
#: Not cosmetic. The baseline writes "The provided content does **not mention**
#: lithium battery degradation", and the asterisks fall inside the phrase being
#: matched — so four correct refusals per run were scored as hallucinations,
#: eleven points of correct-refusal rate, entirely against the baseline. The
#: adaptive system's plainer prose was unaffected, which is exactly the shape
#: of an unfair comparison: a measurement artefact that favours the thing being
#: advocated for.
_EMPHASIS = re.compile(r"[*_`]+")


def declines(text: str) -> bool:
    """Whether an answer declines, however it was produced.

    Applied uniformly. A structural refusal and a paragraph saying "the
    excerpts do not mention this" are the same outcome for a reader, and
    scoring them differently would flatter whichever system happens to express
    it structurally.
    """
    return bool(_TEXTUAL_REFUSAL.search(_EMPHASIS.sub("", text)))


#: Markers of an answer that rejects the question's premise rather than
#: declining to engage with it.
#:
#: Needed because refusal is not the only correct response to a false premise,
#: and it is not even the best one. Asked why his NSGA-II study concluded
#: weighting factors are unnecessary, the ideal answer is not "I cannot say" —
#: it is "instead, it used NSGA-II to optimise them, demonstrating their
#: importance". Scored as refusal-or-nothing, that reads as a failure.
#:
#: Applied to every configuration, and it helps the baselines more than the
#: system being advocated for: of the answers this reclassifies, four are the
#: naive pipeline's, eight the retrieval-only one's and five the adaptive
#: system's. A scoring rule that only ever flatters the favourite is a rule to
#: distrust; this one changes the ordering.
_CONTRADICTS = re.compile(
    r"\b(instead\b|in fact\b|on the contrary\b|rather than\b"
    r"|however,? the (excerpts?|paper|text|study)"
    r"|do(es)? not (conclude|state|show|find|report|claim|indicate|suggest|support))",
    re.IGNORECASE,
)


def contradicts(text: str) -> bool:
    """Whether an answer rejects the premise it was handed.

    Distinct from ``declines``: an answer can contradict a false premise while
    answering fully, which is the outcome to aim for. Kept separate so that
    contradiction never counts toward over-refusal — a good answer that happens
    to say "instead" is not a refusal.
    """
    return bool(_CONTRADICTS.search(_EMPHASIS.sub("", text)))


@dataclass(frozen=True, slots=True)
class NaiveAnswer:
    text: str
    blocks: tuple[ContextBlock, ...]


class NaivePipeline:
    """Retrieve, concatenate, generate. No other steps.

    ``mode`` selects which retrieval the naive generator is fed, which is what
    separates the report's first two rows: the same one-shot generation over
    dense-only chunks, and then over hybrid, reranked and expanded context.
    Whatever changes between those rows is retrieval's contribution alone.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        model: ChatModel,
        mode: RetrievalMode = RetrievalMode.DENSE,
        limit: int = 5,
    ) -> None:
        self._retriever = retriever
        self._model = model
        self._mode = mode
        self._limit = limit

    async def answer(self, question: str) -> NaiveAnswer:
        result = await self._retriever.retrieve(question, mode=self._mode, limit=self._limit)
        context = " ".join(block.text for block in result.blocks)
        reply = await self._model.complete(
            [
                Message(
                    role=Role.USER,
                    content=NAIVE_PROMPT.format(context=context, query=question),
                )
            ]
        )
        return NaiveAnswer(text=reply.strip(), blocks=result.blocks)
