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
    r"|the (excerpts?|context|provided text|passages?) (do|does) not)",
    re.IGNORECASE,
)


def declines(text: str) -> bool:
    """Whether an answer declines, however it was produced.

    Applied uniformly. A structural refusal and a paragraph saying "the
    excerpts do not mention this" are the same outcome for a reader, and
    scoring them differently would flatter whichever system happens to express
    it structurally.
    """
    return bool(_TEXTUAL_REFUSAL.search(text))


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
