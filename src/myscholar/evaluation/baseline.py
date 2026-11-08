"""A reimplementation of the naive RAG pipeline this project replaces.

Dense retrieval of the top 5 matches and their neighbours, every passage concatenated into
one prompt, one model call, and no system prompt, citations, verification or refusal. It uses
the full system's embedder, generator and indexed corpus, so the comparison measures pipeline
design alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from myscholar.domain.documents import ContextBlock
from myscholar.ports.llm import ChatModel, Message, Role
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode

#: The original prompt: context and question concatenated, with no instructions.
NAIVE_PROMPT = "<|content_start>{context} <|content_end> {query}"

#: Phrases a model uses when the context does not contain the answer.
#:
#: The naive pipeline cannot refuse structurally, but its text sometimes declines. The
#: same pattern is applied to every configuration.
_TEXTUAL_REFUSAL = re.compile(
    r"\b(do(es)? not (mention|contain|provide|discuss|include|specify|address)"
    r"|no (information|mention|reference|evidence|indication) (of|about|that|regarding)"
    r"|not (mentioned|provided|discussed|specified|found|available|present) in"
    # Passive forms of the above: "is not addressed in the given materials".
    r"|(is|was|are|were|has been|have been) (not|never) "
    r"(addressed|mentioned|discussed|covered|found|reported|specified)"
    r"|cannot (answer|determine|find|be determined)"
    r"|unable to (answer|determine|find)"
    r"|i (don'?t|do not) (have|know)"
    r"|there is no (mention|information|reference|evidence)"
    r"|the (excerpts?|context|provided text|passages?) (do|does) not"
    # "None of the cited papers reference photovoltaic systems". This reclassifies
    # one answer each for the naive and adaptive pipelines.
    r"|none of (the|his|these)\b[^.?!]{0,60}"
    r"\b(reference|mention|discuss|cover|address|study|examine)"
    r"|no (paper|study|work|excerpt)s? (of his )?(reference|mention|discuss|cover|address))",
    re.IGNORECASE,
)

# Not matched: "the excerpts focus on [other topics]". It would reclassify only one
# answer, the adaptive system's, so the reported refusal rate is conservative.


#: Markdown emphasis, stripped before matching. Asterisks inside "does **not mention**"
#: broke the match and scored four correct baseline refusals per run as hallucinations.
_EMPHASIS = re.compile(r"[*_`]+")


def declines(text: str) -> bool:
    """Whether an answer's text declines to answer.

    Applied to every configuration, so a textual refusal counts the same as a structural one.
    """
    return bool(_TEXTUAL_REFUSAL.search(_EMPHASIS.sub("", text)))


#: Markers of an answer that rejects the question's premise.
#:
#: The best answer to a false premise often corrects it ("instead, it used NSGA-II to
#: optimise them"). Of the answers this reclassifies, four are the naive pipeline's, eight
#: the retrieval-only one's and five the adaptive system's.
_CONTRADICTS = re.compile(
    r"\b(instead\b|in fact\b|on the contrary\b|rather than\b"
    r"|however,? the (excerpts?|paper|text|study)"
    r"|do(es)? not (conclude|state|show|find|report|claim|indicate|suggest|support))",
    re.IGNORECASE,
)


def contradicts(text: str) -> bool:
    """Whether an answer rejects the premise it was handed.

    Separate from ``declines``, so a full answer that contradicts a false premise never
    counts as over-refusal.
    """
    return bool(_CONTRADICTS.search(_EMPHASIS.sub("", text)))


@dataclass(frozen=True, slots=True)
class NaiveAnswer:
    text: str
    blocks: tuple[ContextBlock, ...]


class NaivePipeline:
    """Retrieve, concatenate, generate.

    ``mode`` selects the retrieval: dense-only for the naive row, hybrid for the
    retrieval-only row, so the difference between them is retrieval's alone.
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
