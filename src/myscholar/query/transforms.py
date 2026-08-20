"""Rewriting a question into the queries actually sent to retrieval.

Each transform is a small unit that turns one question into one or more search
queries. The planner decides which fire; none decides for itself. Applying all
of them to every question multiplies latency and, for some classes, actively
damages results — HyDE on a counting question searches for a fabricated number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from myscholar.ports.llm import ChatModel, Message, Role


class TransformKind(StrEnum):
    NORMALISE = "normalise"
    MULTI_QUERY = "multi_query"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    HYDE = "hyde"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """One query to send to retrieval, and where it came from."""

    text: str
    origin: TransformKind
    weight: float = 1.0
    """Scales this query's contribution when results are fused."""


class Variations(BaseModel):
    queries: list[str] = Field(max_length=4, description="Alternative phrasings, one per line")


class SubQuestions(BaseModel):
    questions: list[str] = Field(max_length=4, description="Self-contained sub-questions")


class StepBack(BaseModel):
    question: str = Field(description="A more general question the specific one falls under")


class Hypothetical(BaseModel):
    passage: str = Field(max_length=800, description="A plausible passage answering the question")


_FILLER = re.compile(
    r"^\s*(can\s+you|could\s+you|please|i\s+(want|would\s+like)\s+to\s+know|"
    r"tell\s+me|do\s+you\s+know|i'?m\s+curious)\b[\s,:]*",
    re.IGNORECASE,
)


def normalise(question: str) -> str:
    """Strip conversational filler. Always applied, costs nothing.

    "Can you tell me how the weighting factors are tuned?" and "how are the
    weighting factors tuned?" should embed to nearly the same vector, but the
    filler words are real tokens that dilute a short query.
    """
    # Applied repeatedly: "Can you tell me how ..." stacks two filler phrases,
    # and stripping once leaves "tell me how ...".
    text = question.strip()
    for _ in range(4):
        stripped = _FILLER.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"\s{2,}", " ", text)
    return text or question.strip()


_MULTI_QUERY_PROMPT = """Rewrite this search query in {n} different ways.

Keep the meaning identical. Vary the vocabulary — use synonyms and alternative
technical phrasings, since the documents may not use the same words as the
question. Do not answer the question.

Query: "{question}\""""

_DECOMPOSE_PROMPT = """Break this question into the smaller questions needed to answer it.

Each must be self-contained and answerable on its own. Produce at most {n}.
If the question is already simple, return it unchanged as the only item.

Question: "{question}\""""

_STEP_BACK_PROMPT = """Write the more general question that this specific one falls under.

Example: "what loss function does he use for imbalanced data?" steps back to
"how does he handle class imbalance?".

Question: "{question}\""""

#: HyDE, grounded in what the corpus actually contains.
#:
#: Ungrounded, it invents a different research area. Asked to rate Haseeb's
#: contribution to machine learning it wrote a passage about "a lightweight
#: convolutional neural network for real-time object detection in low-bandwidth
#: environments" — fluent, plausible, and describing a paper that does not
#: exist in any corpus, let alone this one. The retrieved passages were
#: correspondingly wrong.
#:
#: The titles fix that for the same reason they fix entity linking in the
#: classifier: twenty of them is a few hundred tokens, and it turns "guess what
#: a paper on this might say" into "guess which of these papers says it". The
#: hypothetical is still invented — that is the technique — but it is invented
#: in the corpus's own vocabulary, which is the vocabulary being searched.
_HYDE_PROMPT = """Write one paragraph of a research paper. Do not answer, judge
or comment on the question — write the paragraph.

These are the papers in the collection. Use their vocabulary: the specific
methods, models and techniques their titles imply.

{titles}

The paragraph should read like a methods or results section that would contain
the answer to this question. Invented numbers and details are fine; this text is
never shown to anyone and is used only to find real passages by similarity. If
the question is evaluative, write about the work itself rather than about how
good it is.

Question: "{question}"

Paragraph:"""


class QueryTransformer:
    """Applies the transforms a plan asks for."""

    def __init__(self, model: ChatModel | None = None, *, titles: list[str] | None = None) -> None:
        self._model = model
        self._titles = titles or []

    def set_titles(self, titles: list[str]) -> None:
        """Refresh the corpus titles used to ground hypothetical passages."""
        self._titles = titles

    async def apply(
        self,
        question: str,
        kinds: tuple[TransformKind, ...],
        *,
        multi_query_count: int = 3,
        max_sub_questions: int = 3,
    ) -> list[SearchQuery]:
        """Produce the queries to search with.

        The normalised original is always included, so a transform failing or
        producing something useless can only add noise, never remove the
        question that was actually asked.
        """
        base = normalise(question)
        queries = [SearchQuery(text=base, origin=TransformKind.NORMALISE)]

        if self._model is None:
            return queries

        if TransformKind.MULTI_QUERY in kinds:
            queries += [
                SearchQuery(text=text, origin=TransformKind.MULTI_QUERY, weight=0.7)
                for text in await self._variations(base, multi_query_count)
            ]

        if TransformKind.DECOMPOSE in kinds:
            queries += [
                SearchQuery(text=text, origin=TransformKind.DECOMPOSE, weight=0.9)
                for text in await self._sub_questions(base, max_sub_questions)
            ]

        if TransformKind.STEP_BACK in kinds and (broader := await self._step_back(base)):
            queries.append(SearchQuery(text=broader, origin=TransformKind.STEP_BACK, weight=0.6))

        if TransformKind.HYDE in kinds and (passage := await self._hypothetical(base)):
            queries.append(SearchQuery(text=passage, origin=TransformKind.HYDE, weight=0.8))

        # Distinct text only: duplicates would double a query's weight in the
        # fusion without adding information.
        seen: set[str] = set()
        unique: list[SearchQuery] = []
        for query in queries:
            key = query.text.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(query)
        return unique

    async def _variations(self, question: str, count: int) -> list[str]:
        result = await self._structured(
            _MULTI_QUERY_PROMPT.format(n=count, question=question), Variations
        )
        return [q.strip() for q in (result.queries if result else [])][:count]

    async def _sub_questions(self, question: str, count: int) -> list[str]:
        result = await self._structured(
            _DECOMPOSE_PROMPT.format(n=count, question=question), SubQuestions
        )
        parts = [q.strip() for q in (result.questions if result else [])][:count]
        # A decomposition returning only the original adds nothing.
        return [] if len(parts) <= 1 else parts

    async def _step_back(self, question: str) -> str | None:
        result = await self._structured(_STEP_BACK_PROMPT.format(question=question), StepBack)
        return result.question.strip() if result else None

    async def _hypothetical(self, question: str) -> str | None:
        result = await self._structured(
            _HYDE_PROMPT.format(
                question=question,
                titles="\n".join(f"- {t}" for t in self._titles) or "(unknown)",
            ),
            Hypothetical,
        )
        return result.passage.strip() if result else None

    async def _structured[T: BaseModel](self, prompt: str, schema: type[T]) -> T | None:
        """Run one transform. A failure drops that transform, not the request."""
        if self._model is None:
            return None
        try:
            return await self._model.structured([Message(role=Role.USER, content=prompt)], schema)
        except Exception:  # noqa: BLE001 - a failed transform degrades, never fails
            return None
