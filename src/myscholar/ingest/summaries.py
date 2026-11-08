"""Describing each paper in the words a reader would use.

Chunk retrieval finds passages but not which paper is about a topic. Six papers here apply
machine learning while naming only LFBLSTM, AFL-1DCNN and the like, and none of five
query-transform combinations retrieved them. A per-document summary in ordinary terms,
embedded separately, lets document selection find them.
"""

from __future__ import annotations

import logging
import re

from pydantic import Field

from myscholar.ports.embedding import Embedder
from myscholar.ports.llm import ChatModel, Message, ModelOutput, Role

logger = logging.getLogger(__name__)

#: How much of the paper the summariser reads; the abstract, introduction and
#: methods come first and carry what the paper is about.
_MAX_SOURCE_CHARS = 8000


class PaperSummary(ModelOutput):
    """The four things worth knowing about a paper before opening it."""

    topic: str = Field(description="What the paper is about, in one sentence.")
    methods: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Techniques used. Give both the specific name and the ordinary one: "
            "'LFBLSTM (recurrent neural network)', 'NSGA-II (genetic algorithm)'."
        ),
    )
    fields: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Broad areas this belongs to, as a reader would name them: machine learning, "
            "deep learning, power electronics, control theory, signal processing."
        ),
    )
    findings: str = Field(default="", description="What it found or achieved, in one sentence.")
    is_review: bool = Field(
        default=False,
        description=(
            "True if this paper surveys, reviews or compares methods that other researchers "
            "proposed, rather than proposing something of its own."
        ),
    )


_PROMPT = """Summarise this paper so that someone searching a collection could \
recognise it.

Name the techniques twice over: once as the paper names them, and once in the \
ordinary words a reader would use. A paper that uses an "AFL-1DCNN" is doing \
deep learning; a paper that tunes with "NSGA-II" is doing multi-objective \
optimisation. Someone asking about machine learning must be able to find the \
first, and they will not search for "AFL-1DCNN".

Say whether the paper proposes something of its own or surveys what others \
have done. A review describes other researchers' methods, and crediting those \
to its author is the worst error this system can make.

Be accurate. This is used to route searches, so a field the paper does not \
belong to sends questions to the wrong paper.

Title: {title}

{text}
/no_think"""


#: Titles that announce a survey.
#:
#: The summariser model reported "An Overview of Sequential Learning Algorithms ..."
#: as original work. The title check is OR-ed with the model's verdict; a false
#: positive only stops the assistant crediting the author with a method.
_REVIEW_TITLE = re.compile(
    r"\b(a\s+)?(overview|review|survey|taxonomy|tutorial|state[\s-]of[\s-]the[\s-]art|"
    r"comparative\s+stud(y|ies)|systematic\s+(review|literature)|current\s+issues|"
    r"recent\s+(advances|trends)|literature\s+review)\b",
    re.IGNORECASE,
)


def looks_like_review(title: str) -> bool:
    """Whether the title announces a survey of other people's work."""
    return bool(_REVIEW_TITLE.search(title))


def render(summary: PaperSummary, *, title: str, is_review: bool | None = None) -> str:
    """Flatten the structured summary into the text that gets embedded, led by the title."""
    parts = [title, summary.topic]
    if summary.is_review if is_review is None else is_review:
        parts.append("A review of methods proposed by other researchers.")
    if summary.fields:
        parts.append("Fields: " + ", ".join(summary.fields) + ".")
    if summary.methods:
        parts.append("Methods: " + ", ".join(summary.methods) + ".")
    if summary.findings:
        parts.append(summary.findings)
    return " ".join(p.strip() for p in parts if p and p.strip())


class Summariser:
    def __init__(self, *, model: ChatModel, embedder: Embedder) -> None:
        self._model = model
        self._embedder = embedder

    async def summarise(self, *, title: str, text: str) -> tuple[str, bool] | None:
        """The rendered summary and whether the paper is a review, or None on failure.

        A document without a summary is absent from document selection but still
        reachable by chunk search.
        """
        try:
            result = await self._model.structured(
                [
                    Message(
                        role=Role.USER,
                        content=_PROMPT.format(title=title, text=text[:_MAX_SOURCE_CHARS]),
                    )
                ],
                PaperSummary,
            )
        except Exception:  # noqa: BLE001 - one failed summary must not fail a run
            logger.warning("could not summarise %r", title[:60])
            return None
        is_review = result.is_review or looks_like_review(title)
        rendered = render(result, title=title, is_review=is_review)
        return (rendered, is_review) if rendered else None

    async def embed(self, summaries: list[str]) -> list[list[float]]:
        return await self._embedder.embed_documents(summaries)
