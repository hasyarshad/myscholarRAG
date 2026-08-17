"""Describing each paper in the words a reader would use.

Chunk-level retrieval answers "which passage says X". It cannot reliably
answer "which of these papers is about X", and the difference is not academic:

    Q: how would you rate Haseeb's contribution to machine learning?
    A: he applied machine learning to analyse survey data for an association.

That answer was correct about its sources and wrong about the corpus. Six of
his papers apply machine learning; none of them says so. They say LFBLSTM,
AFL-1DCNN, gradient-boosted meta-learner, unified feature fusion. The one
document that says "machine learning" on every page is a *survey* of learning
algorithms, so it won every slot, and the only passage tying the person to the
phrase was a line on the leadership page about a membership survey.

No amount of query rewriting fixes that, and it was measured rather than
assumed: five transform combinations were tried and none retrieved the applied
papers, because the gap is between the question's vocabulary and the corpus's,
not between one phrasing and another.

A summary closes it. Written per document, in ordinary terminology, and
embedded separately, it lets a document be found for what it *is* rather than
for what one of its passages happens to say.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from myscholar.ports.embedding import Embedder
from myscholar.ports.llm import ChatModel, Message, Role

logger = logging.getLogger(__name__)

#: How much of the paper the summariser reads. The abstract, introduction and
#: methods carry what a paper is about; the rest is detail that a summary would
#: leave out anyway.
_MAX_SOURCE_CHARS = 8000


class PaperSummary(BaseModel):
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
#: The model was asked and would not say. Given a paper called "An Overview of
#: Sequential Learning Algorithms for Single Hidden Layer Networks: Current
#: Issues & Future Trends", a 4B summariser still reported it as original work,
#: and the assistant then credited Haseeb with inventing GGAP-RBF.
#:
#: The title is the more reliable signal and cannot drift. The model's verdict
#: is kept as well and the two are OR-ed: a survey whose title hides it is
#: still worth catching, and the cost of a false positive is only that the
#: assistant declines to credit him with something.
_REVIEW_TITLE = re.compile(
    r"\b(a\s+)?(overview|review|survey|taxonomy|tutorial|state[\s-]of[\s-]the[\s-]art|"
    r"comparative\s+stud(y|ies)|systematic\s+(review|literature)|current\s+issues|"
    r"recent\s+(advances|trends)|literature\s+review)\b",
    re.IGNORECASE,
)


def looks_like_review(title: str) -> bool:
    """Whether the title announces a survey of other people's work."""
    return bool(_REVIEW_TITLE.search(title))


@dataclass(frozen=True, slots=True)
class Summary:
    document_id: object
    text: str
    embedding: list[float]


def render(summary: PaperSummary, *, title: str, is_review: bool | None = None) -> str:
    """Flatten the structured summary into the text that gets embedded.

    The title is included: it is the most reliable sentence about a paper, and
    the summary is a description of *this* document rather than free-floating
    prose.
    """
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
        """One summary, or None if the model could not produce one.

        A failure is not fatal. A document without a summary is simply absent
        from document-level selection and still reachable by chunk search,
        which is how the system behaved before summaries existed.
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
