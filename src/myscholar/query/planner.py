"""Turning a classification into a retrieval strategy.

The whole point of classifying is that different questions want different
treatment. This module is that mapping, kept as data rather than scattered
through the retrieval code, so what fires when is readable in one place and
changeable without touching the retriever.

See docs/learning/04-query-translation-and-routing.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myscholar.domain.enums import QueryClass, SectionKind
from myscholar.query.classifier import Classification
from myscholar.query.transforms import TransformKind

#: Sections excluded from every search. References are a quarter of the corpus
#: and answer nothing.
_ALWAYS_EXCLUDE = frozenset({SectionKind.REFERENCES, SectionKind.ACKNOWLEDGEMENTS})


@dataclass(frozen=True, slots=True)
class Plan:
    """What to do with a question."""

    query_class: QueryClass
    transforms: tuple[TransformKind, ...] = ()

    retrieve: bool = True
    """False for classes answered without touching the index."""

    use_sql: bool = False
    """True for bibliographic questions, answered from the metadata table."""

    canned_response: str | None = None
    """Set for out-of-scope questions, which are answered without generation."""

    needs_clarification: bool = False

    top_k: int = 50
    rerank_k: int = 8
    max_chunks_per_document: int | None = None
    """Caps how much of the context one paper may occupy.

    Without it a verbose paper monopolises a synthesis question, and the answer
    silently becomes about that paper rather than about the corpus.
    """

    filter_document_titles: tuple[str, ...] = ()
    boost_sections: frozenset[SectionKind] = frozenset()
    exclude_sections: frozenset[SectionKind] = field(default_factory=lambda: _ALWAYS_EXCLUDE)

    @property
    def is_terminal(self) -> bool:
        """Whether the plan answers without retrieval or generation."""
        return self.canned_response is not None


def out_of_scope_reply(name: str = "this researcher") -> str:
    return (
        f"I only answer questions about {name}'s research and background. "
        f"Ask me about the papers — the methods used, what they found, or when "
        f"and where they were published."
    )


def ambiguous_reply() -> str:
    return "Which paper do you mean? A title, a topic or a year would let me answer precisely."


OUT_OF_SCOPE_REPLY = out_of_scope_reply()
AMBIGUOUS_REPLY = ambiguous_reply()


class QueryPlanner:
    """Maps a classification to a plan."""

    def __init__(
        self,
        *,
        top_k: int = 50,
        rerank_k: int = 8,
        person_name: str = "this researcher",
    ) -> None:
        self._top_k = top_k
        self._rerank_k = rerank_k
        self._out_of_scope = out_of_scope_reply(person_name)

    def plan(self, classification: Classification) -> Plan:
        titles = tuple(classification.entities.paper_titles)

        match classification.query_class:
            case QueryClass.OUT_OF_SCOPE:
                # No retrieval, no generation. Retrieval would return the five
                # chunks nearest to "what's the weather", which are arbitrary,
                # and a helpful model would then use them.
                return Plan(
                    query_class=QueryClass.OUT_OF_SCOPE,
                    retrieve=False,
                    canned_response=self._out_of_scope,
                )

            case QueryClass.AMBIGUOUS:
                return Plan(
                    query_class=QueryClass.AMBIGUOUS,
                    retrieve=False,
                    canned_response=AMBIGUOUS_REPLY,
                    needs_clarification=True,
                )

            case QueryClass.BIBLIOGRAPHIC:
                # SQL over verified metadata. No passage states the answer, so
                # retrieving one guarantees a fabricated number.
                return Plan(
                    query_class=QueryClass.BIBLIOGRAPHIC,
                    retrieve=False,
                    use_sql=True,
                )

            case QueryClass.SINGLE_PAPER_FACTUAL:
                # Narrow. The target is a specific fact, so expansion techniques
                # add noise: HyDE would invent a different specific fact.
                return Plan(
                    query_class=QueryClass.SINGLE_PAPER_FACTUAL,
                    transforms=(TransformKind.NORMALISE,),
                    top_k=self._top_k,
                    rerank_k=self._rerank_k,
                    filter_document_titles=titles,
                )

            case QueryClass.CROSS_PAPER_SYNTHESIS:
                # Widest budget, and the only class that caps per-document
                # coverage, because the answer must span papers.
                return Plan(
                    query_class=QueryClass.CROSS_PAPER_SYNTHESIS,
                    transforms=(
                        TransformKind.NORMALISE,
                        TransformKind.MULTI_QUERY,
                        TransformKind.DECOMPOSE,
                    ),
                    top_k=self._top_k,
                    rerank_k=10,
                    max_chunks_per_document=4,
                )

            case QueryClass.METHODOLOGICAL:
                # Step-back and HyDE both help here: a hypothetical methods
                # paragraph shares vocabulary with a real one.
                return Plan(
                    query_class=QueryClass.METHODOLOGICAL,
                    transforms=(
                        TransformKind.NORMALISE,
                        TransformKind.STEP_BACK,
                        TransformKind.HYDE,
                    ),
                    top_k=self._top_k,
                    rerank_k=self._rerank_k,
                    boost_sections=frozenset(
                        {SectionKind.METHODS, SectionKind.EXPERIMENTS, SectionKind.RESULTS}
                    ),
                )

            case QueryClass.BIOGRAPHICAL:
                return Plan(
                    query_class=QueryClass.BIOGRAPHICAL,
                    transforms=(TransformKind.NORMALISE,),
                    top_k=self._top_k,
                    rerank_k=self._rerank_k,
                    boost_sections=frozenset(
                        {
                            SectionKind.CV_SUMMARY,
                            SectionKind.CV_EDUCATION,
                            SectionKind.CV_EMPLOYMENT,
                            SectionKind.CV_AWARDS,
                        }
                    ),
                )
