"""Turning a classification into a retrieval strategy.

Kept as data in one place, so what runs for each class can be read and changed without
touching the retriever.
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
    """Set for out-of-scope and ambiguous questions, which are answered without generation."""

    needs_clarification: bool = False

    top_k: int = 50
    context_blocks: int = 8
    select_documents: int = 0
    """How many documents to choose by summary before searching passages.

    Zero searches the whole corpus, which suits a narrow factual question. Selection
    suits questions about the corpus ("what has he done in X"), where the paper that
    repeats the question's words is often not the one that answers it.
    """

    anchor_to_top_document: bool = False
    """Bring the best-matching document's passages to the front, for single-paper questions.

    The anchor is the top block's document, not the highest summed score: on "How are
    weighting factors tuned in FCS-MPTC paper?" the top block came from the right
    paper, but another paper supplied three of the top five blocks.
    """

    max_chunks_per_document: int | None = None
    """Caps how many blocks one paper may contribute to the context."""

    boost_sections: frozenset[SectionKind] = frozenset()
    exclude_sections: frozenset[SectionKind] = field(default_factory=lambda: _ALWAYS_EXCLUDE)

    @property
    def is_terminal(self) -> bool:
        """Whether the plan answers without retrieval or generation."""
        return self.canned_response is not None


def out_of_scope_reply(name: str = "this researcher") -> str:
    return (
        f"I only answer questions about {name}'s research and background. "
        f"Ask me about the papers: the methods used, what they found, or when "
        f"and where they were published."
    )


def ambiguous_reply() -> str:
    return "Which paper do you mean? A title, a topic or a year would let me answer precisely."


AMBIGUOUS_REPLY = ambiguous_reply()


class QueryPlanner:
    """Maps a classification to a plan."""

    def __init__(
        self,
        *,
        top_k: int = 50,
        context_blocks: int = 8,
        person_name: str = "this researcher",
    ) -> None:
        self._top_k = top_k
        self._context_blocks = context_blocks
        self._out_of_scope = out_of_scope_reply(person_name)

    def plan(self, classification: Classification) -> Plan:
        match classification.query_class:
            case QueryClass.OUT_OF_SCOPE:
                # No retrieval or generation: the chunks nearest an off-topic
                # question are arbitrary, and the model would use them.
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
                # SQL over verified metadata. No passage states a count, so a
                # generated answer would invent one.
                return Plan(
                    query_class=QueryClass.BIBLIOGRAPHIC,
                    retrieve=False,
                    use_sql=True,
                )

            case QueryClass.SINGLE_PAPER_FACTUAL:
                # Narrow: expansion adds noise, and HyDE would invent a different
                # specific fact. Anchoring handles shorthand such as "the MMC paper".
                return Plan(
                    query_class=QueryClass.SINGLE_PAPER_FACTUAL,
                    transforms=(TransformKind.NORMALISE,),
                    top_k=self._top_k,
                    context_blocks=self._context_blocks,
                    anchor_to_top_document=True,
                )

            case QueryClass.CROSS_PAPER_SYNTHESIS:
                # The answer must span papers, so at most two blocks per document
                # (at least five papers in ten slots). With a cap of four, a survey
                # that repeats "machine learning" took 13 of 22 passages.
                #
                # HyDE writes in the corpus's vocabulary ("stacking ensembles",
                # "LFBLSTM") instead of the question's. On a machine-learning
                # question the literal query returned six of ten passages from the
                # survey; the hypothetical passage returned the applied papers.
                return Plan(
                    query_class=QueryClass.CROSS_PAPER_SYNTHESIS,
                    transforms=(
                        TransformKind.NORMALISE,
                        TransformKind.MULTI_QUERY,
                        TransformKind.DECOMPOSE,
                        TransformKind.HYDE,
                    ),
                    top_k=self._top_k,
                    context_blocks=10,
                    # Nine papers plus the two slots the selector reserves for the
                    # CV and site pages. At nine in total, selection recall fell
                    # from 88% to 79%.
                    select_documents=11,
                    max_chunks_per_document=2,
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
                    context_blocks=self._context_blocks,
                    # Narrower than synthesis: six papers plus the two reserved slots.
                    select_documents=8,
                    # Uncapped, one paper took all eight slots on a machine-learning
                    # question. Three per document guarantees at least three papers.
                    max_chunks_per_document=3,
                    boost_sections=frozenset(
                        {SectionKind.METHODS, SectionKind.EXPERIMENTS, SectionKind.RESULTS}
                    ),
                )

            case QueryClass.BIOGRAPHICAL:
                return Plan(
                    query_class=QueryClass.BIOGRAPHICAL,
                    transforms=(TransformKind.NORMALISE,),
                    top_k=self._top_k,
                    context_blocks=self._context_blocks,
                    boost_sections=frozenset(
                        {
                            SectionKind.CV_SUMMARY,
                            SectionKind.CV_EDUCATION,
                            SectionKind.CV_EMPLOYMENT,
                            SectionKind.CV_AWARDS,
                        }
                    ),
                )
