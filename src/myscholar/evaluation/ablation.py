"""One rung of the pipeline at a time, so each mechanism can be priced.

The end-to-end comparison says the adaptive pipeline reaches 4.7% unsupported
claims where a naive one reaches 52.0%. It does not say *which part* did that,
and the parts are not free: grading costs a model call per question, the
grounding check costs one per sentence, and regeneration can cost two more
generations. A component that changes nothing should be deleted rather than
described in a README.

So this builds the same pipeline in stages, each adding exactly one mechanism:

    routed          routed retrieval, one generation call, nothing checked
    cited           + deterministic citation verification
    grounded        + grounding check, used to drop sentences, never to retry
    graded          + relevance grading before generation
    graded-strict   grading that also discards `partial`, not only `irrelevant`

Read with `naive` and `hybrid` below it and `adaptive` above it, every delta is
attributable to one change.

Two things this deliberately does *not* do, because they are what `adaptive`
adds and the ladder must not blur them: it never regenerates, and it has no
refusal path of its own. Refusal only happens where routing already decided it
— out of scope, ambiguous — which is shared with every other configuration.

**The measurement hazard, stated here because it is easy to forget when reading
the table.** The grounding check and the faithfulness judge that scores these
runs are the same small model with nearly the same prompt. Configurations from
`grounded` upward are therefore partly optimising against their own marker, and
their advantage on that metric is inflated by an unknown amount. `cited` is the
rung where the judge is least entangled, which is exactly why it is the one
worth hand-labelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from myscholar.domain.answers import Answer, Refusal, Response
from myscholar.domain.documents import ContextBlock
from myscholar.domain.enums import RefusalReason, Verdict
from myscholar.generation import citations
from myscholar.generation.answerer import Answerer
from myscholar.query.router import QueryRouter
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode


@dataclass(frozen=True, slots=True)
class Ablation:
    """Which mechanisms are switched on for one rung."""

    name: str

    route: bool = True
    """Classify, plan, select documents and anchor — or just search.

    False for the rung that isolates the *prompt*. `naive` and `retrieved`
    generate through a template with no instructions at all
    (`"<|content_start>{context}<|content_end>{query}"`), while every rung above
    them uses the seven-rule system prompt. Without a rung that changes only the
    prompt, the large drop between them is unattributable — it was read as
    routing's contribution and could as easily be the instruction to cite every
    sentence, given to a model that previously had none.
    """

    verify_citations: bool = False
    """Deterministic: every marker must resolve, every quote must be real."""

    ground: bool = False
    """Model check per sentence. Here it only ever *drops*, never retries."""

    grade: bool = False
    """Relevance filter over retrieved blocks, before generation."""

    drop_partial: bool = False
    """Discard `partial` blocks as well as `irrelevant`.

    The grader as shipped discards only `irrelevant`, and its prompt tells it to
    use that verdict sparingly — so in practice it discards nothing, and
    measuring it on and off would only confirm that. This rung asks the question
    that matters instead: is the grader unfit for a corpus of near-identical
    papers, or merely tuned to do nothing?
    """


LADDER = (
    Ablation(name="prompted", route=False),
    Ablation(name="routed"),
    Ablation(name="cited", verify_citations=True),
    Ablation(name="grounded", verify_citations=True, ground=True),
    Ablation(name="graded", verify_citations=True, ground=True, grade=True),
    Ablation(
        name="graded-strict",
        verify_citations=True,
        ground=True,
        grade=True,
        drop_partial=True,
    ),
)


class AblationPipeline:
    """Runs one rung. Interface matches Engine.answer_with_context on purpose."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        answerer: Answerer,
        ablation: Ablation,
        retriever: HybridRetriever | None = None,
        context_blocks: int = 8,
    ) -> None:
        self._router = router
        self._answerer = answerer
        self._ablation = ablation
        self._retriever = retriever
        self._context_blocks = context_blocks

    async def answer_with_context(
        self, question: str
    ) -> tuple[Response, tuple[str, ...], tuple[ContextBlock, ...]]:
        if not self._ablation.route:
            # Retrieval only, then the same generator every rung above uses.
            if self._retriever is None:
                msg = "a retriever is required for a rung that does not route"
                raise ValueError(msg)
            searched = await self._retriever.retrieve(
                question, mode=RetrievalMode.HYBRID, limit=self._context_blocks
            )
            blocks = tuple(searched.blocks)
            draft = await self._answerer.generate(question, blocks)
            return (
                Answer(text=draft.strip(), citations=(), confidence=0.5),
                ("retrieve", "generate"),
                blocks,
            )

        path: list[str] = ["route"]
        routed = await self._router.route(question)

        # Classes that routing answers without retrieval are shared with every
        # other configuration, so they pass through unchanged. Generating over
        # them here would make this rung look worse than `adaptive` for a reason
        # that has nothing to do with the mechanism being priced.
        if routed.bibliographic is not None:
            return Answer(text=routed.bibliographic.text, confidence=1.0), ("route",), ()
        if routed.plan.canned_response is not None:
            reason = (
                RefusalReason.NO_RELEVANT_CONTENT
                if routed.plan.needs_clarification
                else RefusalReason.OUT_OF_CORPUS_SCOPE
            )
            return Refusal(reason=reason, text=routed.plan.canned_response), ("route",), ()

        blocks = tuple(routed.blocks)
        if not blocks:
            return (
                Refusal(
                    reason=RefusalReason.NO_RELEVANT_CONTENT,
                    text="I don't have anything on that in these papers.",
                ),
                ("route",),
                (),
            )

        if self._ablation.grade:
            discard = (
                frozenset({Verdict.IRRELEVANT, Verdict.PARTIAL})
                if self._ablation.drop_partial
                else frozenset({Verdict.IRRELEVANT})
            )
            graded = await self._answerer.grade(question, blocks, discard=discard)
            path.append("grade")
            # Keeping nothing would make the rung refuse, which is a behaviour
            # it is not meant to have. Falling back to the retrieved set keeps
            # the comparison about what the grader *filters*.
            blocks = graded.kept or blocks

        draft = await self._answerer.generate(question, blocks)
        path.append("generate")

        result = citations.verify(draft, blocks)
        text = draft
        if self._ablation.verify_citations:
            path.append("verify")
            if not result.valid:
                text = citations.drop_unsupported(draft, result.issues)

        if self._ablation.ground:
            path.append("ground")
            text = await self._drop_ungrounded(text, blocks)

        return (
            Answer(
                text=text.strip(),
                citations=result.citations if self._ablation.verify_citations else (),
                confidence=1.0 if result.valid else 0.5,
            ),
            tuple(path),
            blocks,
        )

    async def _drop_ungrounded(self, text: str, blocks: tuple[ContextBlock, ...]) -> str:
        """Remove sentences the excerpts do not support. No retry, by design."""
        by_label = {b.label.upper(): b for b in blocks}
        kept: list[str] = []
        for sentence in citations.split_sentences(text):
            labels = citations.parse_markers(sentence)
            if not labels or not citations.makes_checkable_claim(sentence):
                kept.append(sentence)
                continue
            excerpt = "\n\n".join(by_label[x].text for x in labels if x in by_label)
            if not excerpt or await self._answerer.is_grounded(
                citations.strip_markers(sentence), excerpt
            ):
                kept.append(sentence)
        return " ".join(kept).strip()


__all__ = ["LADDER", "Ablation", "AblationPipeline"]
