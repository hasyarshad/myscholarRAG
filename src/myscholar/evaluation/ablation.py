"""The pipeline built up one mechanism at a time, so each one's effect can be measured.

    prompted        hybrid retrieval and the system prompt, no routing
    routed          routed retrieval, one generation call, nothing checked
    cited           + deterministic citation verification
    grounded        + grounding check, used to drop sentences, never to retry
    graded          + relevance grading before generation
    graded-strict   grading that discards `partial` as well as `irrelevant`

No rung has a refusal path of its own; refusals come only from routing, as in every
configuration. The grounding check and the faithfulness judge use the same small model with
similar prompts, so faithfulness from `grounded` upward is inflated by an unknown amount.
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
    """Classify, plan, select documents and anchor, or just search.

    False for the rung that isolates the prompt: the naive baselines use a template with no
    instructions, while every routed rung also uses the system prompt.
    """

    verify_citations: bool = False
    """Deterministic: every marker must resolve, every quote must be real."""

    ground: bool = False
    """Model check per sentence. Here it only drops sentences."""

    grade: bool = False
    """Relevance filter over retrieved blocks, before generation."""

    drop_partial: bool = False
    """Discard `partial` blocks as well as `irrelevant`.

    The default grader discards only `irrelevant`, which its prompt reserves for clear
    mismatches, so it rarely discards anything. This rung tests stricter grading.
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
    """Runs one rung, with the same interface as ``Engine.answer_with_context``."""

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

        # Classes that routing answers without retrieval pass through unchanged,
        # as in every other configuration.
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
            # If grading keeps nothing, use the retrieved set: this rung never refuses.
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
        """Remove sentences the excerpts do not support, without retrying."""
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
