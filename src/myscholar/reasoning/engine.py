"""The answering loop: route, check the premise, generate, verify, then answer or refuse.

The deterministic citation checks run before the model-based grounding check, so the
grounding check only sees sentences whose citations already resolve.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace

from myscholar.domain.answers import Answer, Refusal, Response, TraceEvent
from myscholar.domain.documents import ContextBlock
from myscholar.domain.enums import RefusalReason, Stage
from myscholar.domain.narration import describe_class, describe_sources
from myscholar.generation import citations
from myscholar.generation.answerer import Answerer, GradedContext
from myscholar.query.router import QueryRouter
from myscholar.reasoning.graph import Budget, Graph, Trace

logger = logging.getLogger(__name__)

REWRITE = "rewrite"
BROADEN = "broaden"


@dataclass(frozen=True, slots=True)
class LoopState:
    """Everything the loop carries. Frozen; nodes return a new one."""

    question: str
    budget: Budget

    search_text: str = ""
    """The text sent to retrieval: the question, or a rewrite of it."""

    blocks: tuple[ContextBlock, ...] = ()
    draft: str = ""
    premise_correction: str | None = None
    """Set when the question assumes something the sources contradict; passed to the
    generator as an instruction."""

    verification: citations.VerificationResult | None = None
    degraded: bool = False
    grounding_failed: bool = False

    ungrounded: tuple[str, ...] = ()
    """Statements the grounding check rejected, removed before the answer is returned."""

    response: Response | None = None
    events: tuple[TraceEvent, ...] = field(default_factory=tuple)

    def with_event(self, stage: Stage, detail: str = "") -> LoopState:
        return replace(
            self,
            events=(*self.events, TraceEvent(stage=stage, detail=detail[:120])),
        )


class Engine:
    """Answers questions against the indexed corpus."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        answerer: Answerer,
        budget: Budget | None = None,
        short_name: str = "him",
    ) -> None:
        self._router = router
        self._answerer = answerer
        self._budget = budget or Budget()
        self._name = short_name
        self._listeners: list[asyncio.Queue[TraceEvent | None]] = []
        self._graph = self._build()

    async def answer(self, question: str) -> Response:
        response, _ = await self.answer_with_trace(question)
        return response

    async def answer_with_trace(self, question: str) -> tuple[Response, Trace]:
        response, trace, _ = await self.answer_with_context(question)
        return response, trace

    async def answer_with_context(
        self, question: str
    ) -> tuple[Response, Trace, tuple[ContextBlock, ...]]:
        """The response, the path taken, and the passages the generator was given.

        The passages are exposed for evaluation: faithfulness is judged against what
        the generator saw, which the citations alone do not capture.
        """
        # The stored budget is a template; a fresh copy restarts the clock and counters.
        state, trace = await self._graph.run(
            LoopState(question=question, search_text=question, budget=self._budget.fresh())
        )
        return self._finish(state, trace), trace, state.blocks

    async def stream_events(self, question: str) -> AsyncIterator[TraceEvent | Answer | Refusal]:
        """Yield each stage as it happens, then the finished response.

        Tokens are not streamed because verification runs after generation and can
        remove sentences.
        """
        queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
        self._listeners.append(queue)
        task = asyncio.create_task(self.answer_with_trace(question))

        try:
            while True:
                pending = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({pending, task}, return_when=asyncio.FIRST_COMPLETED)
                if pending in done:
                    if (event := pending.result()) is not None:
                        yield event
                    continue

                pending.cancel()
                # Drain anything published between the last read and completion.
                while not queue.empty():
                    if (event := queue.get_nowait()) is not None:
                        yield event
                break

            response, _ = await task
            yield response
        finally:
            self._listeners.remove(queue)
            if not task.done():
                task.cancel()

    def _publishing(
        self, node: Callable[[LoopState], Awaitable[LoopState]]
    ) -> Callable[[LoopState], Awaitable[LoopState]]:
        """Wrap a node so the events it adds reach stream listeners as soon as it finishes."""

        async def wrapped(state: LoopState) -> LoopState:
            before = len(state.events)
            result = await node(state)
            for event in result.events[before:]:
                for queue in self._listeners:
                    queue.put_nowait(event)
            return result

        return wrapped

    def _build(self) -> Graph[LoopState]:
        return Graph[LoopState](
            nodes={
                name: self._publishing(node)
                for name, node in {
                    "route": self._route,
                    "screen": self._screen,
                    "generate": self._generate,
                    "verify": self._verify,
                    "check_grounding": self._check_grounding,
                    "drop_ungrounded": self._drop_ungrounded,
                    "correct": self._correct,
                    "refuse": self._refuse,
                    "done": self._done,
                }.items()
            },
            edges={
                "route": self._after_route,
                "screen": self._after_screen,
                # Without a draft, verification would produce an empty answer
                # instead of a refusal.
                "generate": lambda state: "verify" if state.draft else "refuse",
                "verify": self._after_verify,
                "check_grounding": self._after_grounding,
                "drop_ungrounded": lambda state: "done" if state.draft else "refuse",
                "correct": self._after_correct,
            },
            entry="route",
            on_budget_exhausted="refuse",
        )

    async def _route(self, state: LoopState) -> LoopState:
        routed = await self._router.route(state.search_text)
        budget = state.budget.step().retrieved()

        # Bibliographic, out-of-scope and ambiguous questions are answered by the
        # router and skip the rest of the loop.
        if routed.bibliographic is not None:
            return replace(
                state,
                budget=budget,
                response=Answer(text=routed.bibliographic.text, confidence=1.0),
            ).with_event(Stage.DONE, "answered from metadata")

        if routed.plan.canned_response is not None:
            reason = (
                RefusalReason.OUT_OF_CORPUS_SCOPE
                if not routed.plan.needs_clarification
                else RefusalReason.NO_RELEVANT_CONTENT
            )
            return replace(
                state,
                budget=budget,
                response=Refusal(reason=reason, text=routed.plan.canned_response),
            ).with_event(
                Stage.DONE, describe_class(routed.classification.query_class, short_name=self._name)
            )

        # Named from the retrieved blocks, not the query class: the retriever always
        # runs an unrestricted branch, so a biographical question can return papers.
        sources = describe_sources((b.document_kind, b.document_id) for b in routed.blocks)
        detail = f"{len(routed.blocks)} passages" + (f" from {sources}" if sources else "")
        return replace(
            state, budget=budget, blocks=routed.blocks, degraded=routed.degraded
        ).with_event(Stage.RETRIEVING, detail)

    async def _screen(self, state: LoopState) -> LoopState:
        """Check whether the question rests on a premise the retrieved passages contradict.

        A relevance grader was measured here and removed: it added wrong claims on the
        adversarial set and left attribution precision unchanged on the confusable set.
        """
        state = replace(state, budget=state.budget.step())

        # Checked before generation so the answer corrects the question instead of
        # explaining something that did not happen.
        premise = await self._answerer.check_premise(state.question, state.blocks)
        if premise is None:
            return state

        return replace(
            state,
            premise_correction=(
                f"The question takes for granted that {premise.premise} "
                f"The excerpts do not support that: {premise.correction} "
                f"Open by saying plainly that the premise is wrong, state what the "
                f"excerpts do say, and do not explain the thing that did not happen."
            ),
        ).with_event(Stage.GRADING, f"false premise: {premise.premise[:80]}")

    async def _generate(self, state: LoopState) -> LoopState:
        try:
            draft = await self._answerer.generate(
                state.question, state.blocks, correction=state.premise_correction
            )
        except Exception:
            # A timeout or model error is an outage: end without a draft, which
            # routes to a refusal instead of an HTTP 500.
            logger.exception("generation failed; ending the loop without a draft")
            budget = state.budget.step()
            return replace(state, budget=budget, degraded=True).with_event(
                Stage.GENERATING, "generation unavailable"
            )

        budget = state.budget.step()
        return replace(state, budget=budget, draft=draft).with_event(
            Stage.GENERATING, f"{len(draft)} chars"
        )

    async def _verify(self, state: LoopState) -> LoopState:
        result = citations.verify(state.draft, state.blocks)
        budget = state.budget.step()

        # A sentence that fails a citation check (no marker, a marker for a passage
        # that was never retrieved, or a misquote) is removed and the rest re-verified.
        # The salvaged draft then goes through the grounding check like any other.
        if not result.valid:
            salvaged = citations.drop_unsupported(state.draft, result.issues).strip()
            if salvaged:
                rechecked = citations.verify(salvaged, state.blocks)
                detail = (
                    f"dropped {len(result.issues)} uncited sentence(s)"
                    if all(issue.kind == "uncited" for issue in result.issues)
                    else f"{len(result.issues)} citation issue(s); kept what verified"
                )
                return replace(
                    state, budget=budget, draft=salvaged, verification=rechecked
                ).with_event(Stage.VERIFYING, detail)

        return replace(state, budget=budget, verification=result).with_event(
            Stage.VERIFYING,
            "citations valid" if result.valid else f"{len(result.issues)} citation issue(s)",
        )

    async def _drop_ungrounded(self, state: LoopState) -> LoopState:
        """Remove statements the grounding check could not support."""
        kept = [
            sentence
            for sentence in citations.split_sentences(state.draft)
            if sentence.strip() not in state.ungrounded
        ]
        text = citations.rejoin(kept)
        budget = state.budget.step()
        if not text:
            # The judge rejected every sentence of a draft whose citations all resolve.
            # Refusing here raised over-refusal from 0% to 11.8% on the question set,
            # so the draft is kept and marked degraded.
            return replace(state, budget=budget, degraded=True).with_event(
                Stage.VERIFYING, "grounding uncertain; citations verified"
            )
        return replace(
            state,
            budget=budget,
            draft=text,
            verification=citations.verify(text, state.blocks),
        ).with_event(Stage.VERIFYING, f"dropped {len(state.ungrounded)} unsupported statement(s)")

    async def _check_grounding(self, state: LoopState) -> LoopState:
        """Check each cited sentence against the blocks it cites.

        Runs only after the citation markers resolve, so every cited block exists.
        """
        by_label = {b.label.upper(): b for b in state.blocks}
        unsupported: list[str] = []

        for sentence in citations.split_sentences(state.draft):
            labels = citations.parse_markers(sentence)
            if not labels or not citations.makes_checkable_claim(sentence):
                continue
            excerpt = "\n\n".join(by_label[label].text for label in labels if label in by_label)
            if excerpt and not await self._answerer.is_grounded(
                citations.strip_markers(sentence), excerpt
            ):
                unsupported.append(sentence)

        budget = state.budget.step()
        if not unsupported:
            return replace(state, budget=budget).with_event(Stage.VERIFYING, "grounded")

        return replace(
            state,
            ungrounded=tuple(s.strip() for s in unsupported),
            budget=budget,
            grounding_failed=True,
        ).with_event(Stage.VERIFYING, f"{len(unsupported)} unsupported statement(s)")

    async def _correct(self, state: LoopState) -> LoopState:
        """Search again after retrieval returned nothing, one strategy at a time.

        The budget records each strategy tried, so none repeats.
        """
        # Nothing is graded in the loop, so the rewrite has no rejection reasons.
        graded = GradedContext(kept=(), rejected=())
        budget = state.budget.step()

        if not budget.has_tried(REWRITE):
            rewritten = await self._answerer.rewrite(state.question, graded)
            budget = budget.tried(REWRITE)
            if rewritten:
                return replace(state, budget=budget, search_text=rewritten).with_event(
                    Stage.RETRIEVING, f"rewritten: {rewritten[:60]}"
                )

        if not budget.has_tried(BROADEN):
            # The original question again, which reaches different passages than the rewrite.
            budget = budget.tried(BROADEN)
            return replace(state, budget=budget, search_text=state.question).with_event(
                Stage.RETRIEVING, "broadened"
            )

        return replace(state, budget=budget).with_event(Stage.RETRIEVING, "no strategies left")

    async def _refuse(self, state: LoopState) -> LoopState:
        if state.response is not None:
            return state

        # A draft with verified citations is returned instead of a refusal, even when
        # the budget ran out.
        if state.draft and state.verification is not None:
            salvaged = (
                state.draft
                if state.verification.valid
                else citations.drop_unsupported(state.draft, state.verification.issues)
            )
            if salvaged.strip():
                return replace(
                    state,
                    budget=state.budget.step(),
                    response=Answer(
                        text=salvaged.strip(),
                        citations=state.verification.citations,
                        confidence=0.5,
                        degraded=state.degraded,
                    ),
                ).with_event(Stage.DONE, "returned a verified draft rather than refusing")

        if state.budget.reason_for_stopping():
            reason = RefusalReason.BUDGET_EXHAUSTED
        elif state.grounding_failed:
            # Passages were found, but the answer built on them could not be supported.
            reason = RefusalReason.UNGROUNDED_AFTER_RETRY
        else:
            reason = RefusalReason.NO_RELEVANT_CONTENT

        # Skipped when the budget ran out: nothing has been established about the
        # corpus, and the call would add another model round trip.
        topics: tuple[str, ...] = ()
        if reason is not RefusalReason.BUDGET_EXHAUSTED:
            topics = await self._answerer.nearest_topics(state.blocks)

        return replace(
            state,
            budget=state.budget.step(),
            response=Refusal(
                reason=reason, text=_refusal_text(reason, topics), nearest_topics=topics
            ),
        ).with_event(Stage.DONE, reason.value)

    async def _done(self, state: LoopState) -> LoopState:
        # A response set earlier (the bibliographic answer from the route node) stands.
        if state.response is not None:
            return state

        verification = state.verification
        citations_found = verification.citations if verification else ()

        text = state.draft
        if verification and not verification.valid:
            text = citations.drop_unsupported(text, verification.issues)

        return replace(
            state,
            budget=state.budget.step(),
            response=Answer(
                text=text.strip(),
                citations=citations_found,
                confidence=1.0 if verification and verification.valid else 0.5,
                degraded=state.degraded,
            ),
        ).with_event(Stage.DONE)

    def _after_route(self, state: LoopState) -> str:
        if state.response is not None:
            return "done" if isinstance(state.response, Answer) else "refuse"
        return "screen" if state.blocks else "correct"

    def _after_screen(self, state: LoopState) -> str:
        """Always generate: `_after_route` only sends states with blocks here."""
        return "generate"

    def _after_verify(self, state: LoopState) -> str:
        """Check grounding if the citations verify, otherwise refuse.

        `_verify` has already salvaged what it could. Regenerating under a correction
        was measured and removed: it added wrong claims and lowered both correct
        refusal and attribution precision (results/generation_phase1.md, confusable.md).
        """
        result = state.verification
        if result is None or result.valid:
            return "check_grounding"
        return "refuse"

    def _after_grounding(self, state: LoopState) -> str:
        return "drop_ungrounded" if state.grounding_failed else "done"

    def _after_correct(self, state: LoopState) -> str:
        return "route" if state.budget.can_retrieve_again else "refuse"

    def _finish(self, state: LoopState, trace: Trace) -> Response:
        response = state.response or Refusal(
            reason=RefusalReason.NO_RELEVANT_CONTENT,
            text="I don't have anything on that in these papers.",
        )
        events = (
            *state.events,
            *(
                TraceEvent(stage=Stage.DONE, detail=name, elapsed_ms=ms)
                for name, ms in trace.timings_ms[-1:]
            ),
        )
        return replace(response, trace=events)


def _join(items: tuple[str, ...]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _refusal_text(reason: RefusalReason, topics: tuple[str, ...]) -> str:
    """The refusal message for the reason the loop stopped.

    Only ``NO_RELEVANT_CONTENT`` says the corpus lacks the answer, so a timeout is
    not mistaken for a gap in the corpus.
    """
    if reason is RefusalReason.BUDGET_EXHAUSTED:
        return (
            "I ran out of time working through that one before I could finish. "
            "Asking again usually gets an answer."
        )
    if reason is RefusalReason.UNGROUNDED_AFTER_RETRY:
        text = "I found passages on that, but could not support an answer from them."
        if topics:
            text += " They come from work on " + _join(topics) + "."
        return text
    text = "I don't have anything on that in these papers."
    if topics:
        text += " The closest work here is on " + _join(topics) + "."
    return text


__all__ = ["Engine", "LoopState"]
