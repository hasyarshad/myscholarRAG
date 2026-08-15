"""The adaptive loop: retrieve, grade, generate, verify, correct or refuse.

Replaces the Phase 0 walking skeleton. The signature is unchanged, which was
the point of settling it then.

Ordering is deliberate. The deterministic citation checks run before the
semantic groundedness check, so the cheap checks catch the common failures and
the expensive one only ever sees answers whose citations already resolve.

See docs/learning/05-adaptive-rag-and-hallucination.md.
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
    """What retrieval is actually given — the question, or a rewrite of it."""

    blocks: tuple[ContextBlock, ...] = ()
    graded: GradedContext | None = None
    draft: str = ""
    correction: str | None = None
    verification: citations.VerificationResult | None = None
    degraded: bool = False
    grounding_failed: bool = False

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
        """The response, the path taken, and the passages the model was given.

        The context is exposed for evaluation: judging whether a claim is
        supported means judging it against what the generator actually saw.
        Reconstructing that from the citations is not equivalent — a citation
        carries a quote at most, and the loop may have retrieved, graded and
        broadened several times before settling.
        """
        # A fresh budget per question. The stored one is a template of limits;
        # reusing the instance would carry the previous question's elapsed time
        # and step count into this one.
        state, trace = await self._graph.run(
            LoopState(question=question, search_text=question, budget=self._budget.fresh())
        )
        return self._finish(state, trace), trace, state.blocks

    async def stream_events(self, question: str) -> AsyncIterator[TraceEvent | Answer | Refusal]:
        """Yield each stage as it happens, then the finished response.

        Stages rather than tokens: verification runs after generation, so raw
        output would show the visitor text that is subsequently rejected.
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

    # -- graph ------------------------------------------------------------

    def _publishing(
        self, node: Callable[[LoopState], Awaitable[LoopState]]
    ) -> Callable[[LoopState], Awaitable[LoopState]]:
        """Wrap a node so the events it adds are published as it finishes.

        Without this the API can only replay the trace once the answer is
        complete, so a visitor watches a generic spinner for twenty seconds and
        then receives every stage at once. Streaming progress is the point.
        """

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
                    "grade": self._grade,
                    "generate": self._generate,
                    "verify": self._verify,
                    "check_grounding": self._check_grounding,
                    "correct": self._correct,
                    "refuse": self._refuse,
                    "done": self._done,
                }.items()
            },
            edges={
                "route": self._after_route,
                "grade": self._after_grade,
                "generate": lambda _: "verify",
                "verify": self._after_verify,
                "check_grounding": self._after_grounding,
                "correct": self._after_correct,
            },
            entry="route",
            on_budget_exhausted="refuse",
        )

    # -- nodes ------------------------------------------------------------

    async def _route(self, state: LoopState) -> LoopState:
        routed = await self._router.route(state.search_text)
        budget = state.budget.step().retrieved()

        # Phase 3 already answers three classes without retrieval. Those results
        # pass straight through rather than entering the loop.
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

        # Names what was actually searched, not what the plan intended. The
        # retriever always runs one unrestricted branch, so a biographical
        # question can and does return paper passages — a label predicted from
        # the query class would be wrong exactly when it mattered.
        sources = describe_sources((b.document_kind, b.document_id) for b in routed.blocks)
        detail = f"{len(routed.blocks)} passages" + (f" from {sources}" if sources else "")
        return replace(
            state, budget=budget, blocks=routed.blocks, degraded=routed.degraded
        ).with_event(Stage.RETRIEVING, detail)

    async def _grade(self, state: LoopState) -> LoopState:
        graded = await self._answerer.grade(state.question, state.blocks)
        return replace(
            state, budget=state.budget.step(), graded=graded, blocks=graded.kept
        ).with_event(Stage.GRADING, f"{len(graded.kept)} kept, {len(graded.rejected)} dropped")

    async def _generate(self, state: LoopState) -> LoopState:
        try:
            draft = await self._answerer.generate(
                state.question, state.blocks, correction=state.correction
            )
        except Exception:
            # A model that times out or dies mid-loop is an outage, not a
            # programming error, and the loop already knows how to end without
            # a draft. Letting it propagate turned a slow generation into an
            # HTTP 500 and a red "something went wrong" in the widget, which is
            # both less useful and less honest than a refusal — and it threw
            # away any verified draft from an earlier attempt.
            #
            # Spending the whole regeneration budget stops the loop retrying an
            # unavailable model until the wall clock runs out.
            logger.exception("generation failed; ending the loop without a new draft")
            budget = state.budget.step().exhaust_regenerations()
            return replace(state, budget=budget, correction=None, degraded=True).with_event(
                Stage.GENERATING, "generation unavailable"
            )

        budget = state.budget.step()
        if state.correction:
            budget = budget.regenerated()
        return replace(state, budget=budget, draft=draft, correction=None).with_event(
            Stage.GENERATING, f"{len(draft)} chars"
        )

    async def _verify(self, state: LoopState) -> LoopState:
        result = citations.verify(state.draft, state.blocks)
        # The summary names the specific failures and is fed back verbatim, so
        # a regeneration fixes those rather than rerolling the same prompt.
        correction = None if result.valid else result.summary()
        return replace(
            state, budget=state.budget.step(), verification=result, correction=correction
        ).with_event(
            Stage.VERIFYING,
            "citations valid" if result.valid else f"{len(result.issues)} citation issue(s)",
        )

    async def _check_grounding(self, state: LoopState) -> LoopState:
        """Check each cited sentence against the block it cites.

        Only runs once citation markers already resolve, so every statement
        here has a real block to check against.
        """
        by_label = {b.label.upper(): b for b in state.blocks}
        unsupported: list[str] = []

        for sentence in citations.split_sentences(state.draft):
            labels = citations.parse_markers(sentence)
            if not labels or not citations.is_factual(sentence):
                continue
            excerpt = "\n\n".join(by_label[label].text for label in labels if label in by_label)
            if excerpt and not await self._answerer.is_grounded(
                citations.strip_markers(sentence), excerpt
            ):
                unsupported.append(sentence)

        budget = state.budget.step()
        if not unsupported:
            return replace(state, budget=budget).with_event(Stage.VERIFYING, "grounded")

        quoted = "; ".join(f'"{s[:100]}"' for s in unsupported[:3])
        return replace(
            state,
            budget=budget,
            grounding_failed=True,
            correction=f"These statements are not supported by the excerpts cited: {quoted}",
        ).with_event(Stage.VERIFYING, f"{len(unsupported)} unsupported statement(s)")

    async def _correct(self, state: LoopState) -> LoopState:
        """Try to fix inadequate retrieval, one strategy at a time.

        Strategies are recorded in the budget so none repeats, which is what
        keeps the corrective path from cycling.
        """
        graded = state.graded or GradedContext(kept=(), rejected=())
        budget = state.budget.step()

        if not budget.has_tried(REWRITE):
            rewritten = await self._answerer.rewrite(state.question, graded)
            budget = budget.tried(REWRITE)
            if rewritten:
                return replace(state, budget=budget, search_text=rewritten).with_event(
                    Stage.RETRIEVING, f"rewritten: {rewritten[:60]}"
                )

        if not budget.has_tried(BROADEN):
            # Broaden by searching the original question again with no
            # rewriting, which reaches different passages than the rewrite did.
            budget = budget.tried(BROADEN)
            return replace(state, budget=budget, search_text=state.question).with_event(
                Stage.RETRIEVING, "broadened"
            )

        return replace(state, budget=budget).with_event(Stage.RETRIEVING, "no strategies left")

    async def _refuse(self, state: LoopState) -> LoopState:
        if state.response is not None:
            return state

        # Running out of budget while holding a verified answer is the worst
        # possible outcome: the work is done and correct, and throwing it away
        # produces a refusal the corpus did not deserve. Measured, this was
        # three of seven answerable questions — the groundedness check kept
        # asking for a regeneration until the deadline passed, and the drafts
        # it rejected had valid citations throughout.
        #
        # So a refusal only happens when there is nothing usable to return.
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

        topics = await self._answerer.nearest_topics(state.blocks)
        if state.budget.reason_for_stopping():
            reason = RefusalReason.BUDGET_EXHAUSTED
        elif state.grounding_failed:
            # Distinct from "nothing relevant found": passages *were* found and
            # the answer built on them could not be supported.
            reason = RefusalReason.UNGROUNDED_AFTER_RETRY
        else:
            reason = RefusalReason.NO_RELEVANT_CONTENT

        text = "I don't have anything on that in these papers."
        if topics:
            text += " The closest work here is on " + _join(topics) + "."

        return replace(
            state,
            budget=state.budget.step(),
            response=Refusal(reason=reason, text=text, nearest_topics=topics),
        ).with_event(Stage.DONE, reason.value)

    async def _done(self, state: LoopState) -> LoopState:
        # A response set earlier stands. The bibliographic path answers in the
        # route node and routes straight here; rebuilding from the (empty)
        # draft silently replaced "4 papers in 2023." with an empty string.
        #
        # This regressed once when an unrelated `git checkout` reverted the
        # fix, so it is covered by a test rather than only a comment.
        if state.response is not None:
            return state

        verification = state.verification
        citations_found = verification.citations if verification else ()

        text = state.draft
        if verification and not verification.valid:
            # Last resort: keep the sound sentences, drop the rest. If that
            # leaves nothing, the refusal node handles it.
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

    # -- edges ------------------------------------------------------------

    def _after_route(self, state: LoopState) -> str:
        if state.response is not None:
            return "done" if isinstance(state.response, Answer) else "refuse"
        return "grade" if state.blocks else "correct"

    def _after_grade(self, state: LoopState) -> str:
        if state.graded and state.graded.sufficient:
            return "generate"
        return "correct" if state.budget.can_retrieve_again else "refuse"

    def _after_verify(self, state: LoopState) -> str:
        result = state.verification
        if result is None or result.valid:
            return "check_grounding"
        if state.budget.can_regenerate:
            return "generate"
        # Out of regenerations: salvage what verified, or refuse if that is
        # nothing. An answer with a dangling citation is never returned.
        return "done" if citations.drop_unsupported(state.draft, result.issues) else "refuse"

    def _after_grounding(self, state: LoopState) -> str:
        if state.correction is None:
            return "done"
        return "generate" if state.budget.can_regenerate else "done"

    def _after_correct(self, state: LoopState) -> str:
        return "route" if state.budget.can_retrieve_again else "refuse"

    # -- assembly ---------------------------------------------------------

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
        if isinstance(response, Answer):
            return replace(response, trace=events)
        return replace(response, trace=events)


def _join(items: tuple[str, ...]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


__all__ = ["Engine", "LoopState", "Refusal"]
