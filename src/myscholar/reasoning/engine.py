"""The adaptive loop: retrieve, grade, generate, verify, correct or refuse.

Replaces the Phase 0 walking skeleton. The signature is unchanged, which was
the point of settling it then.

Ordering is deliberate. The deterministic citation checks run before the
semantic groundedness check, so the cheap checks catch the common failures and
the expensive one only ever sees answers whose citations already resolve.
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
    draft: str = ""
    correction: str | None = None
    premise_correction: str | None = None
    """Set when the question assumes something the sources contradict.

    Carried separately from ``correction``, which is consumed and cleared by
    each regeneration. A false premise stays true across every attempt, so it
    has to survive them."""

    verification: citations.VerificationResult | None = None
    degraded: bool = False
    grounding_failed: bool = False

    ungrounded: tuple[str, ...] = ()
    """Statements the grounding check rejected on the most recent attempt.

    Kept so they can be removed if the loop runs out of attempts, rather than
    being returned with the answer."""

    best_draft: str = ""
    best_ungrounded: tuple[str, ...] = ()
    """The attempt that survived verification most intact, and what it lost.

    Regeneration does not reliably improve a draft. Asked to describe someone's
    strengths in machine learning the loop produced 1272 characters with four
    unsupported statements, then 679 with one, then 730 with three — and
    returned the third, because each attempt overwrote the last. The second was
    the best answer the system ever held and the visitor never saw it.

    Rewriting under a correction is a genuinely harder task than writing: the
    model has to keep what worked, drop what did not, and stay grounded, and
    the instruction that names the failed statements also draws attention back
    to them. So attempts are kept rather than assumed to improve, and the
    fewest-rejections one is what the salvage path works from."""

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
                # A failed generation leaves no draft, and verifying an empty
                # string produces an empty *answer* rather than a refusal — the
                # blank reply a visitor least deserves. `_generate` already
                # treats an outage as "end the loop without a draft"; this makes
                # that end at the refusal, which is what its comment intends.
                # Same guard as `drop_ungrounded` below, for the same reason.
                "generate": lambda state: "verify" if state.draft else "refuse",
                "verify": self._after_verify,
                "check_grounding": self._after_grounding,
                "drop_ungrounded": lambda state: "done" if state.draft else "refuse",
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

    async def _screen(self, state: LoopState) -> LoopState:
        """Check the question's premise. Relevance grading used to happen here.

        The grader was removed on two measurements. On the adversarial set it
        cost +0.05 wrong claims per question and ten points of correct refusal.
        On `tests/eval/confusable_set.yaml` — fifteen questions built
        specifically to give it its best case, where several papers cover the
        subject and one is right — it moved attribution precision by **zero**
        points: four configurations with and without it produced identically
        13/15. The reason is structural rather than a tuning failure.
        Contamination was 0% throughout, so the system never cited the wrong
        member of a pair; both failures were papers retrieval never returned,
        and a filter cannot recover a document it was never given.

        The premise check stays, and is why this node still exists. It is what
        answers a question that asserts something untrue, and it runs on the
        retrieved passages directly now that nothing filters them.
        """
        state = replace(state, budget=state.budget.step())

        # A false premise is not a retrieval failure and retrieving harder
        # cannot fix it: the right passages are already here, and they are what
        # disprove the question. Checked after grading so the check sees only
        # passages judged relevant, and before generation so the answer can
        # correct the question rather than elaborate on it.
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
        # Both, when both apply: the premise correction holds for every attempt
        # while the verification correction is specific to the last one.
        correction = " ".join(c for c in (state.premise_correction, state.correction) if c) or None
        try:
            draft = await self._answerer.generate(
                state.question, state.blocks, correction=correction
            )
        except Exception:
            # A model that times out or dies mid-loop is an outage, not a
            # programming error, and the loop already knows how to end without
            # a draft. Letting it propagate turned a slow generation into an
            # HTTP 500 and a red "something went wrong" in the widget, which is
            # both less useful and less honest than a refusal — and it threw
            # away any verified draft from an earlier attempt.
            #
            logger.exception("generation failed; ending the loop without a new draft")
            budget = state.budget.step()
            return replace(state, budget=budget, correction=None, degraded=True).with_event(
                Stage.GENERATING, "generation unavailable"
            )

        budget = state.budget.step()
        return replace(state, budget=budget, draft=draft, correction=None).with_event(
            Stage.GENERATING, f"{len(draft)} chars"
        )

    async def _verify(self, state: LoopState) -> LoopState:
        result = citations.verify(state.draft, state.blocks)
        budget = state.budget.step()

        # A missing marker is a formatting slip; an invented one is not.
        #
        # These were treated identically: any issue meant regenerate the whole
        # answer, up to three times, then salvage destructively. Measured on
        # one question, a ten-sentence draft citing six papers was reduced to
        # two sentences citing one — and the drafts got shorter each retry as
        # the model shrank to avoid the penalty. With eight to ten factual
        # sentences and roughly a ten percent chance of the model forgetting a
        # marker on any of them, the probability of at least one failure is
        # around sixty percent, so this fired on most long answers.
        #
        # `uncited` now costs the sentence rather than the answer, and no
        # attempt: dropping it leaves everything that *was* attributed intact.
        # `dangling` — a marker pointing at a passage that was never retrieved —
        # still forces a regeneration, because that is the model inventing a
        # source and is the failure the whole mechanism exists to catch.
        # One salvage path, not two. There used to be a severity distinction —
        # `uncited` cost the sentence, `dangling` forced a regeneration — and it
        # existed only to choose between dropping and retrying. With retrying
        # gone the choice is gone, and an invented marker is handled the way a
        # missing one always was: the sentence is removed and the rest stands.
        #
        # The guarantee is unchanged and is the one that matters. A marker
        # pointing at a passage that was never retrieved never reaches the
        # reader. It simply leaves with its sentence instead of triggering
        # another attempt that measurement showed to be worse.
        #
        # Salvaging here rather than at the end is deliberate: salvaging late
        # meant the loop went straight to the terminal node, so the *returned*
        # text was the one piece of output the groundedness check never saw.
        # Re-verifying leaves a draft whose citations resolve, which then routes
        # through grounding like any other.
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
                    state, budget=budget, draft=salvaged, verification=rechecked, correction=None
                ).with_event(Stage.VERIFYING, detail)

        # The summary names the specific failures and is fed back verbatim, so
        # a regeneration fixes those rather than rerolling the same prompt.
        correction = None if result.valid else result.summary()
        return replace(state, budget=budget, verification=result, correction=correction).with_event(
            Stage.VERIFYING,
            "citations valid" if result.valid else f"{len(result.issues)} citation issue(s)",
        )

    async def _drop_ungrounded(self, state: LoopState) -> LoopState:
        """Remove statements the grounding check could not support.

        Previously the
        loop returned the draft unchanged in this situation, which meant a
        claim the system had *already decided was unsupported* was shown to the
        visitor anyway.
        """
        # The attempt that lost least, not the attempt that happened to be last.
        draft, rejected = (
            (state.best_draft, state.best_ungrounded)
            if state.best_draft and len(state.best_ungrounded) < len(state.ungrounded)
            else (state.draft, state.ungrounded)
        )
        kept = [
            sentence
            for sentence in citations.split_sentences(draft)
            if sentence.strip() not in rejected
        ]
        # Same tidy-up as the citation salvage path. Joining the survivors
        # directly leaves the seams showing: an answer that opens "In another
        # work, ..." after the work it referred to has been removed.
        text = citations.rejoin(kept)
        budget = state.budget.step()
        if not text:
            # The judge rejected every sentence of a draft whose citations all
            # resolve. Taken at face value that means refusing, and measured
            # over the question set it pushed over-refusal from 0% to 11.8% —
            # the system declining questions the corpus answers.
            #
            # A model judgement that would delete an entire verified answer is
            # more likely wrong than the answer is. The deterministic checks
            # are the gate: every marker resolves to a passage that was
            # retrieved and graded relevant, and any quotation is real. So the
            # draft stands, marked degraded so the interface can say confidence
            # is lower, rather than being thrown away on one model's opinion.
            return replace(state, budget=budget, degraded=True).with_event(
                Stage.VERIFYING, "grounding uncertain; citations verified"
            )
        return replace(
            state,
            budget=budget,
            draft=text,
            verification=citations.verify(text, state.blocks),
            correction=None,
        ).with_event(Stage.VERIFYING, f"dropped {len(state.ungrounded)} unsupported statement(s)")

    async def _check_grounding(self, state: LoopState) -> LoopState:
        """Check each cited sentence against the block it cites.

        Only runs once citation markers already resolve, so every statement
        here has a real block to check against.
        """
        by_label = {b.label.upper(): b for b in state.blocks}
        unsupported: list[str] = []

        for sentence in citations.split_sentences(state.draft):
            labels = citations.parse_markers(sentence)
            # Only sentences making a checkable claim. See
            # citations.makes_checkable_claim for why this is narrower than
            # "every factual sentence".
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

        # Recorded before regenerating, because the next attempt may be worse
        # and this one is about to be overwritten. Ties go to the earlier
        # attempt, which is the longer one in practice.
        better = not state.best_draft or len(unsupported) < len(state.best_ungrounded)

        quoted = "; ".join(f'"{s[:100]}"' for s in unsupported[:3])
        return replace(
            state,
            ungrounded=tuple(s.strip() for s in unsupported),
            best_draft=state.draft if better else state.best_draft,
            best_ungrounded=tuple(s.strip() for s in unsupported)
            if better
            else state.best_ungrounded,
            budget=budget,
            grounding_failed=True,
            correction=f"These statements are not supported by the excerpts cited: {quoted}",
        ).with_event(Stage.VERIFYING, f"{len(unsupported)} unsupported statement(s)")

    async def _correct(self, state: LoopState) -> LoopState:
        """Try to fix inadequate retrieval, one strategy at a time.

        Strategies are recorded in the budget so none repeats, which is what
        keeps the corrective path from cycling.
        """
        # Empty, always: nothing grades any more, so the rewrite is driven by
        # the question alone. The reason string this yields — "nothing
        # retrieved was relevant" — is exactly right for the one case that
        # still reaches here, which is retrieval returning nothing at all.
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

        if state.budget.reason_for_stopping():
            reason = RefusalReason.BUDGET_EXHAUSTED
        elif state.grounding_failed:
            # Distinct from "nothing relevant found": passages *were* found and
            # the answer built on them could not be supported.
            reason = RefusalReason.UNGROUNDED_AFTER_RETRY
        else:
            reason = RefusalReason.NO_RELEVANT_CONTENT

        # Only asked when the reply is going to be about absence. A question
        # that ran out of clock has established nothing about the corpus, and
        # the call is another model round trip on a budget already overspent —
        # it was adding a second to answers that had just spent ninety.
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
        return "screen" if state.blocks else "correct"

    def _after_screen(self, state: LoopState) -> str:
        """Always generate. `_after_route` has already established there are blocks.

        This used to consult the grader's sufficiency verdict and send an
        insufficient one back to retrieval. With nothing filtering the blocks
        that verdict is always true, so the branch is gone rather than left as
        a condition that cannot fail.
        """
        return "generate"

    def _after_verify(self, state: LoopState) -> str:
        """No retry. Salvage, or refuse.

        Regeneration was removed on measurement, not taste. Rewriting under a
        correction is a harder task than writing, and the instruction naming
        the failed statements draws the model back to them: across the
        adversarial set it cost +0.08 wrong claims per question and twenty
        points of correct refusal, and across the confusable set it cost
        fourteen points of attribution precision. Three measurements, all in
        the same direction, on a step that also made every answer slower.

        `_verify` has already salvaged by this point, so an invalid result here
        means nothing survived.
        """
        result = state.verification
        if result is None or result.valid:
            return "check_grounding"
        return "refuse"

    def _after_grounding(self, state: LoopState) -> str:
        if state.correction is None:
            return "done"
        # Remove what could not be supported rather than returning it, and
        # rather than regenerating: showing a claim the system has already
        # judged unsupported is worse than showing a shorter answer, and
        # trying again was measured to produce a worse answer still.
        return "drop_ungrounded"

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


def _refusal_text(reason: RefusalReason, topics: tuple[str, ...]) -> str:
    """What the visitor reads, decided by why the loop actually stopped.

    All three reasons used to share one sentence — "I don't have anything on
    that in these papers" — which is a claim about the corpus that only one of
    them is entitled to make. The loop was already computing the reason
    correctly and then discarding it here.

    What that cost was not hypothetical. A stalled model call on the
    development machine was pushing one call per question past the reasoning
    deadline, and questions the corpus answers in detail came back saying the
    corpus did not cover them — on roughly every second attempt, with no error
    logged anywhere. A message that had said "I ran out of time" would have
    named the bug the first time anyone saw it, instead of sending three of us
    looking at retrieval, the grader and the prompt in turn.

    Saying so is also just honest. A system that cannot tell "I found nothing"
    apart from "I did not finish looking" is guessing on the visitor's behalf,
    and guessing wrong in the direction that makes the corpus look emptier than
    it is.
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


__all__ = ["Engine", "LoopState", "Refusal"]
