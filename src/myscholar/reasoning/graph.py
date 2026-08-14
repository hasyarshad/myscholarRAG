"""A small state machine with a hard budget.

Hand-rolled rather than taken from a framework, for two reasons: the semantics
needed here are about 120 lines, and a compiled graph's transition table cannot
be property-tested the way a plain dispatch dict can.

The budget is the point. A self-correcting loop retries when it is unhappy with
its own output, and a question it will never answer makes it unhappy forever.
Three independent bounds catch three different runaways, and any one of them
forces the terminal state.

See docs/learning/05-adaptive-rag-and-hallucination.md.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

type NodeName = str


class GraphError(RuntimeError):
    """The graph is malformed — an edge points at a node that does not exist."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Bounds guaranteeing the loop halts.

    Independent rather than a single counter, because each catches a different
    runaway: a cycle between two nodes, a query that keeps failing grading, a
    model that keeps producing ungrounded text, and anything unexpectedly slow.
    """

    max_steps: int = 24
    max_retrieval_rounds: int = 3
    max_regenerations: int = 2
    deadline_seconds: float = 45.0

    steps_used: int = 0
    retrieval_rounds_used: int = 0
    regenerations_used: int = 0
    strategies_tried: frozenset[str] = field(default_factory=frozenset)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def exhausted(self) -> bool:
        return self.steps_used >= self.max_steps or self.elapsed_seconds >= self.deadline_seconds

    @property
    def can_retrieve_again(self) -> bool:
        return self.retrieval_rounds_used < self.max_retrieval_rounds and not self.exhausted

    @property
    def can_regenerate(self) -> bool:
        return self.regenerations_used < self.max_regenerations and not self.exhausted

    def fresh(self) -> Budget:
        """A budget with the same limits and the counters reset.

        The deadline is measured from ``started_at``, which is set when the
        Budget is constructed. An Engine holds one Budget for its lifetime, so
        reusing it across questions means the clock never restarts — the second
        question inherits the first one's elapsed time, and after 45 seconds of
        uptime every question is refused as budget-exhausted before it begins.
        """
        return replace(
            self,
            steps_used=0,
            retrieval_rounds_used=0,
            regenerations_used=0,
            strategies_tried=frozenset(),
            started_at=time.monotonic(),
        )

    def step(self) -> Budget:
        return replace(self, steps_used=self.steps_used + 1)

    def retrieved(self) -> Budget:
        return replace(self, retrieval_rounds_used=self.retrieval_rounds_used + 1)

    def regenerated(self) -> Budget:
        return replace(self, regenerations_used=self.regenerations_used + 1)

    def exhaust_regenerations(self) -> Budget:
        """Spend the whole regeneration allowance at once.

        Used when generation fails outright rather than producing something
        flawed. Retrying an unavailable model just burns the wall clock, and
        the loop's exits are all expressed in terms of the budget, so saying
        "there are no attempts left" is how a node declines to continue.
        """
        return replace(self, regenerations_used=self.max_regenerations)

    def tried(self, strategy: str) -> Budget:
        """Record a corrective strategy, so none is attempted twice."""
        return replace(self, strategies_tried=self.strategies_tried | {strategy})

    def has_tried(self, strategy: str) -> bool:
        return strategy in self.strategies_tried

    def reason_for_stopping(self) -> str | None:
        if self.steps_used >= self.max_steps:
            return f"step budget exhausted after {self.steps_used} steps"
        if self.elapsed_seconds >= self.deadline_seconds:
            return f"deadline reached after {self.elapsed_seconds:.1f}s"
        return None


class HasBudget(Protocol):
    """State the driver can meter. Nodes may carry anything else they like.

    Declared as a read-only property rather than an attribute so that frozen
    dataclasses satisfy it — a mutable attribute in a protocol requires the
    implementer to be settable, which is exactly what the states are not.
    """

    @property
    def budget(self) -> Budget: ...


type Node[S] = Callable[[S], Awaitable[S]]
type Edge[S] = Callable[[S], NodeName]


@dataclass(frozen=True, slots=True)
class Trace:
    """Which nodes ran, in order, and how long each took."""

    path: tuple[str, ...] = ()
    timings_ms: tuple[tuple[str, int], ...] = ()
    stopped_because: str | None = None

    @property
    def terminated_early(self) -> bool:
        return self.stopped_because is not None


class Graph[S: HasBudget]:
    """A dispatch table over async node functions.

    ``nodes`` maps a name to the work; ``edges`` maps a name to a function
    choosing what runs next. A node with no edge is terminal.
    """

    def __init__(
        self,
        *,
        nodes: Mapping[NodeName, Node[S]],
        edges: Mapping[NodeName, Edge[S]],
        entry: NodeName,
        on_budget_exhausted: NodeName,
    ) -> None:
        if entry not in nodes:
            msg = f"entry node {entry!r} is not defined"
            raise GraphError(msg)
        if on_budget_exhausted not in nodes:
            msg = f"budget-exhausted node {on_budget_exhausted!r} is not defined"
            raise GraphError(msg)
        for name in edges:
            if name not in nodes:
                msg = f"edge defined for unknown node {name!r}"
                raise GraphError(msg)

        self._nodes = dict(nodes)
        self._edges = dict(edges)
        self._entry = entry
        self._exhausted = on_budget_exhausted

    async def run(self, state: S) -> tuple[S, Trace]:
        """Execute until a terminal node, or until the budget runs out.

        Termination is guaranteed by the step and deadline bounds, which are
        checked before every node. An edge returning an unknown name is a
        programming error and raises rather than silently stopping.
        """
        path: list[str] = []
        timings: list[tuple[str, int]] = []
        current = self._entry
        stopped: str | None = None

        while True:
            if (reason := state.budget.reason_for_stopping()) is not None:
                stopped = reason
                # One final pass through the terminal node, so the caller gets
                # a well-formed refusal rather than a partial state.
                #
                # Keyed on whether it has actually *run*, not on whether the
                # machine happens to be pointing at it. An earlier version
                # compared against `current` and so skipped the node whenever
                # the budget tripped in the same step that routed to it —
                # returning a state that had never reached a terminal node.
                if not path or path[-1] != self._exhausted:
                    state = await self._run_node(self._exhausted, state, path, timings)
                break

            state = await self._run_node(current, state, path, timings)

            if current not in self._edges:
                break

            following = self._edges[current](state)
            if following not in self._nodes:
                msg = f"node {current!r} routed to unknown node {following!r}"
                raise GraphError(msg)
            current = following

        return state, Trace(path=tuple(path), timings_ms=tuple(timings), stopped_because=stopped)

    async def _run_node(
        self,
        name: NodeName,
        state: S,
        path: list[str],
        timings: list[tuple[str, int]],
    ) -> S:
        started = time.monotonic()
        state = await self._nodes[name](state)
        path.append(name)
        timings.append((name, int((time.monotonic() - started) * 1000)))
        return state
