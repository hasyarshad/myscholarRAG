"""A small state machine with a hard budget.

Hand-rolled: the semantics fit in about 120 lines, and a plain dispatch dict is easy to
property-test. Any of the budget's bounds forces the terminal state.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

type NodeName = str


class GraphError(RuntimeError):
    """The graph is malformed, such as an edge pointing at an undefined node."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Bounds that guarantee the loop halts.

    The step count catches cycles, retrieval rounds cap re-searching, and the
    deadline catches slow model calls.
    """

    max_steps: int = 24
    max_retrieval_rounds: int = 3
    deadline_seconds: float = 45.0

    steps_used: int = 0
    retrieval_rounds_used: int = 0
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

    def fresh(self) -> Budget:
        """A budget with the same limits and the counters reset.

        ``started_at`` is set at construction, so a budget reused across questions
        would carry over the previous question's elapsed time.
        """
        return replace(
            self,
            steps_used=0,
            retrieval_rounds_used=0,
            strategies_tried=frozenset(),
            started_at=time.monotonic(),
        )

    def step(self) -> Budget:
        return replace(self, steps_used=self.steps_used + 1)

    def retrieved(self) -> Budget:
        return replace(self, retrieval_rounds_used=self.retrieval_rounds_used + 1)

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
    """State the driver can meter.

    ``budget`` is a read-only property so that frozen dataclasses satisfy the protocol.
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

        The budget is checked before every node. An edge returning an unknown
        node name raises ``GraphError``.
        """
        path: list[str] = []
        timings: list[tuple[str, int]] = []
        current = self._entry
        stopped: str | None = None

        while True:
            if (reason := state.budget.reason_for_stopping()) is not None:
                stopped = reason
                # Run the budget-exhausted node once, unless it was the last node
                # to run, so the returned state is always terminal.
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
