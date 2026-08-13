"""The state machine, and the property that matters: it always halts.

A worked example proves one path terminates. A property test covers the paths
nobody thought of — which, in a self-correcting loop, is where the infinite
retry lives.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.reasoning.graph import Budget, Graph, GraphError


@dataclass(frozen=True, slots=True)
class State:
    budget: Budget
    verdicts: tuple[bool, ...] = ()
    """Scripted grading outcomes, consumed one per visit."""

    visits: int = 0
    done: bool = False


def counting(name: str) -> Callable[[State], Awaitable[State]]:
    async def node(state: State) -> State:
        return replace(state, budget=state.budget.step(), visits=state.visits + 1)

    node.__name__ = name
    return node


async def terminal(state: State) -> State:
    return replace(state, budget=state.budget.step(), done=True)


class TestBudget:
    def test_a_fresh_budget_permits_work(self) -> None:
        budget = Budget()

        assert not budget.exhausted
        assert budget.can_retrieve_again
        assert budget.can_regenerate

    def test_the_step_bound_stops_it(self) -> None:
        budget = Budget(max_steps=3)
        for _ in range(3):
            budget = budget.step()

        assert budget.exhausted
        assert "step budget" in (budget.reason_for_stopping() or "")

    def test_the_retrieval_bound_is_independent_of_steps(self) -> None:
        """Each bound catches a different runaway."""
        budget = Budget(max_retrieval_rounds=2).retrieved().retrieved()

        assert not budget.can_retrieve_again
        assert not budget.exhausted

    def test_the_regeneration_bound_is_independent(self) -> None:
        budget = Budget(max_regenerations=1).regenerated()

        assert not budget.can_regenerate
        assert not budget.exhausted

    def test_the_deadline_stops_it_regardless_of_steps(self) -> None:
        budget = Budget(deadline_seconds=0.0)

        assert budget.exhausted
        assert "deadline" in (budget.reason_for_stopping() or "")

    def test_a_strategy_is_recorded_so_it_is_not_repeated(self) -> None:
        """Corrective strategies are tried once each, or the loop cycles."""
        budget = Budget().tried("rewrite")

        assert budget.has_tried("rewrite")
        assert not budget.has_tried("broaden")

    def test_the_budget_is_immutable(self) -> None:
        original = Budget()

        original.step().retrieved().regenerated()

        assert original.steps_used == 0


class TestGraph:
    async def test_a_linear_path_runs_to_the_end(self) -> None:
        graph = Graph[State](
            nodes={"a": counting("a"), "b": counting("b"), "end": terminal},
            edges={"a": lambda _: "b", "b": lambda _: "end"},
            entry="a",
            on_budget_exhausted="end",
        )

        state, trace = await graph.run(State(budget=Budget()))

        assert trace.path == ("a", "b", "end")
        assert state.done

    async def test_a_node_without_an_edge_is_terminal(self) -> None:
        graph = Graph[State](
            nodes={"only": counting("only")},
            edges={},
            entry="only",
            on_budget_exhausted="only",
        )

        _, trace = await graph.run(State(budget=Budget()))

        assert trace.path == ("only",)

    async def test_an_infinite_cycle_is_cut_by_the_step_budget(self) -> None:
        """The reason the budget exists at all."""
        graph = Graph[State](
            nodes={"loop": counting("loop"), "end": terminal},
            edges={"loop": lambda _: "loop"},
            entry="loop",
            on_budget_exhausted="end",
        )

        state, trace = await graph.run(State(budget=Budget(max_steps=5)))

        assert trace.terminated_early
        assert "step budget" in (trace.stopped_because or "")
        assert state.done, "the terminal node must still run, so callers get a real answer"

    async def test_an_expired_deadline_still_produces_a_terminal_state(self) -> None:
        graph = Graph[State](
            nodes={"loop": counting("loop"), "end": terminal},
            edges={"loop": lambda _: "loop"},
            entry="loop",
            on_budget_exhausted="end",
        )

        state, trace = await graph.run(State(budget=Budget(deadline_seconds=0.0)))

        assert state.done
        assert "deadline" in (trace.stopped_because or "")

    async def test_timings_are_recorded_per_node(self) -> None:
        graph = Graph[State](
            nodes={"a": counting("a"), "end": terminal},
            edges={"a": lambda _: "end"},
            entry="a",
            on_budget_exhausted="end",
        )

        _, trace = await graph.run(State(budget=Budget()))

        assert [name for name, _ in trace.timings_ms] == ["a", "end"]

    async def test_routing_to_an_unknown_node_raises(self) -> None:
        """A silent stop would look like a normal answer with missing steps."""
        graph = Graph[State](
            nodes={"a": counting("a"), "end": terminal},
            edges={"a": lambda _: "nowhere"},
            entry="a",
            on_budget_exhausted="end",
        )

        with pytest.raises(GraphError, match="unknown node"):
            await graph.run(State(budget=Budget()))

    def test_a_missing_entry_node_is_rejected_at_construction(self) -> None:
        with pytest.raises(GraphError, match="entry node"):
            Graph[State](nodes={"a": terminal}, edges={}, entry="missing", on_budget_exhausted="a")

    def test_an_edge_for_an_unknown_node_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="unknown node"):
            Graph[State](
                nodes={"a": terminal},
                edges={"ghost": lambda _: "a"},
                entry="a",
                on_budget_exhausted="a",
            )


class TestTerminationProperty:
    """The property the whole design rests on."""

    @settings(max_examples=200, deadline=None)
    @given(
        st.lists(st.booleans(), min_size=0, max_size=40),
        st.integers(min_value=1, max_value=30),
    )
    async def test_the_machine_always_halts(self, verdicts: list[bool], max_steps: int) -> None:
        """Random verdicts, a retry loop, and a bounded budget.

        Whatever sequence of grading outcomes arrives — including one that
        always says "try again" — the machine must stop.
        """

        async def grade(state: State) -> State:
            return replace(state, budget=state.budget.step(), visits=state.visits + 1)

        async def retry(state: State) -> State:
            return replace(state, budget=state.budget.step().retrieved())

        def after_grade(state: State) -> str:
            index = state.visits - 1
            satisfied = verdicts[index] if 0 <= index < len(verdicts) else False
            if satisfied or not state.budget.can_retrieve_again:
                return "end"
            return "retry"

        graph = Graph[State](
            nodes={"grade": grade, "retry": retry, "end": terminal},
            edges={"grade": after_grade, "retry": lambda _: "grade"},
            entry="grade",
            on_budget_exhausted="end",
        )

        state, trace = await graph.run(
            State(budget=Budget(max_steps=max_steps), verdicts=tuple(verdicts))
        )

        assert state.done
        assert len(trace.path) <= max_steps + 1

    @settings(max_examples=100, deadline=None)
    @given(st.integers(min_value=1, max_value=20))
    async def test_the_step_bound_is_never_exceeded(self, max_steps: int) -> None:
        graph = Graph[State](
            nodes={"loop": counting("loop"), "end": terminal},
            edges={"loop": lambda _: "loop"},
            entry="loop",
            on_budget_exhausted="end",
        )

        state, _ = await graph.run(State(budget=Budget(max_steps=max_steps)))

        # The terminal node runs after the bound trips, so one over is expected.
        assert state.budget.steps_used <= max_steps + 1
