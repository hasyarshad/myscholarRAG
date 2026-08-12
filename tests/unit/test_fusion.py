"""Reciprocal rank fusion."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion, weighted_branches


class TestReciprocalRankFusion:
    def test_a_single_branch_preserves_its_order(self) -> None:
        result = reciprocal_rank_fusion({"dense": ["a", "b", "c"]})

        assert [item for item, _ in result] == ["a", "b", "c"]

    def test_agreement_between_branches_beats_confidence_within_one(self) -> None:
        """The property that makes fusion worth doing.

        A chunk both branches rank moderately well should outrank one that only
        a single branch loves. Two independent methods agreeing is stronger
        evidence than one method being certain.
        """
        result = reciprocal_rank_fusion(
            {
                "dense": ["loved_by_dense", "x", "agreed", "y"],
                "lexical": ["p", "q", "agreed", "r"],
            }
        )

        ranking = [item for item, _ in result]
        assert ranking.index("agreed") < ranking.index("loved_by_dense")

    def test_scores_follow_the_formula(self) -> None:
        [(item, score)] = reciprocal_rank_fusion({"dense": ["a"]})

        assert item == "a"
        assert score == pytest.approx(1 / (DEFAULT_K + 1))

    def test_weights_scale_a_branch_contribution(self) -> None:
        equal = dict(reciprocal_rank_fusion({"dense": ["a"], "lexical": ["b"]}))
        weighted = dict(
            reciprocal_rank_fusion(
                {"dense": ["a"], "lexical": ["b"]},
                weights={"dense": 1.0, "lexical": 0.4},
            )
        )

        assert equal["a"] == equal["b"]
        assert weighted["a"] > weighted["b"]

    def test_a_zero_weighted_branch_is_ignored(self) -> None:
        """How the ablation turns the lexical branch off without a code path."""
        result = reciprocal_rank_fusion(
            {"dense": ["a"], "lexical": ["b"]}, weights={"lexical": 0.0}
        )

        assert [item for item, _ in result] == ["a"]

    def test_limit_truncates(self) -> None:
        result = reciprocal_rank_fusion({"dense": ["a", "b", "c", "d"]}, limit=2)

        assert len(result) == 2

    def test_empty_input_yields_nothing(self) -> None:
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []

    def test_ties_break_deterministically(self) -> None:
        """A non-deterministic ranking would make the eval unable to tell a
        real change from noise."""
        lists = {"dense": ["a", "b"], "lexical": ["b", "a"]}

        first = [item for item, _ in reciprocal_rank_fusion(lists)]
        second = [item for item, _ in reciprocal_rank_fusion(lists)]

        assert first == second

    def test_larger_k_flattens_the_ranking(self) -> None:
        """k damps the head, so no single branch's favourite dominates."""
        tight = dict(reciprocal_rank_fusion({"dense": ["a", "b"]}, k=1))
        flat = dict(reciprocal_rank_fusion({"dense": ["a", "b"]}, k=1000))

        assert tight["a"] / tight["b"] > flat["a"] / flat["b"]


class TestFusionProperties:
    @settings(max_examples=100, deadline=None)
    @given(
        st.lists(st.integers(0, 40), min_size=0, max_size=25, unique=True),
        st.lists(st.integers(0, 40), min_size=0, max_size=25, unique=True),
    )
    def test_output_is_the_union_of_inputs(self, dense: list[int], lexical: list[int]) -> None:
        result = reciprocal_rank_fusion({"dense": dense, "lexical": lexical})

        assert {item for item, _ in result} == set(dense) | set(lexical)

    @settings(max_examples=100, deadline=None)
    @given(
        st.lists(st.integers(0, 40), min_size=0, max_size=25, unique=True),
        st.lists(st.integers(0, 40), min_size=0, max_size=25, unique=True),
    )
    def test_scores_are_non_increasing(self, dense: list[int], lexical: list[int]) -> None:
        result = reciprocal_rank_fusion({"dense": dense, "lexical": lexical})

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.integers(0, 40), min_size=1, max_size=25, unique=True))
    def test_no_item_appears_twice(self, items: list[int]) -> None:
        result = reciprocal_rank_fusion({"dense": items, "lexical": list(reversed(items))})

        seen = [item for item, _ in result]
        assert len(seen) == len(set(seen))

    @settings(max_examples=60, deadline=None)
    @given(st.lists(st.integers(0, 30), min_size=1, max_size=20, unique=True))
    def test_branch_order_does_not_change_the_result(self, items: list[int]) -> None:
        """Fusion must be commutative across branches, or the result depends on
        dictionary insertion order rather than on the rankings."""
        other = list(reversed(items))

        forward = reciprocal_rank_fusion({"dense": items, "lexical": other})
        backward = reciprocal_rank_fusion({"lexical": other, "dense": items})

        assert dict(forward) == dict(backward)


class TestWeightedBranches:
    def test_defaults_weight_both_branches_equally(self) -> None:
        """Measured, not assumed.

        An initial 0.4 for lexical scored MRR 0.887 on the retrieval test set
        against 0.901 for 1.0. A corpus this dense in identifiers rewards exact
        matching more than general guidance suggests.
        """
        scores = dict(weighted_branches(["a"], ["b"]))

        assert scores["a"] == scores["b"]

    def test_the_lexical_weight_can_still_be_lowered(self) -> None:
        scores = dict(weighted_branches(["a"], ["b"], lexical_weight=0.4))

        assert scores["a"] > scores["b"]

    def test_an_identifier_found_only_by_lexical_still_surfaces(self) -> None:
        """The case the lexical branch exists for.

        Dense search smears rare identifiers such as NSGA-II across their
        subword pieces, so the chunk that actually contains one may appear in
        no dense result at all.
        """
        result = weighted_branches(
            ["unrelated_1", "unrelated_2"], ["chunk_with_nsga_ii", "unrelated_1"]
        )

        assert "chunk_with_nsga_ii" in [item for item, _ in result]
