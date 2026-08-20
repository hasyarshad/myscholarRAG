"""Reciprocal rank fusion.

Merges rankings from retrieval branches that score on incomparable scales.
Cosine similarity sits in a narrow band; ``ts_rank_cd`` is unbounded and
length-dependent. Normalising and averaging them means choosing a weight that
is wrong for half the queries, because which branch dominates varies per query.

RRF discards the scores and uses only the ranks::

    score(d) = sum over branches  weight_b / (k + rank_b(d))
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

#: From the original paper. Large enough that the first few ranks differ by
#: under 2%, so agreement across branches outweighs confidence within one. With
#: k=0 the top hit of a single branch would dominate the fusion outright.
DEFAULT_K = 60


def reciprocal_rank_fusion[T: Hashable](
    ranked_lists: Mapping[str, Sequence[T]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[tuple[T, float]]:
    """Fuse ranked lists into one ranking.

    ``ranked_lists`` maps a branch name to its results, best first. ``weights``
    scales each branch's contribution; unnamed branches default to 1.0.

    Returns ``(item, score)`` pairs, best first. Ties break on the item's first
    appearance across branches, so the result is deterministic — which matters,
    because a non-deterministic ranking makes the evaluation harness unable to
    distinguish a real change from noise.
    """
    weights = weights or {}
    scores: dict[T, float] = {}
    first_seen: dict[T, int] = {}

    position = 0
    for branch, items in ranked_lists.items():
        weight = weights.get(branch, 1.0)
        if weight == 0.0:
            continue
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            if item not in first_seen:
                first_seen[item] = position
                position += 1

    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], first_seen[pair[0]]))
    return ordered[:limit] if limit is not None else ordered


def weighted_branches[T: Hashable](
    dense: Sequence[T],
    lexical: Sequence[T],
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    k: int = DEFAULT_K,
    limit: int | None = None,
) -> list[tuple[T, float]]:
    """Fuse the two standard branches.

    The defaults weight both branches equally, which was measured rather than
    assumed: an initial 0.4 for lexical scored worse than 1.0 on the retrieval
    test set. A corpus dense in identifiers such as "NSGA-II" or "T-Type VSI"
    rewards exact matching more than general guidance suggests.
    """
    return reciprocal_rank_fusion(
        {"dense": dense, "lexical": lexical},
        weights={"dense": dense_weight, "lexical": lexical_weight},
        k=k,
        limit=limit,
    )
