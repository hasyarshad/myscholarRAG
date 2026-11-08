"""Reciprocal rank fusion.

Merges rankings whose scores are incomparable: cosine similarity sits in a narrow band,
while ``ts_rank_cd`` is unbounded and length-dependent. RRF uses only the ranks::

    score(d) = sum over branches  weight_b / (k + rank_b(d))
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

#: From the original paper. The first few ranks differ by under 2%, so agreement across
#: branches outweighs confidence within one.
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

    Returns ``(item, score)`` pairs, best first. Ties break on first appearance across
    branches, so the ranking is deterministic and evaluation runs are comparable.
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
    """Fuse the dense and lexical branches.

    Equal weights were measured: 0.4 for lexical scored worse than 1.0 on the retrieval
    test set, as the corpus is dense in identifiers such as "NSGA-II".
    """
    return reciprocal_rank_fusion(
        {"dense": dense, "lexical": lexical},
        weights={"dense": dense_weight, "lexical": lexical_weight},
        k=k,
        limit=limit,
    )
