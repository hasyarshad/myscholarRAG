"""Measure retrieval quality across the three configurations.

    uv run python scripts/eval_retrieval.py

Runs the same test set through dense, hybrid, and hybrid+rerank, and reports
hit@k and MRR. No language model is involved, so this is fast, deterministic,
and cheap enough to run after any retrieval change.

The per-kind breakdown is the interesting part: an average that improves tells
you something worked, but not what. Identifier questions should show the
lexical branch earning its place, and paraphrase questions the reranker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.disable(logging.INFO)

from myscholar.adapters.persistence.repository import (  # noqa: E402
    PostgresStore,
    create_engine,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.retrieval.retriever import HybridRetriever, RetrievalMode  # noqa: E402

TEST_SET = Path("tests/eval/retrieval_set.yaml")
RESULTS = Path("results")
KS = (1, 3, 5)


@dataclass
class Outcome:
    question: str
    kind: str
    expected: str
    rank: int | None
    """1-based rank of the expected document, or None if absent."""

    context_precision: float = 0.0
    """Fraction of the returned chunks that come from the expected document.

    Document-level rank cannot see intra-document reordering, which is most of
    what a reranker does once the right paper is already found. This measures
    what the generator actually receives, so the reranker has somewhere to
    show up.
    """


@dataclass
class Metrics:
    hit_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    context_precision: float = 0.0
    found: int = 0
    total: int = 0

    @classmethod
    def of(cls, outcomes: list[Outcome]) -> Metrics:
        if not outcomes:
            return cls()
        return cls(
            hit_at={
                k: sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / len(outcomes)
                for k in KS
            },
            mrr=sum(1.0 / o.rank for o in outcomes if o.rank is not None) / len(outcomes),
            context_precision=sum(o.context_precision for o in outcomes) / len(outcomes),
            found=sum(1 for o in outcomes if o.rank is not None),
            total=len(outcomes),
        )


async def evaluate(mode: RetrievalMode, cases: list[dict[str, Any]]) -> list[Outcome]:
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    slugs = {row.id: row.slug for row in await store.list_documents()}

    retriever = HybridRetriever(
        store=store,
        embedder=container.embedder,
        reranker=container.reranker if mode is RetrievalMode.FULL else None,
        top_k=settings.retrieval.top_k,
        rerank_k=settings.retrieval.rerank_k,
        window=settings.retrieval.window,
        dense_weight=settings.retrieval.dense_weight,
        lexical_weight=settings.retrieval.lexical_weight,
    )

    outcomes: list[Outcome] = []
    for case in cases:
        result = await retriever.retrieve(case["question"], mode=mode)
        # Ranked by document, not chunk: the question is "did it find the right
        # paper", and one paper contributing several chunks should not count as
        # several results.
        ranked = [slugs.get(document_id, "") for document_id in result.document_ids]
        rank = ranked.index(case["expect"]) + 1 if case["expect"] in ranked else None

        chunk_slugs = [slugs.get(hit.document_id, "") for hit in result.ranked]
        precision = (
            sum(1 for s in chunk_slugs if s == case["expect"]) / len(chunk_slugs)
            if chunk_slugs
            else 0.0
        )

        outcomes.append(
            Outcome(
                question=case["question"],
                kind=case.get("kind", "topical"),
                expected=case["expect"],
                rank=rank,
                context_precision=precision,
            )
        )
    return outcomes


def render(results: dict[str, list[Outcome]]) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(next(iter(results.values())))} questions, document-level ranking.",
        "No language model involved: these numbers are deterministic.",
        "",
        "## Overall",
        "",
        "| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, outcomes in results.items():
        m = Metrics.of(outcomes)
        lines.append(
            f"| {mode} | {m.hit_at[1]:.2f} | {m.hit_at[3]:.2f} | {m.hit_at[5]:.2f} "
            f"| {m.mrr:.3f} | {m.context_precision:.3f} | {m.found}/{m.total} |"
        )

    kinds = sorted({o.kind for o in next(iter(results.values()))})
    lines += ["", "## hit@3 by question kind", "", "| Configuration | " + " | ".join(kinds) + " |"]
    lines.append("|---" * (len(kinds) + 1) + "|")
    for mode, outcomes in results.items():
        cells = []
        for kind in kinds:
            subset = [o for o in outcomes if o.kind == kind]
            cells.append(f"{Metrics.of(subset).hit_at[3]:.2f}" if subset else "—")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    lines += ["", "## Questions no configuration answered", ""]
    modes = list(results)
    missed = [
        results[modes[0]][i].question
        for i in range(len(results[modes[0]]))
        if all(results[mode][i].rank is None for mode in modes)
    ]
    if missed:
        lines += [f"- {question}" for question in missed]
    else:
        lines.append("None — every question found its document in at least one configuration.")

    lines += ["", "## Questions the reranker changed", ""]
    if "hybrid" in results and "full" in results:
        moved = [
            (results["hybrid"][i], results["full"][i])
            for i in range(len(results["hybrid"]))
            if results["hybrid"][i].rank != results["full"][i].rank
        ]
        if moved:
            lines += [
                f"- {before.rank or '—'} → {after.rank or '—'}  ({after.kind}) {after.question}"
                for before, after in moved
            ]
        else:
            lines.append("None — reranking left every document rank unchanged.")

    return "\n".join(lines) + "\n"


async def main() -> int:
    cases = yaml.safe_load(TEST_SET.read_text("utf-8"))
    print(f"{len(cases)} questions\n")

    results: dict[str, list[Outcome]] = {}
    for mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID, RetrievalMode.FULL):
        outcomes = await evaluate(mode, cases)
        results[mode.value] = outcomes
        m = Metrics.of(outcomes)
        print(
            f"{mode.value:8s} hit@1={m.hit_at[1]:.2f} hit@3={m.hit_at[3]:.2f} "
            f"hit@5={m.hit_at[5]:.2f} MRR={m.mrr:.3f} ctx-prec={m.context_precision:.3f}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.md").write_text(render(results), "utf-8")
    (RESULTS / "retrieval.json").write_text(
        json.dumps(
            {
                mode: [
                    {
                        "question": o.question,
                        "kind": o.kind,
                        "expected": o.expected,
                        "rank": o.rank,
                        "context_precision": round(o.context_precision, 4),
                    }
                    for o in outcomes
                ]
                for mode, outcomes in results.items()
            },
            indent=2,
        ),
        "utf-8",
    )
    print(f"\nwritten to {RESULTS / 'retrieval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
