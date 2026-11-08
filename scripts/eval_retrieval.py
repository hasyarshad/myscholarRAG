"""Measure retrieval quality: dense, hybrid and the routed pipeline.

    uv run python scripts/eval_retrieval.py

Reports hit@k and MRR overall and per question kind. Dense and hybrid involve no
language model, so they are fast and deterministic.
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
from myscholar.assembly import build_router  # noqa: E402
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

    bypassed: bool = False
    """The router answered without retrieval (by SQL, or as out of scope); scored separately."""

    context_precision: float = 0.0
    """Fraction of the returned blocks that come from the expected document."""


@dataclass
class Metrics:
    hit_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    context_precision: float = 0.0
    found: int = 0
    total: int = 0

    bypassed: int = 0

    @classmethod
    def of(cls, outcomes: list[Outcome]) -> Metrics:
        bypassed = sum(1 for o in outcomes if o.bypassed)
        outcomes = [o for o in outcomes if not o.bypassed]
        if not outcomes:
            return cls(bypassed=bypassed)
        return cls(
            hit_at={
                k: sum(1 for o in outcomes if o.rank is not None and o.rank <= k) / len(outcomes)
                for k in KS
            },
            mrr=sum(1.0 / o.rank for o in outcomes if o.rank is not None) / len(outcomes),
            context_precision=sum(o.context_precision for o in outcomes) / len(outcomes),
            found=sum(1 for o in outcomes if o.rank is not None),
            total=len(outcomes),
            bypassed=bypassed,
        )


async def evaluate(mode: RetrievalMode, cases: list[dict[str, Any]]) -> list[Outcome]:
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    slugs = {row.id: row.slug for row in await store.list_documents()}

    # Constructed here rather than through assembly so the dense and hybrid
    # rows differ in exactly one component.
    retriever = HybridRetriever(
        store=store,
        embedder=container.embedder,
        top_k=settings.retrieval.top_k,
        context_blocks=settings.retrieval.context_blocks,
        window=settings.retrieval.window,
        dense_weight=settings.retrieval.dense_weight,
        lexical_weight=settings.retrieval.lexical_weight,
        rrf_k=settings.retrieval.rrf_k,
        max_span_chunks=settings.retrieval.max_span_chunks,
    )

    outcomes: list[Outcome] = []
    for case in cases:
        result = await retriever.retrieve(case["question"], mode=mode)
        # Ranked over expanded blocks, the unit the generator and the routed
        # configuration both see.
        ranked: list[str] = []
        for block in result.blocks:
            slug = slugs.get(block.document_id, "")
            if slug not in ranked:
                ranked.append(slug)
        rank = ranked.index(case["expect"]) + 1 if case["expect"] in ranked else None

        block_slugs = [slugs.get(b.document_id, "") for b in result.blocks]
        precision = (
            sum(1 for s in block_slugs if s == case["expect"]) / len(block_slugs)
            if block_slugs
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
        f"{len(next(iter(results.values())))} questions, document-level ranking over the",
        "expanded context blocks the generator would receive.",
        "",
        "## Overall",
        "",
        "| Configuration | hit@1 | hit@3 | hit@5 | MRR | context precision | found | bypassed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, outcomes in results.items():
        m = Metrics.of(outcomes)
        lines.append(
            f"| {mode} | {m.hit_at[1]:.2f} | {m.hit_at[3]:.2f} | {m.hit_at[5]:.2f} "
            f"| {m.mrr:.3f} | {m.context_precision:.3f} | {m.found}/{m.total} | {m.bypassed} |"
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
        lines.append("None: every question found its document in at least one configuration.")

    lines += ["", "## Questions routing changed", ""]
    if "hybrid" in results and "routed" in results:
        moved = [
            (results["hybrid"][i], results["routed"][i])
            for i in range(len(results["hybrid"]))
            if results["hybrid"][i].rank != results["routed"][i].rank
        ]
        lines += (
            [
                f"- {before.rank or '—'} -> {after.rank or '—'}  ({after.kind}) {after.question}"
                for before, after in moved
            ]
            if moved
            else ["None."]
        )

    return "\n".join(lines) + "\n"


async def evaluate_routed(cases: list[dict[str, Any]]) -> list[Outcome]:
    """The shipped path: classify, plan, select documents, retrieve, fuse.

    Questions the router answers without retrieval are marked as bypassed.
    """
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    documents = await store.list_documents()
    slugs = {row.id: row.slug for row in documents}

    # Assembled through myscholar.assembly so this row measures the shipped pipeline.
    router = build_router(settings, container, store, titles=[d.title for d in documents])

    outcomes: list[Outcome] = []
    for case in cases:
        result = await router.route(case["question"])
        ranked: list[str] = []
        for block in result.blocks:
            slug = slugs.get(block.document_id, "")
            if slug not in ranked:
                ranked.append(slug)
        rank = ranked.index(case["expect"]) + 1 if case["expect"] in ranked else None

        block_slugs = [slugs.get(b.document_id, "") for b in result.blocks]
        precision = (
            sum(1 for s in block_slugs if s == case["expect"]) / len(block_slugs)
            if block_slugs
            else 0.0
        )
        outcomes.append(
            Outcome(
                question=case["question"],
                kind=case.get("kind", "topical"),
                expected=case["expect"],
                rank=rank,
                bypassed=result.answered_without_retrieval,
                context_precision=precision,
            )
        )
    return outcomes


async def main() -> int:
    cases = yaml.safe_load(TEST_SET.read_text("utf-8"))
    print(f"{len(cases)} questions\n")

    results: dict[str, list[Outcome]] = {}
    for mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
        outcomes = await evaluate(mode, cases)
        results[mode.value] = outcomes
        m = Metrics.of(outcomes)
        print(
            f"{mode.value:8s} hit@1={m.hit_at[1]:.2f} hit@3={m.hit_at[3]:.2f} "
            f"hit@5={m.hit_at[5]:.2f} MRR={m.mrr:.3f} ctx-prec={m.context_precision:.3f}"
        )

    outcomes = await evaluate_routed(cases)
    results["routed"] = outcomes
    m = Metrics.of(outcomes)
    print(
        f"{'routed':13s} hit@1={m.hit_at[1]:.2f} hit@3={m.hit_at[3]:.2f} "
        f"hit@5={m.hit_at[5]:.2f} MRR={m.mrr:.3f} ctx-prec={m.context_precision:.3f} "
        f"bypassed={m.bypassed}"
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
