"""Measure refusal behaviour and citation validity.

    uv run python scripts/eval_adversarial.py

Three numbers, reported together because none means anything alone:

  correct refusal — refuses questions the corpus genuinely cannot answer
  over-refusal    — refuses questions it could have answered
  citation validity — emitted markers that resolve to a real retrieved block

Correct refusal alone is trivially maximised by refusing everything, which is
why over-refusal is measured on a control group of answerable questions and
reported alongside. The pair measures discrimination; either number on its own
measures a bias.

Citation validity should be 1.0 by construction. If it is not, the verifier has
a hole, and that is a more serious finding than any of the rates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

logging.disable(logging.INFO)

from myscholar.adapters.persistence.repository import (  # noqa: E402
    PostgresStore,
    create_engine,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.domain.answers import Refusal  # noqa: E402
from myscholar.generation import citations  # noqa: E402
from myscholar.generation.answerer import Answerer  # noqa: E402
from myscholar.query.classifier import QueryClassifier  # noqa: E402
from myscholar.query.planner import QueryPlanner  # noqa: E402
from myscholar.query.router import QueryRouter  # noqa: E402
from myscholar.query.transforms import QueryTransformer  # noqa: E402
from myscholar.reasoning.engine import Engine  # noqa: E402
from myscholar.reasoning.graph import Budget  # noqa: E402
from myscholar.retrieval.retriever import HybridRetriever  # noqa: E402

TEST_SET = Path("tests/eval/adversarial_set.yaml")
RESULTS = Path("results")


@dataclass
class Outcome:
    question: str
    kind: str
    expected: str
    refused: bool
    text: str
    citations: int
    dangling: int
    path: str

    @property
    def correct(self) -> bool:
        return self.refused == (self.expected == "refuse")


async def build_engine() -> Engine:
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    documents = await store.list_documents()

    router = QueryRouter(
        classifier=QueryClassifier(container.utility, titles=[d.title for d in documents]),
        planner=QueryPlanner(top_k=settings.retrieval.top_k, rerank_k=settings.retrieval.rerank_k),
        transformer=QueryTransformer(container.utility),
        retriever=HybridRetriever(
            store=store,
            embedder=container.embedder,
            reranker=container.reranker,
            top_k=settings.retrieval.top_k,
            rerank_k=settings.retrieval.rerank_k,
            window=settings.retrieval.window,
            dense_weight=settings.retrieval.dense_weight,
            lexical_weight=settings.retrieval.lexical_weight,
            rrf_k=settings.retrieval.rrf_k,
            max_span_chunks=settings.retrieval.max_span_chunks,
        ),
        store=store,
    )
    return Engine(
        router=router,
        answerer=Answerer(generator=container.generator, utility=container.utility),
        budget=Budget(
            max_steps=settings.reasoning.max_steps,
            max_retrieval_rounds=settings.reasoning.max_retrieval_rounds,
            max_regenerations=settings.reasoning.max_regenerations,
            deadline_seconds=settings.reasoning.deadline_seconds,
        ),
    )


async def main() -> int:
    cases = yaml.safe_load(TEST_SET.read_text("utf-8"))
    engine = await build_engine()

    outcomes: list[Outcome] = []
    for index, case in enumerate(cases, 1):
        response, trace = await engine.answer_with_trace(case["question"])
        refused = isinstance(response, Refusal)
        markers = citations.parse_markers(response.text)
        labels = {c.label for c in getattr(response, "citations", ())}
        outcomes.append(
            Outcome(
                question=case["question"],
                kind=case["kind"],
                expected=case["expect"],
                refused=refused,
                text=response.text,
                citations=len(labels),
                dangling=sum(1 for m in markers if m not in labels),
                path=" -> ".join(trace.path),
            )
        )
        mark = "ok " if outcomes[-1].correct else "MISS"
        print(f"  [{index:2d}/{len(cases)}] {mark} {case['kind']:14s} {case['question'][:46]}")

    print()
    _report(outcomes)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "adversarial.json").write_text(
        json.dumps([o.__dict__ for o in outcomes], indent=2), "utf-8"
    )
    (RESULTS / "adversarial.md").write_text(_render(outcomes), "utf-8")
    print(f"\nwritten to {RESULTS / 'adversarial.md'}")
    return 0


def _rates(outcomes: list[Outcome]) -> tuple[float, float, int, int]:
    should_refuse = [o for o in outcomes if o.expected == "refuse"]
    should_answer = [o for o in outcomes if o.expected == "answer"]
    correct_refusal = (
        sum(1 for o in should_refuse if o.refused) / len(should_refuse) if should_refuse else 0.0
    )
    over_refusal = (
        sum(1 for o in should_answer if o.refused) / len(should_answer) if should_answer else 0.0
    )
    return correct_refusal, over_refusal, len(should_refuse), len(should_answer)


def _report(outcomes: list[Outcome]) -> None:
    correct_refusal, over_refusal, n_refuse, n_answer = _rates(outcomes)
    dangling = sum(o.dangling for o in outcomes)
    emitted = sum(o.citations + o.dangling for o in outcomes)
    validity = (emitted - dangling) / emitted if emitted else 1.0

    print(
        f"correct refusal   {correct_refusal:.1%}  ({n_refuse} questions the corpus cannot answer)"
    )
    print(f"over-refusal      {over_refusal:.1%}  ({n_answer} answerable control questions)")
    print(f"citation validity {validity:.1%}  ({dangling} dangling of {emitted} emitted)")
    print()
    for kind in sorted({o.kind for o in outcomes}):
        group = [o for o in outcomes if o.kind == kind]
        hits = sum(1 for o in group if o.correct)
        print(f"  {kind:14s} {hits}/{len(group)}")


def _render(outcomes: list[Outcome]) -> str:
    correct_refusal, over_refusal, n_refuse, n_answer = _rates(outcomes)
    dangling = sum(o.dangling for o in outcomes)
    emitted = sum(o.citations + o.dangling for o in outcomes)
    validity = (emitted - dangling) / emitted if emitted else 1.0

    lines = [
        "# Refusal and citation behaviour",
        "",
        f"{len(outcomes)} questions.",
        "",
        "| Metric | Value | Basis |",
        "|---|---|---|",
        f"| Correct refusal | **{correct_refusal:.1%}** | {n_refuse} unanswerable questions |",
        f"| Over-refusal | **{over_refusal:.1%}** | {n_answer} answerable control questions |",
        f"| Citation validity | **{validity:.1%}** | {dangling} dangling of {emitted} markers |",
        "",
        "Correct refusal is reported with over-refusal because either alone is",
        "trivially maximised — refusing everything scores 100% on the first.",
        "",
        "## By question kind",
        "",
        "| Kind | correct | n |",
        "|---|---|---|",
    ]
    for kind in sorted({o.kind for o in outcomes}):
        group = [o for o in outcomes if o.kind == kind]
        hits = sum(1 for o in group if o.correct)
        lines.append(f"| `{kind}` | {hits / len(group):.0%} | {len(group)} |")

    if misses := [o for o in outcomes if not o.correct]:
        lines += ["", "## Incorrect", ""]
        for o in misses:
            wanted = "refusal" if o.expected == "refuse" else "an answer"
            lines.append(f"- **{o.kind}** — expected {wanted}: {o.question}")
            lines.append(f"  > {o.text[:220]}")

    lines += ["", "## Paths taken", "", "| Path | count |", "|---|---|"]
    for path, count in Counter(o.path for o in outcomes).most_common(8):
        lines.append(f"| `{path}` | {count} |")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
