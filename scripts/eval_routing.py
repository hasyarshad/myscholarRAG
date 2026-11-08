"""Measure query classification accuracy, with the most common error per class.

uv run python scripts/eval_routing.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import yaml

logging.disable(logging.INFO)

from myscholar.adapters.persistence.repository import (  # noqa: E402
    PostgresStore,
    create_engine,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.query.classifier import QueryClassifier  # noqa: E402

#: A question expected as `content` may land in any of these. The model cannot
#: choose out_of_scope, so a general-knowledge question takes a content path and
#: is refused later on the evidence; which content class it gets is arbitrary.
CONTENT_CLASSES = frozenset(
    {"single_paper_factual", "cross_paper_synthesis", "methodological", "biographical"}
)


def _matches(expected: str, actual: str) -> bool:
    if expected == "content":
        return actual in CONTENT_CLASSES
    return actual == expected


TEST_SET = Path("tests/eval/routing_set.yaml")
RESULTS = Path("results")

#: Classes where a misroute causes fabrication or wasted generation, not just a
#: worse search.
CRITICAL = {"bibliographic", "out_of_scope"}


async def main() -> int:
    cases = yaml.safe_load(TEST_SET.read_text("utf-8"))
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    # Must match assembly.py: the prompt interpolates the name and classifications
    # change with it.
    classifier = QueryClassifier(
        container.utility,
        titles=[d.title for d in await store.list_documents()],
        person_name=settings.person.name,
    )

    rows = []
    for case in cases:
        result = await classifier.classify(case["question"])
        rows.append(
            {
                "question": case["question"],
                "expected": case["expect"],
                "actual": result.query_class.value,
                "decided_by": result.decided_by,
                "correct": _matches(case["expect"], result.query_class.value),
                "papers": result.entities.paper_titles,
            }
        )

    correct = sum(1 for r in rows if r["correct"])
    print(f"{len(rows)} questions, {correct} correct ({correct / len(rows):.1%})\n")

    critical = [r for r in rows if r["expected"] in CRITICAL]
    critical_correct = sum(1 for r in rows if r["correct"] and r["expected"] in CRITICAL)
    print(
        f"critical classes (bibliographic, out_of_scope): "
        f"{critical_correct}/{len(critical)} ({critical_correct / len(critical):.1%})"
    )
    print(f"decided by: {dict(Counter(r['decided_by'] for r in rows))}\n")

    by_class: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_class.setdefault(str(row["expected"]), []).append(row)

    print(f"{'expected':24s} {'n':>3s} {'correct':>8s}   most common error")
    for expected, group in sorted(by_class.items()):
        hits = sum(1 for r in group if r["correct"])
        errors = Counter(str(r["actual"]) for r in group if not r["correct"])
        worst = f"{errors.most_common(1)[0][0]} x{errors.most_common(1)[0][1]}" if errors else "—"
        print(f"{expected:24s} {len(group):3d} {hits / len(group):7.0%}   {worst}")

    if wrong := [r for r in rows if not r["correct"]]:
        print("\nmisclassified:")
        for row in wrong:
            print(f"  {row['expected']:22s} -> {row['actual']!s:22s} {row['question'][:52]}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "routing.json").write_text(json.dumps(rows, indent=2), "utf-8")

    lines = [
        "# Query classification",
        "",
        f"{len(rows)} questions. Accuracy **{correct / len(rows):.1%}**, "
        f"critical classes **{critical_correct / len(critical):.1%}**.",
        "",
        "Critical classes are bibliographic and out-of-scope. A misroute there causes",
        "fabrication or wasted generation; in other classes it only worsens the search.",
        "",
        "| Expected class | n | correct | most common error |",
        "|---|---|---|---|",
    ]
    for expected, group in sorted(by_class.items()):
        hits = sum(1 for r in group if r["correct"])
        errors = Counter(str(r["actual"]) for r in group if not r["correct"])
        worst = f"`{errors.most_common(1)[0][0]}` x{errors.most_common(1)[0][1]}" if errors else "—"
        lines.append(f"| `{expected}` | {len(group)} | {hits / len(group):.0%} | {worst} |")

    if wrong:
        lines += ["", "## Misclassified", ""]
        lines += [f"- `{r['expected']}` → `{r['actual']}` — {r['question']}" for r in wrong]
    (RESULTS / "routing.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(f"\nwritten to {RESULTS / 'routing.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
