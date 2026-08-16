"""Measure query classification accuracy.

    uv run python scripts/eval_routing.py

Routing is only worth doing if the classifier is right. A confidently wrong
route is worse than no routing at all — it sends a whole category of question
down a pipeline built for something else, so the failure is systematic.

The confusion breakdown matters more than the headline: which classes get
mistaken for which tells you whether the prompt needs a better description or
the taxonomy needs a different boundary.
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

#: Content classes. A question expected to be `content` may be classified as
#: any of them.
#:
#: Needed because out_of_scope was removed from the classifier's schema. A
#: general-knowledge question now takes the content path deliberately, is
#: searched, and is refused by the grader on the evidence — so which content
#: class it lands in carries no information and pinning one would test the
#: model's mood.
CONTENT_CLASSES = frozenset(
    {"single_paper_factual", "cross_paper_synthesis", "methodological", "biographical"}
)


def _matches(expected: str, actual: str) -> bool:
    if expected == "content":
        return actual in CONTENT_CLASSES
    return actual == expected


TEST_SET = Path("tests/eval/routing_set.yaml")
RESULTS = Path("results")

#: Classes where a misroute causes confident fabrication or wasted generation,
#: rather than merely a suboptimal search. Reported separately because their
#: cost is different in kind, not just degree.
CRITICAL = {"bibliographic", "out_of_scope"}


async def main() -> int:
    cases = yaml.safe_load(TEST_SET.read_text("utf-8"))
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    classifier = QueryClassifier(
        container.utility, titles=[d.title for d in await store.list_documents()]
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
        "Critical classes are bibliographic and out-of-scope: a misroute there causes",
        "confident fabrication or wasted generation, rather than merely a worse search.",
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
