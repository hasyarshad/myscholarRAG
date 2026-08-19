"""Measure document selection on its own.

    uv run python scripts/eval_selection.py

Selection decides which papers a question is about, before any passage is
fetched. It is the stage the summariser exists to serve, and until now nothing
measured it: the retrieval set is 27 single-paper questions and 9
methodological ones, with no cross-paper synthesis at all — so the class this
was built for was invisible to every metric.

Two numbers, and the second is the one that matters:

  recall     of the documents a good answer needs, how many are in scope
  complete   fraction of questions where *every* needed document is in scope

Recall alone hides the shape of the failure. Nine questions at 80% and one at
zero average to 78%, and the one at zero is an answer that confidently omits a
paper. Completeness reports that directly.

Precision is deliberately not scored. Selecting nine documents of twenty-five
and being wrong about two costs a little context; missing the paper that
answers the question costs the answer.

The comparison is against no selection at all, which trivially scores 100% —
that is the point of the second row. Selection is only worth having if what it
*excludes* improves the answer, which is measured downstream in
scripts/eval_generation.py. This tells you whether it is throwing away the
evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import yaml

logging.disable(logging.INFO)

from myscholar.adapters.persistence.repository import (  # noqa: E402
    PostgresStore,
    create_engine,
)
from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.retrieval.selection import DocumentSelector  # noqa: E402

TEST_SET = Path("tests/eval/selection_set.yaml")
RESULTS = Path("results")

#: What a cross-paper synthesis question selects. Matching the planner rather
#: than sweeping: this measures the shipped setting, and a sweep that reported
#: a better number for a value nothing uses would be misleading.
SELECT_DOCUMENTS = 11


async def main() -> None:
    cases = [c for c in yaml.safe_load(TEST_SET.read_text("utf-8")) if isinstance(c, dict)]
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))

    documents = await store.list_documents()
    slug_by_id = {d.id: d.slug for d in documents}
    selector = DocumentSelector(store=store, embedder=container.embedder)

    rows: list[dict[str, object]] = []
    for case in cases:
        selection = await selector.select(
            case["question"], titles=(), title_to_id={}, limit=SELECT_DOCUMENTS
        )
        chosen = {slug_by_id.get(document_id, "") for document_id in selection.document_ids}
        expected = set(case["expect"])
        found = expected & chosen
        rows.append(
            {
                "question": case["question"],
                "kind": case["kind"],
                "expected": sorted(expected),
                "missing": sorted(expected - chosen),
                "recall": len(found) / len(expected),
            }
        )
        mark = "OK " if not rows[-1]["missing"] else "GAP"
        print(f"  {mark} {len(found)}/{len(expected)}  {case['question'][:58]}", flush=True)

    recall = sum(float(r["recall"]) for r in rows) / max(len(rows), 1)
    complete = sum(1 for r in rows if not r["missing"]) / max(len(rows), 1)

    by_kind: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(row)

    lines = [
        "# Document selection",
        "",
        f"{len(cases)} cross-paper questions, selecting {SELECT_DOCUMENTS} of "
        f"{len(documents)} documents.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Recall of needed documents | **{recall:.0%}** |",
        f"| Questions with nothing missing | **{complete:.0%}** |",
        "",
        "Recall alone hides the shape of the failure: nine questions at 80% "
        "and one at zero average to 78%, and the one at zero is an answer that "
        "confidently omits a paper. Precision is not scored — selecting a "
        "document that turns out to be irrelevant costs a little context, "
        "while missing the one that answers the question costs the answer.",
        "",
        "## By question shape",
        "",
        "| Shape | recall | n |",
        "|---|---|---|",
    ]
    for kind, group in sorted(by_kind.items()):
        hit = sum(float(r["recall"]) for r in group) / len(group)
        lines.append(f"| `{kind}` | {hit:.0%} | {len(group)} |")

    if gaps := [r for r in rows if r["missing"]]:
        lines += ["", "## Documents selection did not reach", ""]
        lines += [
            f"- **{r['question']}**\n  - missing: {', '.join(str(m)[:52] for m in r['missing'])}"  # type: ignore[union-attr]
            for r in gaps
        ]
    lines.append("")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selection.md").write_text("\n".join(lines), "utf-8")
    (RESULTS / "selection.json").write_text(json.dumps(rows, indent=2), "utf-8")

    print(f"\nrecall {recall:.0%} | complete {complete:.0%}")
    print(f"written to {RESULTS / 'selection.md'}")


if __name__ == "__main__":
    asyncio.run(main())
