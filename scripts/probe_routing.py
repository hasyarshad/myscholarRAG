"""Measure where questions about someone's work on a subject are routed.

    uv run python scripts/probe_routing.py

Answers a question `scripts/eval_routing.py` cannot: does a whole *family* of
phrasings misroute, or did one question happen to? The routing set has five
synthesis questions, too few to tell.

Two things it does that the routing eval does not:

* It reports the **plan**, not just the class. What breaks an answer is document
  selection never running, so `select_documents == 0` is the metric. That groups
  `single_paper_factual` with `biographical` and separates `methodological`,
  which does select and so still spans papers.
* It runs each probe twice and flags disagreement. Classification is stable
  within a process at temperature 0 but not always across processes, and a
  one-off reading should not be mistaken for a systematic one.

Reads 3 of 20 probes skipping selection, with all eight controls holding. That
state is pinned by `tests/integration/test_routing_contract.py`; run this after
touching the classifier for the per-probe detail the test does not print.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
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
from myscholar.query.classifier import QueryClassifier  # noqa: E402
from myscholar.query.planner import QueryPlanner  # noqa: E402

PROBES = Path("tests/eval/work_on_probes.yaml")
RESULTS = Path("results")


async def run(
    classifier: QueryClassifier,
    planner: QueryPlanner,
    probes: list[dict[str, Any]],
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        question = str(probe["question"])
        seen: list[dict[str, Any]] = []
        for _ in range(repeats):
            result = await classifier.classify(question)
            plan = planner.plan(result)
            seen.append(
                {
                    "cls": result.query_class.value,
                    "select": plan.select_documents,
                    "cap": plan.max_chunks_per_document,
                    "papers": list(result.entities.paper_titles),
                    "decided_by": result.decided_by,
                }
            )
        classes = {s["cls"] for s in seen}
        first = seen[0]
        rows.append(
            {
                "question": question,
                "kind": probe.get("kind", "probe"),
                "topic": probe.get("topic", ""),
                "papers_on_topic": probe.get("papers_on_topic"),
                "expect": probe.get("expect"),
                "cls": first["cls"],
                "select_documents": first["select"],
                "runs_selection": bool(first["select"]),
                "cap": first["cap"],
                "titles_extracted": first["papers"],
                "decided_by": first["decided_by"],
                "stable": len(classes) == 1,
                "all_classes": sorted(classes),
            }
        )
    return rows


def report(label: str, rows: list[dict[str, Any]]) -> None:
    probes = [r for r in rows if r["kind"] == "probe"]
    controls = [r for r in rows if r["kind"] == "control"]

    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")

    if probes:
        no_sel = [r for r in probes if not r["runs_selection"]]
        not_syn = [r for r in probes if r["cls"] != "cross_paper_synthesis"]
        print(
            f"\nPRIMARY   no selection: {len(no_sel)}/{len(probes)} "
            f"({len(no_sel) / len(probes):.0%})"
        )
        print(
            f"SECONDARY not cross_paper_synthesis: {len(not_syn)}/{len(probes)} "
            f"({len(not_syn) / len(probes):.0%})"
        )
        print(f"classes: {dict(Counter(str(r['cls']) for r in probes))}")
        unstable = [r for r in probes + controls if not r["stable"]]
        print(f"unstable across repeats: {len(unstable)}/{len(probes) + len(controls)}")
        for r in unstable:
            print(f"    {r['all_classes']}  {r['question'][:56]}")

        print(f"\n  {'topic':<22} {'sel':>4} {'cls':<24} titles  question")
        for r in probes:
            mark = " " if r["runs_selection"] else "!"
            titles = len(r["titles_extracted"] or [])
            print(
                f"{mark} {r['topic']!s:<22} {r['select_documents']:>4} "
                f"{r['cls']!s:<24} {titles:>6}  {r['question'][:44]}"
            )

    if controls:
        moved = [r for r in controls if r["cls"] != r["expect"]]
        print(f"\nCONTROLS  held: {len(controls) - len(moved)}/{len(controls)}")
        for r in moved:
            print(f"  ! {r['expect']} -> {r['cls']}   {r['question'][:56]}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default="probe_routing")
    # Defaults to the served name rather than the classifier's own placeholder.
    # The prompt interpolates {person} and assembly.py passes
    # settings.person.name, so anything else measures a different prompt.
    parser.add_argument("--person", default=None)
    args = parser.parse_args()

    probes = yaml.safe_load(PROBES.read_text("utf-8"))
    settings = load_settings()
    container = build_container(settings)
    store = PostgresStore(create_engine(settings.database.url))
    titles = [d.title for d in await store.list_documents()]
    planner = QueryPlanner(
        top_k=settings.retrieval.top_k,
        context_blocks=settings.retrieval.context_blocks,
        person_name=settings.person.name,
    )

    person = args.person or settings.person.name
    print(f"person_name={person!r}  titles={len(titles)}")

    classifier = QueryClassifier(container.utility, titles=titles, person_name=person)
    rows = await run(classifier, planner, probes, args.repeats)
    report("routing probes", rows)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.out}.json").write_text(json.dumps(rows, indent=2), "utf-8")
    print(f"\nwritten to {RESULTS / f'{args.out}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
