"""Calibrate the faithfulness judge before trusting anything it says.

    uv run python scripts/eval_judge.py

The judge is a 4B model whose verdicts become the headline number of the whole
evaluation. An uncalibrated judge does not produce a soft number — it produces
a confident wrong one, and nothing downstream can tell.

Two rates, reported separately because they fail in opposite directions and an
average hides which:

  recall on supported claims    how often it recognises a claim that is fine
  recall on unsupported claims  how often it catches one that is not

A judge biased toward "unsupported" makes every configuration look terrible and
the *comparison* still roughly holds. A judge biased toward "supported" makes
everything look clean and the comparison collapses. Neither is acceptable, and
they are not visible in a single accuracy figure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import yaml

logging.disable(logging.INFO)

from myscholar.config import load_settings  # noqa: E402
from myscholar.container import build_container  # noqa: E402
from myscholar.evaluation.faithfulness import FaithfulnessJudge  # noqa: E402

TEST_SET = Path("tests/eval/judge_set.yaml")
RESULTS = Path("results")


async def main() -> None:
    cases = [c for c in yaml.safe_load(TEST_SET.read_text()) if isinstance(c, dict)]
    settings = load_settings()
    container = build_container(settings)
    judge = FaithfulnessJudge(container.utility, person=settings.person.name)

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, 1):
        verdict = await judge.judge_one(case["statement"], case["excerpt"])
        rows.append(
            {
                "kind": case["kind"],
                "expected": case["supported"],
                "actual": verdict,
                "correct": verdict is case["supported"],
                "statement": case["statement"],
            }
        )
        mark = "OK " if rows[-1]["correct"] else "BAD"
        print(
            f"  {mark} {index}/{len(cases)} [{case['kind']}] {case['statement'][:58]}", flush=True
        )

    supported = [r for r in rows if r["expected"] is True]
    unsupported = [r for r in rows if r["expected"] is False]
    recall_ok = sum(bool(r["correct"]) for r in supported) / max(len(supported), 1)
    recall_bad = sum(bool(r["correct"]) for r in unsupported) / max(len(unsupported), 1)

    by_kind: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"]), []).append(row)

    lines = [
        "# Faithfulness judge calibration",
        "",
        f"{len(cases)} cases where the right answer is not in doubt.",
        "",
        "| Measure | Value | Basis |",
        "|---|---|---|",
        f"| Recognises a supported claim | **{recall_ok:.0%}** | {len(supported)} cases |",
        f"| Catches an unsupported claim | **{recall_bad:.0%}** | {len(unsupported)} cases |",
        "",
        "Reported separately because they fail in opposite directions. A judge "
        "biased toward *unsupported* makes every configuration look bad while "
        "roughly preserving the comparison; one biased toward *supported* makes "
        "everything look clean and destroys it. A single accuracy figure hides "
        "which is happening.",
        "",
        "## By case shape",
        "",
        "| Shape | correct | n |",
        "|---|---|---|",
    ]
    for kind, group in sorted(by_kind.items()):
        hits = sum(bool(r["correct"]) for r in group)
        lines.append(f"| `{kind}` | {hits}/{len(group)} | {len(group)} |")

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        lines += ["", "## Wrong", ""]
        lines += [
            f"- expected `{r['expected']}`, said `{r['actual']}` ({r['kind']}) — {r['statement']}"
            for r in wrong
        ]
    lines.append("")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "judge.md").write_text("\n".join(lines), "utf-8")
    (RESULTS / "judge.json").write_text(json.dumps(rows, indent=2), "utf-8")

    print(f"\nsupported recall {recall_ok:.0%} | unsupported recall {recall_bad:.0%}")
    print(Counter(r["kind"] for r in wrong))
    print(f"written to {RESULTS / 'judge.md'}")


if __name__ == "__main__":
    asyncio.run(main())
