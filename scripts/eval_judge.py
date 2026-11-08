"""Calibrate the faithfulness judge on cases with a known answer.

    uv run python scripts/eval_judge.py

The judge (the utility model) produces the unsupported-claim rate. Recall on
supported and on unsupported claims is reported separately, because a bias either
way is hidden by a single accuracy figure.
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
        "The two rates are reported separately because the errors have different "
        "effects. A judge biased toward *unsupported* makes every configuration look "
        "worse but roughly preserves the comparison; one biased toward *supported* "
        "makes every configuration look clean and hides the differences. A single "
        "accuracy figure cannot show which bias is present.",
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
