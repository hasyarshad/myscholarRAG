"""Run every evaluation and assemble one report.

    make eval

When something regresses, read retrieval before generation: a generation drop
with retrieval unchanged is a generation problem; a drop in both starts in retrieval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS = Path("results")

#: name -> (script, what it measures, roughly how long)
STAGES: dict[str, tuple[str, str, str]] = {
    # First, because the generation numbers depend on the judge. An uncalibrated
    # version reported 57% of the adaptive system's claims as unsupported when the
    # sampled failures were all correct and cited.
    "judge": (
        "scripts/eval_judge.py",
        "can the faithfulness judge be trusted",
        "~1 min",
    ),
    "selection": (
        "scripts/eval_selection.py",
        "are the right papers brought into scope",
        "~1 min",
    ),
    "retrieval": ("scripts/eval_retrieval.py", "is the right document found", "~3 min"),
    "routing": ("scripts/eval_routing.py", "is the right strategy chosen", "~1 min"),
    "generation": (
        "scripts/eval_generation.py",
        "is the answer supported, and does it decline what it cannot answer",
        "~30 min",
    ),
}


def _run(script: str) -> tuple[bool, float]:
    started = time.monotonic()
    completed = subprocess.run(  # fixed argv, no shell
        [sys.executable, script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:], file=sys.stderr)
    return completed.returncode == 0, time.monotonic() - started


def _load(name: str) -> Any:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def headline() -> list[str]:
    """The three numbers worth putting in front of a reader."""
    generation = _load("generation")
    if not generation:
        return []

    lines = [
        "## The comparison",
        "",
        "| Configuration | unsupported claims | correct refusal | over-refusal |",
        "|---|---|---|---|",
    ]
    must_refuse = {"unanswerable", "false_premise", "injection"}
    for name in ("naive", "retrieved", "adaptive"):
        if (config := generation.get(name)) is None:
            continue
        # Recomputed from the outcomes so the headline matches the by-kind table.
        refusing = [o for o in config["outcomes"] if o["kind"] in must_refuse]
        answering = [o for o in config["outcomes"] if o["kind"] not in must_refuse]
        correct_refusal = sum(o["correct"] for o in refusing) / max(len(refusing), 1)
        over_refusal = sum(o["declined"] for o in answering) / max(len(answering), 1)
        lines.append(
            f"| {name} | {_pct(config['unsupported_claims'])} "
            f"| {_pct(correct_refusal)} | {_pct(over_refusal)} |"
        )
    counts = {
        name: (
            sum(o["claims_judged"] for o in config["outcomes"]),
            sum(1 for o in config["outcomes"] if not o["declined"]),
        )
        for name, config in generation.items()
    }
    basis = ", ".join(
        f"{name} {claims} claims over {answers} answers"
        for name, (claims, answers) in counts.items()
    )

    lines += [
        "",
        "`naive` reimplements the baseline pipeline this project replaces, with the "
        "same index, embedder and generator as the others, so the differences come "
        "from pipeline design. `retrieved` pairs hybrid retrieval and parent "
        "expansion with the naive one-shot generator, so it isolates the effect of "
        "retrieval quality. `adaptive` is the full system.",
        "",
        f"Basis: {basis}.",
        "",
        "The configurations write different amounts, so the denominators differ. "
        "The rate is per sentence, but the adaptive figure rests on far fewer "
        "judgements, so each one moves it further and small gaps between rows are "
        "not meaningful.",
        "",
        "Part of the gap comes from the prompt: the adaptive system must cite every "
        "factual sentence, which limits what it writes. That is part of the design "
        "under test, and the `retrieved` row separates it from the effect of retrieval.",
        "",
        _attribution(generation),
        "",
    ]
    return lines


def _attribution(generation: dict[str, Any]) -> str:
    """State which stage earned the improvement, in the run's own numbers."""
    rates = {
        name: config.get("unsupported_claims")
        for name, config in generation.items()
        if config.get("unsupported_claims") is not None
    }
    if not {"naive", "retrieved", "adaptive"} <= rates.keys():
        return ""
    retrieval_delta = (rates["naive"] - rates["retrieved"]) * 100
    loop_delta = (rates["retrieved"] - rates["adaptive"]) * 100
    return (
        f"Retrieval alone moved the unsupported-claim rate by "
        f"{retrieval_delta:.1f} points, and the verification loop by a further "
        f"{loop_delta:.1f}. Without the `retrieved` row, the loop's share would be "
        f"credited to retrieval."
    )


def compose(ran: dict[str, tuple[bool, float]]) -> str:
    lines = [
        "# Evaluation",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC by `make eval`.",
        "",
        *headline(),
        "## Stages",
        "",
        "| Stage | measures | status | took |",
        "|---|---|---|---|",
    ]
    for name, (_, measures, _estimate) in STAGES.items():
        ok, seconds = ran.get(name, (False, 0.0))
        status = "ok" if ok else ("skipped" if name not in ran else "**failed**")
        lines.append(f"| [{name}](./{name}.md) | {measures} | {status} | {seconds:.0f}s |")

    if judge := _load("judge"):
        supported = [r for r in judge if r["expected"] is True]
        unsupported = [r for r in judge if r["expected"] is False]
        lines += [
            "",
            "## Judge calibration",
            "",
            "The unsupported-claim rate depends on the faithfulness judge, so the "
            "judge is measured first, on cases where the right answer is not in doubt.",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Recognises a supported claim | "
            f"{sum(bool(r['correct']) for r in supported) / max(len(supported), 1):.0%} |",
            f"| Catches an unsupported claim | "
            f"{sum(bool(r['correct']) for r in unsupported) / max(len(unsupported), 1):.0%} |",
        ]

    if selection := _load("selection"):
        recall = sum(float(r["recall"]) for r in selection) / max(len(selection), 1)
        complete = sum(1 for r in selection if not r["missing"]) / max(len(selection), 1)
        lines += [
            "",
            "## Document selection",
            "",
            "Which papers a cross-paper question is about, decided before any "
            "passage is fetched. The retrieval set below has no synthesis questions "
            "(27 single-paper and 9 methodological), so it cannot measure this.",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Recall of needed documents | {recall:.0%} |",
            f"| Questions with nothing missing | {complete:.0%} |",
        ]

    if retrieval := _load("retrieval"):
        # Recomputed from the per-question ranks so it cannot drift from them.
        lines += [
            "",
            "## Retrieval",
            "",
            "| Configuration | hit@1 | hit@3 | MRR | n |",
            "|---|---|---|---|---|",
        ]
        for config, outcomes in retrieval.items():
            searched = [o for o in outcomes if not o.get("bypassed")]
            if not searched:
                continue
            ranks = [o.get("rank") for o in searched]
            hit1 = sum(1 for r in ranks if r == 1) / len(ranks)
            hit3 = sum(1 for r in ranks if r is not None and r <= 3) / len(ranks)
            mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
            lines.append(f"| {config} | {hit1:.2f} | {hit3:.2f} | {mrr:.3f} | {len(searched)} |")

    lines += [
        "",
        "## Reading these",
        "",
        "Read correct refusal together with over-refusal. A system that declines "
        "everything scores 100% on correct refusal, and one that answers everything "
        "scores 0% on over-refusal; only the pair shows whether it tells the two "
        "kinds of question apart.",
        "",
        "Unsupported claims are counted per sentence, over the answers actually "
        "given. A refusal contains no claims, so declining more cannot improve the rate.",
        "",
        "Full detail for each stage is in the linked file.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated stages to skip. Generation is the slow one.",
    )
    parser.add_argument("--only", default="", help="Comma-separated stages to run, instead of all.")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    RESULTS.mkdir(exist_ok=True)
    ran: dict[str, tuple[bool, float]] = {}

    for name, (script, _measures, estimate) in STAGES.items():
        if name in skip or (only and name not in only):
            print(f"skipping {name}")
            continue
        print(f"running {name} ({estimate}) ...", flush=True)
        ran[name] = _run(script)
        ok, seconds = ran[name]
        print(f"  {'ok' if ok else 'FAILED'} in {seconds:.0f}s", flush=True)

    (RESULTS / "report.md").write_text(compose(ran), "utf-8")
    print(f"\nwritten to {RESULTS / 'report.md'}")

    # A failed stage fails the run, so a partial evaluation is never reported as complete.
    return 0 if all(ok for ok, _ in ran.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
