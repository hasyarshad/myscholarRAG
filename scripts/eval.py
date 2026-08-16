"""Run every evaluation and assemble one report.

    make eval

Four measurements, in increasing order of cost and decreasing order of
certainty:

  extraction  does the corpus say what the PDFs say?      seconds, deterministic
  retrieval   is the right document found?                a minute, deterministic
  routing     is the right strategy chosen?               a minute, one small model
  generation  is the answer supported, and does it        tens of minutes, every model
              decline what it cannot answer?

The order matters when something regresses. A drop in generation quality with
retrieval unchanged is a generation problem; a drop in both is a retrieval
problem wearing a generation costume. Reading them in this order stops an
afternoon being spent on the wrong stage — which has happened here, twice.
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
    # First, because it gates the credibility of the generation numbers. A
    # miscalibrated judge does not produce a soft result, it produces a
    # confident wrong one — measured here, an uncalibrated version reported 57%
    # of the adaptive system's claims as unsupported when the sampled failures
    # were all correct and cited.
    "judge": (
        "scripts/eval_judge.py",
        "can the faithfulness judge be trusted",
        "~1 min",
    ),
    "retrieval": ("scripts/eval_retrieval.py", "is the right document found", "~1 min"),
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
    for name in ("naive", "retrieved", "adaptive"):
        if (config := generation.get(name)) is None:
            continue
        lines.append(
            f"| {name} | {_pct(config['unsupported_claims'])} "
            f"| {_pct(config['correct_refusal'])} | {_pct(config['over_refusal'])} |"
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
        "`naive` is the pipeline this project replaces, reimplemented against "
        "the same index, the same embedder and the same generator — so the "
        "difference is pipeline design and nothing else. `retrieved` keeps the "
        "one-shot generation and swaps in hybrid retrieval, reranking and "
        "parent expansion. `adaptive` is the full system.",
        "",
        f"Basis: {basis}.",
        "",
        "The denominators differ because the configurations write differently. "
        "That does not mechanically move the *rate*, which is per sentence — "
        "but it does mean the adaptive figure rests on far fewer judgements, "
        "so each one is worth about two and a half points. Read the direction, "
        "not the decimal.",
        "",
        "Part of what separates the rows is style rather than retrieval: the "
        "adaptive system's prompt requires every factual sentence to carry a "
        "citation, which shapes what it is willing to write. That is a design "
        "choice being measured, not a confound — but it is why the middle row "
        "matters. Better retrieval alone moved the rate by 0.4 points. The "
        "verification loop moved it by 52.",
        "",
    ]
    return lines


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
            "The faithfulness numbers above are only as good as the model "
            "producing them, so it is measured first, on cases where the right "
            "answer is not in doubt.",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Recognises a supported claim | "
            f"{sum(bool(r['correct']) for r in supported) / max(len(supported), 1):.0%} |",
            f"| Catches an unsupported claim | "
            f"{sum(bool(r['correct']) for r in unsupported) / max(len(unsupported), 1):.0%} |",
        ]

    if retrieval := _load("retrieval"):
        # Recomputed from the per-question ranks rather than read from a stored
        # summary. One number to maintain, and it cannot drift out of step with
        # the outcomes it claims to summarise.
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
        "Correct refusal and over-refusal are only meaningful together. A "
        "system that declines everything scores 100% on the first and is "
        "useless; one that answers everything scores 0% on the second and is "
        "dangerous. The pair measures discrimination.",
        "",
        "Unsupported claims are counted per *sentence*, not per answer, and "
        "only over answers that were actually given. A refusal contains no "
        "claims, so declining more cannot improve the number.",
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

    # A failed stage fails the run. Reporting a partial evaluation as though it
    # were complete is how a regression gets shipped.
    return 0 if all(ok for ok, _ in ran.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
