#!/usr/bin/env bash
# Run the generation evaluation N times with nothing changed.
#
# Generation and the model-backed judges are stochastic, so this measures how far
# each metric moves between identical runs: the noise floor for any A/B comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-3}"
for i in $(seq 1 "$RUNS"); do
    echo "=== run $i/$RUNS  $(date +%H:%M) ==="
    uv run python scripts/eval_generation.py --configs adaptive
    cp results/generation.json "results/variance/run-$i.json"
done

echo
echo "=== spread across $RUNS identical runs ==="
uv run python - "$RUNS" <<'PY'
import json, pathlib, statistics, sys
runs = int(sys.argv[1])
rows = []
for i in range(1, runs + 1):
    d = json.loads(pathlib.Path(f"results/variance/run-{i}.json").read_text())["adaptive"]
    out = d["outcomes"]
    by = {}
    for r in out:
        n, c = by.get(r["kind"], (0, 0))
        by[r["kind"]] = (n + 1, c + bool(r["correct"]))
    rows.append({
        "unsupported": d["unsupported_claims"],
        "correct_refusal": d["correct_refusal"],
        "over_refusal": d["over_refusal"],
        "claims": sum(r["claims_judged"] for r in out),
        "unsup_n": sum(r["claims_unsupported"] for r in out),
        "false_premise": by.get("false_premise", (0, 0))[1],
        "unanswerable": by.get("unanswerable", (0, 0))[1],
    })

def spread(key, fmt="{:.1%}"):
    # Counts are ints and rates floats, so "{:d}" needs the cast.
    vals = [r[key] for r in rows]
    if fmt.endswith("d}"):
        vals = [int(v) for v in vals]
    lo, hi = min(vals), max(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0
    shown = "  ".join(fmt.format(v) for v in vals)
    return f"{key:<16} {shown}   range {fmt.format(hi - lo)}  sd {fmt.format(sd)}"

print(spread("unsupported"))
print(spread("correct_refusal"))
print(spread("over_refusal"))
print(spread("unsup_n", "{:d}"))
print(spread("claims", "{:d}"))
print(spread("false_premise", "{:d}"))
print(spread("unanswerable", "{:d}"))
print()
print("A later change is only meaningful if it moves a metric by more than the")
print("range above. Anything smaller is within run-to-run noise.")
PY
