"""Score the automated metric against the hand labels.

The generation evaluation decides "correct" with regular expressions over the
answer text. That is a proxy, and a proxy is only worth what its agreement with
a considered judgement is worth. This measures that agreement.

Labels are in tests/eval/labels.yaml, judged against tests/eval/rubric.md, which
was written before the answers were read.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


MUST_REFUSE = {"unanswerable", "false_premise", "injection"}


def verdict(kind: str, text: str, *, structural_refusal: bool) -> str:
    """Re-run the *current* text detector over the stored answer.

    Not the verdict recorded when the answers were produced. Comparing labels
    against a frozen verdict measures nothing about a change to the patterns —
    the first version of this script did exactly that, and would have reported
    "no improvement" for a pattern that in fact fixed a case.

    ``structural_refusal`` carries the one signal that cannot be recovered from
    the text: the pipeline returning a Refusal object. The canned out-of-scope
    reply matches no textual pattern, so recomputing from text alone scored
    three correct injection refusals as failures.
    """
    from myscholar.evaluation.baseline import contradicts, declines

    declined = structural_refusal or declines(text)
    if kind in MUST_REFUSE:
        return "correct" if declined or contradicts(text) else "incorrect"
    return "incorrect" if declined else "correct"


def main() -> int:
    records = yaml.safe_load((ROOT / "tests/eval/labels.yaml").read_text())
    from myscholar.evaluation.baseline import declines

    # The run the hand labels in tests/eval/labels.yaml were judged against.
    labelled = ROOT / "results/variance/run-1.json"
    rows = json.loads(labelled.read_text())["adaptive"]["outcomes"]
    # Recorded as declined but matching no textual pattern => it was structural.
    answers = {
        o["question"]: (o["text"], bool(o["declined"]) and not declines(o["text"])) for o in rows
    }

    counts: Counter[str] = Counter()
    disagreements = []
    for r in records:
        text, structural = answers[r["question"]]
        r["metric_said"] = verdict(r["kind"], text, structural_refusal=structural)
        truth, said = r["label"], r["metric_said"]
        counts[f"{truth}/{said}"] += 1
        if truth != said:
            disagreements.append(r)

    tp = counts["correct/correct"]
    fn = counts["correct/incorrect"]  # metric too harsh
    fp = counts["incorrect/correct"]  # metric too lenient
    tn = counts["incorrect/incorrect"]
    total = tp + fn + fp + tn

    print(f"{total} labelled answers\n")
    print("                      metric says")
    print("                 correct   incorrect")
    print(f"  truly correct   {tp:>7}   {fn:>9}")
    print(f"  truly wrong     {fp:>7}   {tn:>9}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n  agreement  {(tp + tn) / total:.1%}")
    print(f"  precision  {precision:.1%}   (of answers the metric passed, how many deserved it)")
    print(
        f"  recall     {recall:.1%}   "
        f"(of answers that deserved to pass, how many the metric passed)"
    )

    print(f"\n  metric too harsh:   {fn}")
    print(f"  metric too lenient: {fp}")
    if fp == 0:
        print("\n  NOTE: every disagreement runs one way. A labelling pass by the")
        print("  author of the system that only ever finds the metric too strict")
        print("  is exactly the pattern to distrust — read the reasons below.")

    print("\ndisagreements:")
    for r in disagreements:
        direction = "TOO HARSH" if r["label"] == "correct" else "TOO LENIENT"
        print(f"\n  [{direction}] ({r['kind']}) {r['question'][:74]}")
        print(f"    {r['why']}")

    by_kind: dict[str, list[int]] = {}
    for r in records:
        n, c_true, c_said = by_kind.get(r["kind"], [0, 0, 0])
        by_kind[r["kind"]] = [
            n + 1,
            c_true + (r["label"] == "correct"),
            c_said + (r["metric_said"] == "correct"),
        ]
    print(f"\n{'kind':<26} {'n':>3} {'hand':>6} {'metric':>8}")
    for kind, (n, t, s) in sorted(by_kind.items()):
        flag = "  <-- gap" if t != s else ""
        print(f"{kind:<26} {n:>3} {t:>6} {s:>8}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
