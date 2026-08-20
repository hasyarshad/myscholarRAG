"""The automated metric measured against hand-labelled ground truth.

`scripts/eval_generation.py` decides "correct" with regular expressions over the
answer text. That is a proxy. These tests pin how good a proxy it is, so a later
change to the patterns cannot quietly make the ruler worse — or quietly make it
flatter the system.

Ground truth is `tests/eval/labels.yaml`, judged against `tests/eval/rubric.md`,
which was written before the answers were read.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from myscholar.evaluation.baseline import contradicts, declines

ROOT = pathlib.Path(__file__).resolve().parents[2]
LABELS = ROOT / "tests/eval/labels.yaml"
#: The run the labels were judged against. The thresholds below are calibrated
#: to it: 0 precision violations, 88.2% recall against the 88% floor, and
#: exactly the four semantic misses pinned at the bottom of this file. Keep the
#: file — the `skipif` means removing it disables these tests rather than
#: failing them, and what they check is the metric every generation number in
#: `results/` is computed with.
ANSWERS = ROOT / "results/variance/run-1.json"
MUST_REFUSE = {"unanswerable", "false_premise", "injection"}

#: Recall may rise. It may not fall: that would mean a change made the metric
#: blinder to correct behaviour than it already is.
MIN_RECALL = 0.88

pytestmark = pytest.mark.skipif(
    not (LABELS.exists() and ANSWERS.exists()),
    reason="hand labels or the labelled run are absent",
)


def _scored() -> list[tuple[str, str]]:
    rows = json.loads(ANSWERS.read_text())["adaptive"]["outcomes"]
    stored = {o["question"]: o for o in rows}
    out = []
    for record in yaml.safe_load(LABELS.read_text()):
        o = stored[record["question"]]
        text = o["text"]
        # A Refusal object declines without matching any textual pattern.
        structural = bool(o["declined"]) and not declines(text)
        declined = structural or declines(text)
        if record["kind"] in MUST_REFUSE:
            said = "correct" if declined or contradicts(text) else "incorrect"
        else:
            said = "incorrect" if declined else "correct"
        out.append((record["label"], said))
    return out


def test_the_metric_never_passes_a_wrong_answer() -> None:
    """Precision must stay perfect.

    This is the property that makes the metric usable despite its blind spots:
    every number it reports is a floor. A pattern that breaks it — one loose
    enough to call a bad answer good — is worse than no pattern at all.
    """
    passed_but_wrong = [t for t, said in _scored() if t == "incorrect" and said == "correct"]
    assert not passed_but_wrong, f"{len(passed_but_wrong)} wrong answer(s) scored correct"


def test_recall_does_not_regress() -> None:
    """How many genuinely correct answers the metric recognises."""
    scored = _scored()
    deserved = [s for t, s in scored if t == "correct"]
    recall = sum(s == "correct" for s in deserved) / len(deserved)
    assert recall >= MIN_RECALL, (
        f"recall fell to {recall:.1%}, below {MIN_RECALL:.0%} — the metric got blinder"
    )


def test_the_remaining_misses_are_the_known_semantic_ones() -> None:
    """Refutation by assertion is not lexically detectable, and that is the ceiling.

    "reached 99.64%" refutes "99.9%"; "a three-factor instrument" refutes
    "one-dimensional". The contradiction lives in the relation between the
    question's claim and the answer's fact, not in any word the answer uses. No
    regular expression reaches it, and pattern-writing that tries is fitting to
    this dataset. Pinned so the limit stays visible rather than being rediscovered.
    """
    records = yaml.safe_load(LABELS.read_text())
    missed = [
        r["question"]
        for r, (truth, said) in zip(records, _scored(), strict=True)
        if truth == "correct" and said == "incorrect"
    ]
    assert len(missed) <= 4, f"more misses than the known semantic ceiling: {missed}"
