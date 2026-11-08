"""The regex answer metric measured against hand-labelled ground truth.

`scripts/eval_generation.py` scores answers with regular expressions. These tests pin
its precision and recall so a pattern change cannot make it stricter or more lenient.
Ground truth is `tests/eval/labels.yaml`, judged against `tests/eval/rubric.md`.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from myscholar.evaluation.baseline import contradicts, declines

ROOT = pathlib.Path(__file__).resolve().parents[2]
LABELS = ROOT / "tests/eval/labels.yaml"
#: The run the labels were judged against. The thresholds below are calibrated to it:
#: no precision violations, 88.2% recall and four semantic misses. Removing the file
#: skips these tests instead of failing them.
ANSWERS = ROOT / "results/variance/run-1.json"
MUST_REFUSE = {"unanswerable", "false_premise", "injection"}

#: Recall may rise but must not fall.
MIN_RECALL = 0.88

pytestmark = pytest.mark.skipif(
    not (LABELS.exists() and ANSWERS.exists()),
    reason="hand labels or the labelled run are absent",
)


def _scored() -> list[tuple[str, str]]:
    rows = json.loads(ANSWERS.read_text(encoding="utf-8"))["adaptive"]["outcomes"]
    stored = {o["question"]: o for o in rows}
    out = []
    for record in yaml.safe_load(LABELS.read_text(encoding="utf-8")):
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
    """Precision must stay perfect, so every score the metric reports is a lower bound."""
    passed_but_wrong = [t for t, said in _scored() if t == "incorrect" and said == "correct"]
    assert not passed_but_wrong, f"{len(passed_but_wrong)} wrong answer(s) scored correct"


def test_recall_does_not_regress() -> None:
    scored = _scored()
    deserved = [s for t, s in scored if t == "correct"]
    recall = sum(s == "correct" for s in deserved) / len(deserved)
    assert recall >= MIN_RECALL, f"recall fell to {recall:.1%}, below {MIN_RECALL:.0%}"


def test_the_remaining_misses_are_the_known_semantic_ones() -> None:
    # The known misses refute by assertion ("reached 99.64%" refutes "99.9%"), which no
    # regular expression detects without fitting this dataset.
    records = yaml.safe_load(LABELS.read_text(encoding="utf-8"))
    missed = [
        r["question"]
        for r, (truth, said) in zip(records, _scored(), strict=True)
        if truth == "correct" and said == "incorrect"
    ]
    assert len(missed) <= 4, f"more misses than the known semantic ceiling: {missed}"
