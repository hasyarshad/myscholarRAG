"""Parsing and verifying citation markers.

The strongest anti-hallucination mechanism in the pipeline, and the cheapest:
every check here is string work, involves no model, and cannot be talked out of.

A model asked "are your citations valid?" is performing the same kind of
judgement that produced the invalid ones. A set lookup is not.

Three checks:

1. Every emitted marker resolves to a block that was actually retrieved.
   A marker outside the range is a *dangling citation* — an invented source,
   and the most common failure mode.
2. Every factual sentence carries at least one marker. A sentence asserting a
   number or a named entity without one is the model speaking from its own
   knowledge rather than from the context.
3. Any verbatim quote appears in the block it cites. This is why Phase 1 stores
   chunk text verbatim and keeps the generated contextual header separate — a
   quote must be checkable against what the author wrote.

See docs/learning/05-adaptive-rag-and-hallucination.md.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from myscholar.domain.answers import Citation
from myscholar.domain.documents import ContextBlock

#: A citation marker: [C1], [C12], or several at once as [C1][C3] or [C1, C3].
_MARKER = re.compile(r"\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]", re.IGNORECASE)
_LABEL = re.compile(r"C\d+", re.IGNORECASE)

#: Text in double quotes of a length worth checking. Short quoted fragments are
#: usually terminology rather than a claim of verbatim reproduction.
_QUOTE = re.compile(r"[\"“]([^\"”]{25,400})[\"”]")

#: Sentence split. Deliberately simple: an over-eager split produces more
#: sentences to check, which errs toward stricter verification.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")

#: Shortest sentence that could carry a checkable claim. Below this it is
#: almost certainly a fragment, a heading or a transition.
_MIN_FACTUAL_CHARS = 15

#: Sentences that carry no factual load. Requiring citations on these produces
#: false failures on perfectly good answers.
_NON_FACTUAL = re.compile(
    r"^\s*(i\s+(don'?t|do\s+not|can'?t|cannot|have\s+no)\b"
    r"|there\s+is\s+no\b|the\s+(corpus|papers?|documents?)\s+(do(es)?\s+not|don'?t)\b"
    r"|(none|nothing)\s+of\s+the\b"
    r"|in\s+(summary|short)\b|overall\b|however\b|that\s+said\b"
    r"|would\s+you\s+like|let\s+me\s+know|hope\s+(this|that)\s+helps"
    r"|(here|the\s+following)\s+(is|are)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CitationIssue:
    kind: str
    """'dangling', 'uncited', or 'unquoted'."""

    detail: str
    sentence: str = ""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    citations: tuple[Citation, ...] = ()
    issues: tuple[CitationIssue, ...] = field(default_factory=tuple)

    @property
    def dangling(self) -> tuple[CitationIssue, ...]:
        return tuple(i for i in self.issues if i.kind == "dangling")

    @property
    def uncited(self) -> tuple[CitationIssue, ...]:
        return tuple(i for i in self.issues if i.kind == "uncited")

    def summary(self) -> str:
        """A description of what failed, for feeding back into a regeneration."""
        parts = []
        if dangling := self.dangling:
            labels = ", ".join(sorted({i.detail for i in dangling}))
            parts.append(f"These citation labels do not exist: {labels}.")
        if uncited := self.uncited:
            quoted = "; ".join(f'"{i.sentence[:120]}"' for i in uncited[:3])
            parts.append(f"These statements carry no citation: {quoted}")
        if unquoted := [i for i in self.issues if i.kind == "unquoted"]:
            parts.append(f'This quotation is not in the cited source: "{unquoted[0].detail}"')
        return " ".join(parts)


def parse_markers(text: str) -> list[str]:
    """Every citation label in the text, in order of appearance, deduplicated."""
    labels: list[str] = []
    for match in _MARKER.finditer(text):
        for label in _LABEL.findall(match.group(1)):
            upper = label.upper()
            if upper not in labels:
                labels.append(upper)
    return labels


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def is_factual(sentence: str) -> bool:
    """Whether a sentence asserts something that needs a source.

    Everything counts as factual unless it is recognisably framing, hedging or
    a refusal. The direction matters: an earlier version listed the verbs that
    signal a claim, and "It also halves switching losses" — an uncited
    assertion — passed, because "halves" was not on the list. Any such list is
    incomplete, and every omission is a hallucination that slips through.

    Inverting it makes the failure mode a redundant citation on a harmless
    sentence rather than a missing one on a claim.
    """
    stripped = _MARKER.sub("", sentence).strip()
    return len(stripped) >= _MIN_FACTUAL_CHARS and not _NON_FACTUAL.match(stripped)


def strip_markers(text: str) -> str:
    """The answer without its markers, for display or for length checks."""
    text = re.sub(r"\s+", " ", _MARKER.sub("", text))
    # Removing "[C1]" from "a claim [C1]." leaves a space before the full stop.
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify(text: str, blocks: Sequence[ContextBlock]) -> VerificationResult:
    """Check an answer's citations against the blocks it was given.

    Returns the resolved citations and every issue found. Callers decide what to
    do about them; this function only reports.
    """
    by_label = {block.label.upper(): block for block in blocks}
    issues: list[CitationIssue] = []

    # 1. Dangling markers — a label that was never in the context.
    used = parse_markers(text)
    for label in used:
        if label not in by_label:
            issues.append(
                CitationIssue(kind="dangling", detail=label, sentence=_sentence_with(text, label))
            )

    # 2. Factual sentences with no marker.
    for sentence in split_sentences(text):
        if is_factual(sentence) and not parse_markers(sentence):
            issues.append(CitationIssue(kind="uncited", detail="no citation", sentence=sentence))

    # 3. Quotations that do not appear in any cited block.
    normalised = {label: _normalise(block.text) for label, block in by_label.items()}
    for sentence in split_sentences(text):
        cited = [label for label in parse_markers(sentence) if label in normalised]
        for quote in _QUOTE.findall(sentence):
            needle = _normalise(quote)
            searchable = [normalised[label] for label in cited] or list(normalised.values())
            if not any(needle in haystack for haystack in searchable):
                issues.append(CitationIssue(kind="unquoted", detail=quote[:120], sentence=sentence))

    citations = tuple(
        Citation(
            label=label,
            document_id=by_label[label].document_id,
            section_id=by_label[label].section_id,
            document_title=by_label[label].document_title,
            section_heading=by_label[label].section_heading,
            page_start=by_label[label].page_start,
            page_end=by_label[label].page_end,
        )
        for label in used
        if label in by_label
    )

    return VerificationResult(valid=not issues, citations=citations, issues=tuple(issues))


def _sentence_with(text: str, label: str) -> str:
    for sentence in split_sentences(text):
        if label in parse_markers(sentence):
            return sentence
    return ""


def drop_unsupported(text: str, issues: Sequence[CitationIssue]) -> str:
    """Remove the sentences that failed verification.

    The last resort before refusing: if most of an answer is sound and one
    sentence is not, returning the sound part is better than discarding it —
    provided the unsupported claim is genuinely gone rather than softened.
    """
    bad = {_normalise(issue.sentence) for issue in issues if issue.sentence}
    kept = [s for s in split_sentences(text) if _normalise(s) not in bad]
    return " ".join(kept).strip()
