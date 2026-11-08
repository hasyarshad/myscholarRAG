"""Parsing and verifying citation markers.

All checks are deterministic string work, with no model involved:

1. Every marker resolves to a retrieved block. One that does not is a *dangling*
   citation, an invented source.
2. Every factual sentence carries at least one marker.
3. Every verbatim quote appears in the block it cites. Chunk text is stored verbatim,
   apart from its generated header, so quotes can be checked against it.
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

#: A simple sentence split. Over-splitting only means more sentences are checked.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")

#: Shortest sentence that could carry a checkable claim. Below this it is
#: almost certainly a fragment, a heading or a transition.
_MIN_FACTUAL_CHARS = 15

#: A topic sentence announces what the answer covers and asserts no fact of its own,
#: so it needs no citation. Only a leading sentence with no figure or quotation counts.
_TOPIC_SENTENCE = re.compile(
    r"\b(demonstrates?|shows?|has|have|spans?|covers?|includes?|"
    r"brings?|combines?|works?|focus(es)?|contributes?|conducted|studied|"
    r"examin(es|ed)|investigat(es|ed))\b",
    re.IGNORECASE,
)

#: A stated figure or a quotation. A digit counts only at the start of a token, so
#: "99.64%" and "in 2023" are figures, while the digits in "COVID-19" belong to a name.
_STATES_A_FIGURE = re.compile(r"(?<![A-Za-z0-9-])\d|[\"“]")

#: Framing, hedging and refusal sentences, which need no citation.
_NON_FACTUAL = re.compile(
    r"^\s*(i\s+(don'?t|do\s+not|can'?t|cannot|have\s+no)\b"
    # "excerpts" is the prompt's word for the passages, so refusals use it ("The
    # excerpts do not mention ..."). Without it those refusals were deleted as
    # uncited claims, dropping `unanswerable` accuracy from 100% to 67%.
    r"|there\s+is\s+no\b"
    r"|the\s+(provided\s+)?(corpus|papers?|documents?|excerpts?|sources?|"
    r"passages?|materials?|text)\s+(do(es)?\s+not|don'?t)\b"
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


def is_factual(sentence: str, *, leading: bool = False) -> bool:
    """Whether a sentence asserts something that needs a source.

    Everything counts as factual unless it is recognisably framing, hedging or a
    refusal, because a list of claim verbs is never complete. ``leading`` marks the
    first sentence, which may be a topic sentence.
    """
    stripped = _MARKER.sub("", sentence).strip()
    if len(stripped) < _MIN_FACTUAL_CHARS or _NON_FACTUAL.match(stripped):
        return False
    return not (
        leading and _TOPIC_SENTENCE.search(stripped) and not _STATES_A_FIGURE.search(stripped)
    )


#: A specific assertion an excerpt can settle: a figure, a quotation or a named technique.
_CHECKABLE = re.compile(
    r"\d"  # any figure, percentage or year
    r"|[\"\u201c]"  # a quoted span
    r"|\b[A-Z]{2,}(?:-[A-Za-z0-9]+)*\b"  # LSTM, NSGA-II, AFL-1DCNN
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"  # BorderlineSMOTE, GradientBoosting
)


def makes_checkable_claim(sentence: str) -> bool:
    """Whether a sentence asserts something an excerpt can confirm or deny.

    Only these sentences go to the grounding judge. It is reliable on figures,
    quotations and named techniques but rejects reasonable characterisation ("he
    has expertise in X") at a high rate. Every sentence still faces the
    deterministic checks.
    """
    return bool(_CHECKABLE.search(_MARKER.sub("", sentence)))


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

    # 1. Dangling markers: labels that were never in the context.
    used = parse_markers(text)
    for label in used:
        if label not in by_label:
            issues.append(
                CitationIssue(kind="dangling", detail=label, sentence=_sentence_with(text, label))
            )

    # 2. Factual sentences with no marker.
    for position, sentence in enumerate(split_sentences(text)):
        if is_factual(sentence, leading=position == 0) and not parse_markers(sentence):
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
            source_url=by_label[label].source_url,
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


#: An adverbial opener that refers back to a previous sentence, such as "In another
#: work,". Each ends at a comma, so removing it leaves a sentence. Subject phrases such
#: as "These methods" cannot be cut cleanly; the system prompt discourages them instead.
_ORPHANED_OPENER = re.compile(
    r"^(in\s+(another|this|that|the\s+other)\s+(work|study|paper)|"
    r"in\s+(addition|contrast|comparison)|additionally|furthermore|moreover|"
    r"similarly|likewise|by\s+contrast|however)\s*,\s*",
    re.IGNORECASE,
)


def drop_unsupported(text: str, issues: Sequence[CitationIssue]) -> str:
    """Remove the sentences that failed verification and tidy what remains."""
    bad = {_normalise(issue.sentence) for issue in issues if issue.sentence}
    kept = [s for s in split_sentences(text) if _normalise(s) not in bad]
    return rejoin(kept)


def rejoin(kept: list[str]) -> str:
    """Reassemble surviving sentences so they read as prose.

    Used by both sentence-removal paths: citation salvage and the grounding drop in
    the reasoning loop.
    """
    if not kept:
        return ""

    # Only the first survivor: a backward reference later in the answer still
    # has sentences in front of it to refer to.
    first = _ORPHANED_OPENER.sub("", kept[0])
    kept = [first[:1].upper() + first[1:] if first else kept[0], *kept[1:]]

    joined = " ".join(kept).strip()
    # Sentence boundaries survive the split with their punctuation attached, so
    # a source sentence already ending in one produces "values [C4].." here.
    return re.sub(r"([.!?])[.\s]*\1+", r"\1", joined)
