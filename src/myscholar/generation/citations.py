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

#: A topic sentence: it announces what the answer will cover and asserts no
#: specific fact of its own.
#:
#: These were being flagged as uncited claims, and the cost was out of all
#: proportion. "X demonstrates strengths in machine learning through ensemble
#: methods, deep learning and feature engineering" states nothing that the
#: cited sentences below it do not state precisely — but the failure spent all
#: three regeneration attempts on a formatting problem, and the drafts got
#: worse each time. A four-sentence answer citing five sources across four
#: papers was reduced to one sentence citing one.
#:
#: Deliberately narrow: only a *leading* sentence, and only one carrying no
#: figure and no quotation, so anything stating a number, a measurement or a
#: quoted span is still a claim that must be cited.
_TOPIC_SENTENCE = re.compile(
    r"\b(demonstrates?|shows?|has|have|spans?|covers?|includes?|"
    r"brings?|combines?|works?|focus(es)?|contributes?|conducted|studied|"
    r"examin(es|ed)|investigat(es|ed))\b",
    re.IGNORECASE,
)

#: A figure being *stated*, as opposed to a digit appearing inside a name.
#:
#: The distinction is the whole point. The rule above once excluded any sentence
#: containing a digit, which is right for "the framework reached 99.64%" and
#: wrong for "he has conducted research related to COVID-19" — a name that
#: happens to contain a number. The consequence was specific and bad: asked
#: "has he done any work related to COVID-19?", the model opened with exactly
#: the sentence a reader wants, the checker read "-19" as an uncited
#: measurement, and the opening was deleted. What survived began mid-thought,
#: "Findings reveal that toxic content is widespread...", answering a question
#: nobody could tell had been asked.
#:
#: So a digit disqualifies only when it starts a token. In "COVID-19" and
#: "Industry-4" the digit is bound to a word; in "99.64%", "1.8 million" and
#: "in 2023" it stands alone, and those are claims.
_STATES_A_FIGURE = re.compile(r"(?<![A-Za-z0-9-])\d|[\"“]")

#: Sentences that carry no factual load. Requiring citations on these produces
#: false failures on perfectly good answers.
_NON_FACTUAL = re.compile(
    r"^\s*(i\s+(don'?t|do\s+not|can'?t|cannot|have\s+no)\b"
    # "excerpts" belongs here because it is the word the *prompt* uses — the
    # generator is told to answer "using only the excerpts provided", so that is
    # what it calls them when it has to say they contain nothing. Without it the
    # checker read "The excerpts do not mention lithium battery degradation" as
    # an uncited factual claim and deleted it, which left an empty answer where
    # a correct refusal had been. Measured over the question set, that cost
    # `unanswerable` 100% -> 67% and `false_premise` 91% -> 64%.
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


def is_factual(sentence: str, *, leading: bool = False) -> bool:
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
    if len(stripped) < _MIN_FACTUAL_CHARS or _NON_FACTUAL.match(stripped):
        return False
    # Only the first sentence may be a topic sentence. Later ones introducing
    # themselves the same way are making claims.
    return not (
        leading and _TOPIC_SENTENCE.search(stripped) and not _STATES_A_FIGURE.search(stripped)
    )


#: A specific, checkable assertion: a number, a measurement, or a named
#: technique. These are what a model fabricates, and what an excerpt can settle.
_CHECKABLE = re.compile(
    r"\d"  # any figure, percentage or year
    r"|[\"\u201c]"  # a quoted span
    r"|\b[A-Z]{2,}(?:-[A-Za-z0-9]+)*\b"  # LSTM, NSGA-II, AFL-1DCNN
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"  # BorderlineSMOTE, GradientBoosting
)


def makes_checkable_claim(sentence: str) -> bool:
    """Whether a sentence asserts something an excerpt can confirm or deny.

    The groundedness check exists to catch *fabrication* — a number that is not
    in the paper, a method it never used. It is poor at judging characterisation
    ("he has expertise in X", "his work emphasises Y"), because no passage
    states those in the abstract, and a model asked to rule on them rejects
    reasonable synthesis at a high rate.

    That mattered because the judgement was destructive. Two model-backed stages
    in series, each rejecting a fraction of good sentences, compose into an
    answer that can only contain single-source single-fact statements — which is
    the wrong shape for any question about the range of someone's work. Measured
    on one such question, grounding rejected half of a correct draft, including
    the sentence describing the very paper it cited.

    So the check now applies where it is reliable: sentences carrying a figure,
    a quotation or a named technique. Everything else is still held to the
    deterministic checks — it must carry a marker, and that marker must resolve
    to a passage that was actually retrieved and graded relevant. A
    characterisation of cited material is a much smaller risk than an invented
    number, and the invented number is still caught.
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

    # 1. Dangling markers — a label that was never in the context.
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


#: An opening phrase that points at a sentence which is no longer there.
#:
#: Removing a sentence from the middle of an answer leaves the ones after it
#: referring backwards to nothing. The visible result was an answer that began
#: "In another work, MOPSO combined with the Vikor decision-making method was
#: applied..." — another work than *what*? The sentence naming the first one had
#: been dropped, and the reader is left holding a comparison with one side
#: missing.
#:
#: Only phrases that can be lifted off cleanly are listed: each is adverbial and
#: ends at a comma, so the sentence behind it is still a sentence. "These
#: methods" and "The study also" are not here — they are the subject, and
#: cutting them leaves a fragment. The prompt is what prevents those; this is
#: the repair for what still gets through.
_ORPHANED_OPENER = re.compile(
    r"^(in\s+(another|this|that|the\s+other)\s+(work|study|paper)|"
    r"in\s+(addition|contrast|comparison)|additionally|furthermore|moreover|"
    r"similarly|likewise|by\s+contrast|however)\s*,\s*",
    re.IGNORECASE,
)


def drop_unsupported(text: str, issues: Sequence[CitationIssue]) -> str:
    """Remove the sentences that failed verification.

    The last resort before refusing: if most of an answer is sound and one
    sentence is not, returning the sound part is better than discarding it —
    provided the unsupported claim is genuinely gone rather than softened.

    What survives is then tidied, because the removal is invisible to the reader
    and its seams are not. An answer is judged as prose, and one that opens
    mid-comparison or ends on a doubled full stop reads as broken regardless of
    how sound the claims inside it are.
    """
    bad = {_normalise(issue.sentence) for issue in issues if issue.sentence}
    kept = [s for s in split_sentences(text) if _normalise(s) not in bad]
    return rejoin(kept)


def rejoin(kept: list[str]) -> str:
    """Reassemble surviving sentences so they read as prose.

    Shared by both paths that remove sentences — citation salvage here, and the
    grounding drop in the reasoning loop — because the seams are identical and
    fixing them in one place left the other producing "In another work, MOPSO
    was applied..." as an opening sentence, with the work it was another of
    already deleted. A reader cannot tell a sentence was removed, only that the
    answer does not make sense.
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
