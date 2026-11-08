"""Splitting sections into embeddable chunks.

A chunk belongs to exactly one section, so its vector carries one meaning and it can be
filtered by section. Chunks are small (around 350 tokens) to keep vectors specific;
neighbouring chunks are merged back into a contiguous span after retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from myscholar.domain.enums import ChunkKind

#: Candidate sentence boundary: a terminator, optional closing punctuation,
#: whitespace, then a capital or digit. ``re`` only allows fixed-width lookbehind,
#: so abbreviations are re-joined afterwards.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9])")

#: Fragments ending in these are not sentence ends ("Fig. 3", "see Sec. 4").
_ABBREVIATIONS = frozenset(
    {
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "sec",
        "secs",
        "tab",
        "tabs",
        "no",
        "nos",
        "vol",
        "vols",
        "pp",
        "ch",
        "al",
        "vs",
        "approx",
        "cf",
        "dr",
        "prof",
        "mr",
        "ms",
        "st",
        "e.g",
        "i.e",
        "etc",
        "inc",
        "ltd",
    }
)

#: A single initial, as in "H. Arshad" or "J. K. Rowling".
_INITIAL = re.compile(r"\b[A-Z]\.$")

#: LaTeX from formula enrichment. Kept in the stored text but stripped before
#: embedding, where it matches no natural-language question and dilutes the vector.
_LATEX_BLOCK = re.compile(r"\$\$.*?\$\$|\$[^$\n]{3,}\$", re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\s*(?:\{[^{}]*\})*")
_FORMULA_PLACEHOLDER = re.compile(r"<!--\s*formula-not-decoded\s*-->")


def estimate_tokens(text: str) -> int:
    """Approximate token count at four characters per token.

    Chunk boundaries need not be exact, and a real tokeniser would tie this module to the
    embedder's weights.
    """
    return max(1, len(text) // 4)


def _ends_with_abbreviation(fragment: str) -> bool:
    if _INITIAL.search(fragment):
        return True
    last = fragment.rstrip().rsplit(maxsplit=1)
    if not last:
        return False
    word = last[-1].rstrip(".").lower()
    return word in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, tolerating scientific abbreviations.

    Splits on every candidate boundary, then re-joins where the preceding
    fragment ended in an abbreviation or an initial.
    """
    parts = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []

    sentences: list[str] = []
    for part in parts:
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part.strip()}"
        else:
            sentences.append(part.strip())
    return [s for s in sentences if s]


def strip_mathematics(text: str) -> str:
    """Remove LaTeX and formula placeholders, for embedding only."""
    text = _FORMULA_PLACEHOLDER.sub(" ", text)
    text = _LATEX_BLOCK.sub(" ", text)
    text = _LATEX_COMMAND.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk, before it acquires database identity."""

    text: str
    kind: ChunkKind
    ordinal: int
    page: int
    token_count: int


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    target_tokens: int = 350
    overlap_tokens: int = 60
    min_tokens: int = 40
    """Below this a prose chunk is merged into the one before it."""

    max_tokens: int = 700
    """Hard ceiling. A single sentence longer than this is split mid-sentence."""


def chunk_section(
    blocks: Sequence[tuple[ChunkKind, str, int]],
    policy: ChunkingPolicy | None = None,
) -> list[TextChunk]:
    """Split one section's blocks into chunks.

    ``blocks`` is ``(kind, text, page)`` in document order. Tables, captions and
    equations are never split or merged.
    """
    policy = policy or ChunkingPolicy()
    chunks: list[TextChunk] = []

    for kind, text, page in blocks:
        if not text.strip():
            continue
        if kind is not ChunkKind.PROSE:
            chunks.append(
                TextChunk(
                    text=text.strip(),
                    kind=kind,
                    ordinal=len(chunks),
                    page=page,
                    token_count=estimate_tokens(text),
                )
            )
            continue
        for body in _pack_sentences(text, policy):
            chunks.append(
                TextChunk(
                    text=body,
                    kind=ChunkKind.PROSE,
                    ordinal=len(chunks),
                    page=page,
                    token_count=estimate_tokens(body),
                )
            )

    return _merge_runts(chunks, policy)


def _pack_sentences(text: str, policy: ChunkingPolicy) -> Iterator[str]:
    """Greedily fill chunks to the target size, with a sentence overlap."""
    sentences = split_sentences(text)
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = estimate_tokens(sentence)

        if tokens > policy.max_tokens:
            if current:
                yield " ".join(current)
                current, current_tokens = [], 0
            yield from _split_oversized(sentence, policy)
            continue

        if current and current_tokens + tokens > policy.target_tokens:
            yield " ".join(current)
            current, current_tokens = _overlap_tail(current, policy)

        current.append(sentence)
        current_tokens += tokens

    if current:
        yield " ".join(current)


def _overlap_tail(sentences: list[str], policy: ChunkingPolicy) -> tuple[list[str], int]:
    """Carry the last few sentences into the next chunk.

    A fact stated across a chunk boundary then appears whole in at least one chunk.
    """
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        tokens = estimate_tokens(sentence)
        if total + tokens > policy.overlap_tokens:
            break
        tail.insert(0, sentence)
        total += tokens
    return tail, total


def _split_oversized(sentence: str, policy: ChunkingPolicy) -> Iterator[str]:
    """Break a sentence that alone exceeds the ceiling, on word boundaries."""
    words = sentence.split()
    budget = policy.target_tokens * 4  # characters
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > budget and current:
            yield " ".join(current)
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        yield " ".join(current)


def _merge_runts(chunks: list[TextChunk], policy: ChunkingPolicy) -> list[TextChunk]:
    """Fold undersized prose chunks into the preceding prose chunk."""
    if not chunks:
        return []

    merged: list[TextChunk] = []
    for chunk in chunks:
        too_small = chunk.kind is ChunkKind.PROSE and chunk.token_count < policy.min_tokens
        can_merge = (
            merged
            and merged[-1].kind is ChunkKind.PROSE
            and merged[-1].token_count + chunk.token_count <= policy.max_tokens
        )
        if too_small and can_merge:
            previous = merged.pop()
            text = f"{previous.text} {chunk.text}"
            merged.append(
                TextChunk(
                    text=text,
                    kind=ChunkKind.PROSE,
                    ordinal=previous.ordinal,
                    page=previous.page,
                    token_count=estimate_tokens(text),
                )
            )
        else:
            merged.append(chunk)

    # Ordinals must be contiguous: parent expansion fetches ranges by ordinal.
    return [
        TextChunk(text=c.text, kind=c.kind, ordinal=i, page=c.page, token_count=c.token_count)
        for i, c in enumerate(merged)
    ]


def build_context_header(
    *,
    title: str,
    year: int | None,
    venue: str | None,
    section_heading: str,
    section_kind: str,
) -> str:
    """The text prepended to a chunk before embedding.

    A chunk such as "the proposed method reduces torque ripple by 18%" names neither its
    paper nor its section; the header supplies both.
    """
    attribution = title
    if year and venue:
        attribution = f"{title} ({venue}, {year})"
    elif year:
        attribution = f"{title} ({year})"

    section = section_heading.strip() or section_kind.replace("_", " ")
    return f"Paper: {attribution}. Section: {section}."
