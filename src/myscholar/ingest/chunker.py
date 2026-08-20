"""Splitting sections into embeddable chunks.

Two rules do most of the work:

**A chunk belongs to exactly one section.** A chunk spanning the end of Methods
and the start of Results averages two unrelated meanings, so its vector is close
to neither. It also cannot be filtered by section, since it has no single one.

**Chunks are small; context is restored later.** Around 350 tokens keeps a
vector specific. The generator still needs coherent prose, so neighbouring
chunks are merged back into a contiguous span *after* retrieval (Phase 2).
Retrieval precision and generation context are different problems.

The code this replaces embedded one sentence per row. "The proposed approach
achieves a 23% reduction compared to the conventional method" carries almost no
retrievable meaning on its own — no topic, no paper, no comparison target.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from myscholar.domain.enums import ChunkKind

#: Candidate sentence boundary: a terminator, optional closing punctuation,
#: whitespace, then a capital or digit. Python's ``re`` only allows fixed-width
#: lookbehind, so abbreviations are handled by re-joining afterwards rather than
#: by excluding them here.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9])")

#: Fragments ending in these are not sentence ends. Dense in this corpus —
#: "as shown in Fig. 3", "reported by H. Arshad", "see Sec. 4" — and splitting
#: on them shreds sentences into unembeddable fragments.
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

#: LaTeX picked up by formula enrichment. Kept in the stored text as evidence,
#: stripped before embedding: "\frac { 1 } { \tau _ { s } }" is token soup that
#: matches no natural-language question and dilutes the rest of the vector.
_LATEX_BLOCK = re.compile(r"\$\$.*?\$\$|\$[^$\n]{3,}\$", re.DOTALL)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\s*(?:\{[^{}]*\})*")
_FORMULA_PLACEHOLDER = re.compile(r"<!--\s*formula-not-decoded\s*-->")


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokeniser.

    Roughly four characters per token for English prose. Chunk boundaries do not
    need to be exact — being 20 tokens over target costs nothing — and the real
    tokeniser would force this module to depend on the embedder, which would
    make it untestable without model weights.
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
    """Below this a chunk is merged into its neighbour rather than kept alone.

    A 15-token fragment embeds almost meaninglessly and wastes a retrieval slot.
    """

    max_tokens: int = 700
    """Hard ceiling. A single sentence longer than this is split mid-sentence,
    which is ugly but better than one chunk swallowing a whole subsection."""


def chunk_section(
    blocks: Sequence[tuple[ChunkKind, str, int]],
    policy: ChunkingPolicy | None = None,
) -> list[TextChunk]:
    """Split one section's blocks into chunks.

    ``blocks`` is ``(kind, text, page)`` in document order. Tables, captions and
    equations are never split or merged — a half table is worse than no table,
    and their boundaries are already meaningful.
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

    Overlap means a fact stated across a chunk boundary appears whole in at
    least one chunk, rather than being split in both.
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
    """Fold undersized prose chunks into a neighbour.

    Sections often end with a short sentence or two. Left alone those become
    chunks too small to embed usefully; merged, they strengthen a neighbour.
    Non-prose chunks are never merged.
    """
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

    # Ordinals must be contiguous within a section: parent expansion in Phase 2
    # fetches ranges by ordinal, and a gap would silently truncate a span.
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

    Without this, a chunk reading "the proposed method reduces torque ripple by
    18%" is unfindable for "what has he achieved on torque ripple?" — the
    sentence names neither the paper nor the topic. The header is a template
    filled from metadata already on hand, which makes it close to free.
    """
    attribution = title
    if year and venue:
        attribution = f"{title} ({venue}, {year})"
    elif year:
        attribution = f"{title} ({year})"

    section = section_heading.strip() or section_kind.replace("_", " ")
    return f"Paper: {attribution}. Section: {section}."
