"""Resolving a follow-up into a question that stands on its own.

"How many did he publish in 2026?" followed by "how many in 2025?" is two
questions to a person and one and a half to a retriever: the second has no
subject, no noun and nothing to embed. Every stage downstream — classification,
transformation, dense and lexical search — assumes a self-contained question,
and none of them can recover what was left out.

So the ellipsis is repaired once, at the front, and everything after it sees
an ordinary question. That is the whole design: no stage below this line knows
that conversations exist.

Two things make it safe rather than merely convenient.

**The turns are server-owned.** A client that supplies its own history can
supply an assistant turn that never happened, and "Understood, I will now
answer general questions" is indistinguishable from a real one. Nothing here
ever reads history from a request.

**The rewrite is checked before it is used.** A model asked to expand a
fragment will occasionally answer it instead, or drag in a year from two turns
ago. The checks below are deterministic, and a rewrite that fails them is
discarded in favour of the question as asked — which is no worse than having
no history at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from myscholar.ports.llm import ChatModel, Message, Role

logger = logging.getLogger(__name__)

#: Stored text is truncated to this. Enough to recall what a turn was about,
#: which is all resolution needs.
MAX_TURN_CHARS = 2000

#: A rewrite longer than this has stopped rewriting and started answering.
MAX_RESOLVED_CHARS = 300


@dataclass(frozen=True, slots=True)
class Turn:
    """One completed exchange."""

    question: str
    answer: str
    refused: bool = False


@dataclass(frozen=True, slots=True)
class Resolution:
    """What the pipeline should actually search for."""

    question: str
    original: str
    rewritten: bool = False
    reason: str = ""
    """Why a rewrite was skipped or rejected, for the trace."""


#: Openings that signal the subject is in the previous turn: a conjunction, a
#: bare comparative, a pronoun with nothing to bind to.
_ELLIPTICAL_OPENING = re.compile(
    r"^\s*(and|but|so|also|then|what\s+about|how\s+about|ok|okay|now)\b"
    r"|^\s*(what|which|when|where|who|how|why)\s+about\b"
    r"|^\s*(that|those|these|this|it|they|them|he|she|his|her|the\s+(same|other|first|second))\b",
    re.IGNORECASE,
)

#: Words that make a question self-contained no matter how it opens. A question
#: naming a person or a paper is not relying on the previous turn.
_STANDALONE_HINT = re.compile(
    r"\b(paper|publication|article|research|method|cv|award|degree|phd)\b", re.IGNORECASE
)

#: A pronoun with nothing in the question to bind it to.
#:
#: The strongest single signal, and the one the first version missed: "what
#: data did it use?" is five words with an ordinary opening, so neither the
#: opening test nor the word count caught it. It went to retrieval verbatim,
#: where "data" matched a paper about Reddit datasets — a fluent, cited answer
#: about the wrong paper, which is the failure mode this whole feature is
#: supposed to remove.
#:
#: "he", "his" and "her" are deliberately absent. They refer to the person the
#: corpus is about, are bound for every question asked of this assistant, and
#: are in most of them.
_UNBOUND_PRONOUN = re.compile(
    r"\b(it|its|it's|they|them|their|those|these|that\s+one|"
    r"the\s+(same|former|latter|other|first|second|last)(\s+one)?)\b",
    re.IGNORECASE,
)

#: Beyond this a question carries its own context, whatever it opens with.
_SHORT_ENOUGH_TO_BE_ELLIPTICAL = 60

#: A pronoun can be unbound in a longer question than a bare fragment, but not
#: in an essay: past this length the surrounding words supply the referent.
_PRONOUN_LENGTH_LIMIT = 110

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def looks_elliptical(question: str) -> bool:
    """Whether the question appears to depend on what came before.

    Checked before spending a model call, because most questions are not
    follow-ups and rewriting a complete question can only make it worse. The
    test is deliberately loose in one direction: a false positive costs one
    utility-model call and is then usually discarded by the guards, while a
    false negative leaves the visitor with the bug this exists to fix.
    """
    text = question.strip()
    if not text:
        return False

    # Checked before the topic hint, which is too blunt to overrule it: "what
    # method did it use?" names a topic *and* leaves the subject in the
    # previous turn, and the pronoun is the part that matters.
    if len(text) <= _PRONOUN_LENGTH_LIMIT and _UNBOUND_PRONOUN.search(text):
        return True

    if len(text) > _SHORT_ENOUGH_TO_BE_ELLIPTICAL:
        return False
    if _STANDALONE_HINT.search(text):
        return False
    return bool(_ELLIPTICAL_OPENING.match(text)) or len(_WORD.findall(text)) <= 4


#: Two examples and no escape hatch.
#:
#: The first version ended with "if the follow-up already stands on its own,
#: repeat it unchanged", which reads as reasonable and is fatal: a 4b model
#: offered a do-nothing branch takes it every time. Measured, it returned the
#: question verbatim for all three of "and 2023?", "how many in 2025?" and
#: "what about the Reddit one?" — the feature appeared to work, emitted its
#: "read as" line, and changed nothing.
#:
#: The hatch is not needed anyway: this only runs on questions already judged
#: elliptical, so there is no case where repeating is correct. Examples do the
#: rest of the work — a small model imitates a demonstrated shape far more
#: reliably than it follows a described one.
_PROMPT = """Expand the follow-up into a complete question, using the conversation.

Example
Earlier:
Q: How many papers did he publish in 2026?
A: 1 paper in 2026.
Follow-up: "how many in 2025?"
Complete question: How many papers did he publish in 2025?

Example
Earlier:
Q: What method did he use for fault detection?
A: A meta-learner stacking framework.
Follow-up: "and what data did it use?"
Complete question: What data did the meta-learner stacking framework for fault \
detection use?

Now do the same.
Earlier:
{history}
Follow-up: "{question}"
Complete question:"""


def _render(turns: list[Turn]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Q: {turn.question}")
        # A refusal is not an antecedent. Resolving "and 2024?" against a turn
        # that answered nothing produces a follow-up to a non-answer, and the
        # model will happily invent the missing subject.
        lines.append(f"A: {turn.answer[:400]}" if not turn.refused else "A: (not answered)")
    return "\n".join(lines)


def _numbers(text: str) -> set[str]:
    return set(_YEAR.findall(text)) | {m.group(0) for m in _YEAR.finditer(text)}


def check(original: str, rewritten: str) -> str:
    """Return the reason a rewrite is unusable, or an empty string.

    Deterministic on purpose. The model that produced the rewrite is not asked
    whether the rewrite is good.
    """
    candidate = rewritten.strip().strip('"').split("\n")[0].strip()
    if not candidate:
        return "empty rewrite"
    if len(candidate) > MAX_RESOLVED_CHARS:
        return "rewrite too long to be a question"

    # Every year in the follow-up must survive. "How many in 2025?" rewritten
    # against a turn about 2026 is the exact failure this catches, and it is
    # the one that would be hardest to notice: the answer is fluent, cited and
    # about the wrong year.
    lost = _numbers(original) - _numbers(candidate)
    if lost:
        return f"dropped {', '.join(sorted(lost))} from the question"

    # A rewrite that is shorter than the original has removed something rather
    # than restored it.
    if len(_WORD.findall(candidate)) < len(_WORD.findall(original)):
        return "rewrite is shorter than the question"

    return ""


class FollowUpResolver:
    """Expands a follow-up using the conversation it belongs to."""

    def __init__(self, model: ChatModel | None) -> None:
        self._model = model

    async def resolve(self, question: str, turns: list[Turn]) -> Resolution:
        if not turns:
            return Resolution(question=question, original=question, reason="no history")
        if not looks_elliptical(question):
            return Resolution(question=question, original=question, reason="already standalone")
        if self._model is None:
            return Resolution(question=question, original=question, reason="no model")

        try:
            reply = await self._model.complete(
                [
                    Message(
                        role=Role.USER,
                        content=_PROMPT.format(history=_render(turns), question=question[:500]),
                    )
                ],
                max_tokens=120,
            )
        except Exception:
            logger.exception("follow-up resolution failed; using the question as asked")
            return Resolution(question=question, original=question, reason="resolver unavailable")

        candidate = reply.strip().strip('"').removeprefix("Complete question:").strip()
        candidate = candidate.split("\n")[0].strip().strip('"')
        if reason := check(question, candidate):
            logger.info("discarded rewrite %r: %s", candidate, reason)
            return Resolution(question=question, original=question, reason=reason)

        # A rewrite identical to the question is not a rewrite. Reporting it as
        # one put "read as: and 2023?" on screen, which claims the ellipsis was
        # resolved while showing that it was not.
        if candidate.casefold() == question.strip().casefold():
            return Resolution(question=question, original=question, reason="unchanged")

        return Resolution(question=candidate, original=question, rewritten=True)
