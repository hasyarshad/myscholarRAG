"""Resolving a follow-up into a question that stands on its own.

"how many in 2025?" after "How many did he publish in 2026?" has nothing to embed. The
follow-up is rewritten once, before classification, so later stages never see history.

Turns are stored server-side and never read from a request, since a client could forge
an assistant turn. A rewrite that fails the deterministic checks is discarded.
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

#: A pronoun with nothing in the question to bind it to: "what data did it use?".
#: "he", "his" and "her" are absent because they refer to the corpus owner.
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
    """Whether the question appears to depend on the previous turn.

    Errs toward yes: a false positive costs one utility-model call, and the checks
    usually discard the rewrite.
    """
    text = question.strip()
    if not text:
        return False

    # Before the topic hint: "what method did it use?" names a topic but leaves the
    # subject in the previous turn.
    if len(text) <= _PRONOUN_LENGTH_LIMIT and _UNBOUND_PRONOUN.search(text):
        return True

    if len(text) > _SHORT_ENOUGH_TO_BE_ELLIPTICAL:
        return False
    if _STANDALONE_HINT.search(text):
        return False
    return bool(_ELLIPTICAL_OPENING.match(text)) or len(_WORD.findall(text)) <= 4


#: Two examples and no "repeat it unchanged" option: offered one, the 4b utility model
#: returned every follow-up verbatim. This only runs on questions judged elliptical.
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
        # A refused turn has no subject to resolve against; the model would invent one.
        lines.append(f"A: {turn.answer[:400]}" if not turn.refused else "A: (not answered)")
    return "\n".join(lines)


def _numbers(text: str) -> set[str]:
    return set(_YEAR.findall(text)) | {m.group(0) for m in _YEAR.finditer(text)}


def check(original: str, rewritten: str) -> str:
    """Return the reason a rewrite is unusable, or an empty string."""
    candidate = rewritten.strip().strip('"').split("\n")[0].strip()
    if not candidate:
        return "empty rewrite"
    if len(candidate) > MAX_RESOLVED_CHARS:
        return "rewrite too long to be a question"

    # Every year in the follow-up must survive, or "how many in 2025?" can turn
    # into a question about the previous turn's year.
    lost = _numbers(original) - _numbers(candidate)
    if lost:
        return f"dropped {', '.join(sorted(lost))} from the question"

    # A rewrite shorter than the original has removed something.
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

        # An unchanged rewrite is not reported as one, so no "read as" line is shown.
        if candidate.casefold() == question.strip().casefold():
            return Resolution(question=question, original=question, reason="unchanged")

        return Resolution(question=candidate, original=question, rewritten=True)
