"""A chat model that returns scripted responses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from pydantic import BaseModel

from myscholar.ports.llm import ChatModel, Message

T = TypeVar("T", bound=BaseModel)

_DEFAULT_REPLY = (
    "This is a placeholder response from the fake language model. No inference was performed."
)


class FakeChatModel(ChatModel):
    """Returns canned text, optionally keyed by a substring of the prompt.

    ``responses`` maps a substring to the reply returned whenever that substring
    appears anywhere in the conversation. This is enough to drive the classifier,
    the grader and the answer generator through every branch of the adaptive loop
    deterministically.

    Calls are recorded on ``calls`` so tests can inspect prompts when they need
    to — but assertions should generally target behaviour, not call counts.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        *,
        default: str = _DEFAULT_REPLY,
        structured_responses: Sequence[BaseModel] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self._structured = list(structured_responses or [])
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def _reply_for(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        haystack = "\n".join(m.content for m in messages)
        for needle, reply in self._responses.items():
            if needle.lower() in haystack.lower():
                return reply
        return self._default

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._reply_for(messages)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        # Yield word by word so streaming consumers get realistic chunking,
        # including the split-marker case the UI has to buffer around.
        for word in self._reply_for(messages).split(" "):
            yield word + " "

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        self.calls.append(list(messages))
        for candidate in self._structured:
            if isinstance(candidate, schema):
                self._structured.remove(candidate)
                return candidate
        msg = (
            f"FakeChatModel has no queued structured response of type {schema.__name__}. "
            "Pass one via structured_responses=[...]."
        )
        raise LookupError(msg)
