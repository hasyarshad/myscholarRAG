"""The language model interface.

Implemented by Ollama (local GPU), Groq (hosted, free tier) and a fake used in
tests. Nothing above this layer knows which is in play.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class ChatModel(ABC):
    """A chat-completion backend.

    Three methods, deliberately: free text, streamed text, and schema-constrained
    structured output. The pipeline uses ``structured`` for every classification
    and grading decision, because parsing prose for a verdict is a reliability
    problem you do not need.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the model actually in use, for logging and eval manifests."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a complete response."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text incrementally."""

    @abstractmethod
    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        """Generate a response conforming to ``schema``.

        Implementations should constrain decoding to the schema where the
        backend supports it, rather than parsing free text and hoping.
        """
