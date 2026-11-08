"""The language model interface.

Implemented by Ollama, any OpenAI-compatible endpoint, and a fake used in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, model_validator

T = TypeVar("T", bound=BaseModel)


class ModelOutput(BaseModel):
    """A schema a language model fills in, coerced to its declared bounds.

    Constrained decoding enforces a response's shape but not ``maxLength`` or
    ``maxItems``: Ollama and the hosted OpenAI-compatible endpoints tested,
    Google's included, ignore them. Callers of ``structured`` treat a validation
    error as a failure and fall back, so one over-long field would silently
    disable a step. Long strings are therefore truncated, long lists trimmed and
    out-of-range numbers clamped.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_to_bounds(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)
        for name, field in cls.model_fields.items():
            if name not in coerced:
                continue
            value = coerced[name]
            for meta in field.metadata:
                limit = getattr(meta, "max_length", None)
                if limit is not None and isinstance(value, str | list) and len(value) > limit:
                    value = value[:limit]
                upper = getattr(meta, "le", None)
                if upper is not None and isinstance(value, int | float) and value > upper:
                    value = upper
                lower = getattr(meta, "ge", None)
                if lower is not None and isinstance(value, int | float) and value < lower:
                    value = lower
            coerced[name] = value
        return coerced


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class ChatModel(ABC):
    """A chat-completion backend: free text, streamed text and schema-constrained output.

    The pipeline uses ``structured`` for classification and verification
    decisions, so no verdict is parsed out of prose.
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

        Implementations constrain decoding to the schema where the backend supports it.
        """
