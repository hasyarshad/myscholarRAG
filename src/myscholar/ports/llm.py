"""The language model interface.

Implemented by Ollama (local GPU), any OpenAI-compatible hosted endpoint, and a
fake used in tests. Nothing above this layer knows which is in play.
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
    """A schema a language model fills in, tolerant of its own stated bounds.

    Constrained decoding guarantees the *shape* of a response — the fields, their
    types, the allowed enum members — and nothing more. It does not enforce
    `maxLength` or `maxItems`, and the backends disagree about how much they even
    try: Ollama's local grammar ignores them, and so does every hosted
    OpenAI-compatible endpoint measured here, Google's included.

    That matters because every caller of `structured` treats a validation error
    as an outage and degrades quietly. A model writing 260 characters into a
    200-character field therefore does not raise anything a reader would notice
    — it makes the classifier fall back to one class for every question, or a
    query transform silently stop running, or the groundedness check report
    "unavailable" and pass every statement through. All three were observed.

    So bounds declared on these schemas are treated as *limits to enforce*
    rather than *conditions to assert*: over-long strings are truncated,
    over-full lists are trimmed, out-of-range numbers are clamped. The bound
    still holds downstream; it just stops being a way to lose the whole answer.

    Inherited rather than repeated per field, because the failure is systemic —
    nine schemas across query, generation and ingest carry such bounds.
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
