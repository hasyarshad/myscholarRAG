"""The embedding interface.

Carries its own identity. ``model_id`` and ``dim`` are attributes of the object
doing the embedding rather than free-floating configuration, so the value stored
with each vector cannot drift from the model that produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

type Vector = list[float]


class Embedder(ABC):
    """Turns text into vectors.

    One class serves both indexing and querying. Some models require different
    prefixes for documents and queries, which is why the two methods exist —
    but they share weights and identity by construction.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stored alongside every vector and asserted against at startup."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality. Must match the database column."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed passages for indexing."""

    @abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """Embed a single search query."""
