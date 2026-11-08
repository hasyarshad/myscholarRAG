"""The embedding interface.

``model_id`` and ``dim`` belong to the embedder itself, so the identity stored with
each vector cannot drift from the model that produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

type Vector = list[float]


class Embedder(ABC):
    """Turns text into vectors.

    Documents and queries have separate methods because some models prefix them
    differently; both use the same model and identity.
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
