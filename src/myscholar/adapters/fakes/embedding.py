"""A deterministic embedder that hashes text instead of running a model."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from myscholar.ports.embedding import Embedder, Vector


class HashEmbedder(Embedder):
    """Produces stable, normalised vectors by hashing token text.

    The vectors carry no semantic meaning, so retrieval *quality* cannot be
    tested with this. What it does give is everything else: correct
    dimensionality, deterministic output, identical vectors for identical text,
    and instant execution with no model download.

    That covers the properties most worth testing — that indexing and querying
    agree, that dimensions match the schema, and that the plumbing works.
    """

    def __init__(self, dim: int = 768, model_id: str = "fake/hash-embedder") -> None:
        self._dim = dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, text: str) -> Vector:
        # Bag-of-words projection: each token contributes to one bucket. Similar
        # texts therefore share buckets, which makes ordering weakly meaningful
        # and keeps fixture-based tests from being entirely arbitrary.
        buckets = [0.0] * self._dim
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            buckets[int.from_bytes(digest, "big") % self._dim] += 1.0

        norm = math.sqrt(sum(v * v for v in buckets))
        if norm == 0.0:
            # Unit vector along the first axis, so empty text stays normalised
            # rather than producing a zero vector cosine distance can't handle.
            buckets[0] = 1.0
            return buckets
        return [v / norm for v in buckets]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._encode(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._encode(text)
