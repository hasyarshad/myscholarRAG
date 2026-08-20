"""ONNX Runtime embedder — the production path.

PyTorch is about two gigabytes before any weights, and the free tiers this
deploys to give two CPU cores. ONNX Runtime is 15MB and the quantised model
110MB, which is the difference between an image that starts and one that does
not.

Two details are load-bearing and neither is obvious:

**Pooling is read from the artifact, not assumed.** bge uses the CLS token;
mean pooling is the more common default and is wrong here. Assuming it produced
embeddings that agreed with PyTorch at only 0.973 — plausible-looking vectors,
correct shape, quietly misaligned with the index. That is the same failure as
the two-embedder bug this project replaced, and the export's parity check is
what caught it.

**The model id says what this is.** Embeddings are keyed by ``(chunk_id,
model_id)`` and asserted at startup, so an index built with PyTorch cannot be
searched with int8 by accident. They are close but not identical — 0.973 cosine
— and mixing them is exactly the silent degradation the key exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from myscholar.adapters.embedding import QUERY_INSTRUCTION
from myscholar.ports.embedding import Embedder, Vector


def pool(hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Reduce token vectors to one sentence vector, then L2 normalise."""
    if mode == "cls":
        pooled = hidden[:, 0]
    else:
        expanded = mask[..., None].astype(hidden.dtype)
        counts = np.clip(expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = (hidden * expanded).sum(axis=1) / counts
    normalised: np.ndarray = pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)
    return normalised


class OnnxEmbedder(Embedder):
    """Embeds with a quantised ONNX graph on CPU."""

    def __init__(
        self,
        directory: Path | str,
        *,
        model_id: str,
        dim: int = 768,
        batch_size: int = 32,
        max_length: int = 512,
        threads: int = 0,
        query_instruction: str | None = QUERY_INSTRUCTION,
    ) -> None:
        self._directory = Path(directory)
        self._model_id = model_id
        self._dim = dim
        self._batch_size = batch_size
        self._max_length = max_length
        self._threads = threads
        self._query_instruction = query_instruction or ""
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._pooling = "cls"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> tuple[Any, Any]:
        """Built once. Session construction dominates the first call."""
        if self._session is None or self._tokenizer is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            options = ort.SessionOptions()
            if self._threads:
                # Left at the runtime's default unless set. On a two-core free
                # tier, oversubscribing threads makes latency worse, not better.
                options.intra_op_num_threads = self._threads

            self._session = ort.InferenceSession(
                str(self._directory / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(self._directory / "tokenizer.json"))
            self._tokenizer.enable_truncation(max_length=self._max_length)
            self._tokenizer.enable_padding()

            if (config := self._directory / "pooling.json").exists():
                self._pooling = str(json.loads(config.read_text()).get("mode", "cls"))
        return self._session, self._tokenizer

    def _encode(self, texts: Sequence[str]) -> list[Vector]:
        session, tokenizer = self._load()
        encoded = tokenizer.encode_batch(list(texts))

        expected = {i.name for i in session.get_inputs()}
        feed: dict[str, Any] = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        hidden = session.run(None, feed)[0]
        return [row.tolist() for row in pool(hidden, feed["attention_mask"], self._pooling)]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []

        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            # Off the event loop: ONNX Runtime holds the GIL through a session
            # run, and blocking it stalls every other request on the server.
            vectors.extend(await asyncio.to_thread(self._encode, batch))
        return vectors

    async def embed_query(self, text: str) -> Vector:
        vectors = await self.embed_documents([f"{self._query_instruction}{text}"])
        return vectors[0]
