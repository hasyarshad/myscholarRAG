"""Behaviour every embedder adapter must share.

The index is built by the torch adapter and searched by the ONNX adapter. A difference
between them degrades ranking without raising: a missing query instruction cost 0.07 MRR.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import myscholar.adapters.embedding
from myscholar.adapters.embedding import QUERY_INSTRUCTION
from myscholar.adapters.embedding.onnx_embedder import OnnxEmbedder
from myscholar.adapters.embedding.torch_embedder import TorchEmbedder

ADAPTERS = (TorchEmbedder, OnnxEmbedder)


class _Recorder:
    """Captures what would have been embedded, without loading a model."""

    def __init__(self, adapter: type) -> None:
        self.seen: list[str] = []
        if adapter is TorchEmbedder:
            self.embedder = TorchEmbedder()
        else:
            self.embedder = OnnxEmbedder("unused", model_id="unused")  # type: ignore[assignment]
        self.embedder._encode = self._encode  # type: ignore[method-assign]  # noqa: SLF001

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.0] * 768 for _ in texts]


class TestQueryInstruction:
    def test_every_adapter_prefixes_queries(self) -> None:
        for adapter in ADAPTERS:
            recorder = _Recorder(adapter)
            asyncio.run(recorder.embedder.embed_query("how are weighting factors tuned?"))
            assert recorder.seen == [f"{QUERY_INSTRUCTION}how are weighting factors tuned?"], (
                f"{adapter.__name__} does not apply the query instruction"
            )

    def test_no_adapter_prefixes_passages(self) -> None:
        for adapter in ADAPTERS:
            recorder = _Recorder(adapter)
            asyncio.run(recorder.embedder.embed_documents(["a passage of text"]))
            assert recorder.seen == ["a passage of text"], adapter.__name__

    def test_the_instruction_has_one_definition(self) -> None:
        # Read from disk: ``inspect`` resolves modules through ``sys.modules``,
        # which depends on what other tests left there.
        package = Path(myscholar.adapters.embedding.__file__).parent
        defining = [
            path.name
            for path in sorted(package.glob("*.py"))
            if "Represent this sentence" in path.read_text("utf-8")
        ]
        assert defining == ["__init__.py"], (
            f"the query instruction is defined in {defining}; import it from the package"
        )
