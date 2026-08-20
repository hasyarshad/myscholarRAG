"""What every embedder adapter must agree on.

The index is built by the torch adapter on a GPU and searched by the ONNX
adapter on a CPU. Anything the two decide separately is a place they can drift,
and drift here does not raise — it returns ranked, plausible, worse results.

This is not hypothetical. The ONNX adapter shipped without the BGE query
instruction that the torch adapter applies. The two models agreed at 1.00000
cosine, because the models were identical; what differed was the text each was
asked to embed. Retrieval lost 0.07 MRR and the cause looked exactly like
quantisation damage.
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
        """Applying the instruction to passages is worse than omitting it."""
        for adapter in ADAPTERS:
            recorder = _Recorder(adapter)
            asyncio.run(recorder.embedder.embed_documents(["a passage of text"]))
            assert recorder.seen == ["a passage of text"], adapter.__name__

    def test_the_instruction_has_one_definition(self) -> None:
        """A copy in each adapter is how they came apart the first time.

        Read from disk rather than through ``inspect``, which resolves modules
        via ``sys.modules`` and so depends on whatever another test left there.
        """
        package = Path(myscholar.adapters.embedding.__file__).parent
        defining = [
            path.name
            for path in sorted(package.glob("*.py"))
            if "Represent this sentence" in path.read_text("utf-8")
        ]
        assert defining == ["__init__.py"], (
            f"the query instruction is defined in {defining}; import it from the package"
        )
