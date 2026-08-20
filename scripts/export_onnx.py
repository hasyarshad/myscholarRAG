"""Export the embedder to ONNX, quantise it, and prove it still works.

    uv run --extra export python scripts/export_onnx.py

Production cannot run PyTorch. The free tiers this deploys to give two CPU
cores and no GPU, and `torch` alone is about two gigabytes before any model
weights. ONNX Runtime is 15MB and the quantised embedder is roughly 110MB, so
the difference is between an image that starts and one that does not.

What ONNX does and does not buy is worth being precise about, because the usual
framing is wrong. It removes *framework overhead* — the Python dispatch, the
autograd bookkeeping, the general-purpose kernels — and replaces them with a
fixed graph and kernels chosen for the shapes actually used. It does not remove
parameters. That is why it is the right tool for a 110M embedder and useless
for a 14B generator: nobody serves a 14B model on two CPU cores, which is why
generation moves to a hosted API rather than into this file.

**The parity check is the point of this script.** A broken export does not
raise. It produces vectors, the vectors have the right shape, cosine similarity
returns plausible numbers, and retrieval quietly gets worse in a way no test
would catch — the same failure mode as the two-embedder bug this project was
built to replace. So every export is verified against the PyTorch model it came
from, on real corpus text, and the script fails if agreement drops below the
threshold.

Int8 is exported too, and is *not* what ships. The two wins are independent and
have different risk profiles: dropping the framework is arithmetic-preserving —
float32 agrees with PyTorch to 1.00000 and retrieves identically to five decimal
places — while quantising changes the numbers, reaching 0.9888. Since float32 is
provably lossless and the 306MB it costs are free on the target host, there is
nothing to trade and int8 is kept only for comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

#: Agreement below which the export is rejected.
#:
#: Not a round number chosen for comfort: retrieval ranks by cosine similarity,
#: and two vectors agreeing at 0.99 rank almost identically while 0.95 reorders
#: neighbours. The check is on the *embeddings*, so it fails before a bad model
#: reaches an index.
MIN_COSINE = 0.99

#: Sentences the parity check runs on. Drawn from the corpus rather than
#: invented, because a quantised model degrades unevenly and the vocabulary
#: that matters here is dense with identifiers — FCS-MPCC, NSGA-II, LSPMSM —
#: which tokenise into many subwords and are exactly where precision loss shows.
PROBES = [
    "Weighting factors are selected using NSGA-II combined with TOPSIS decision making.",
    "The proposed framework achieves 98.53% accuracy on the held-out recordings.",
    "Chaotic SVPWM distributes the switching harmonics across a wider frequency band.",
    "A gradient boosting machine serves as the meta-learner over out-of-fold vectors.",
    "Muhammad Haseeb Arshad is a PhD candidate in Electrical & Computer Engineering.",
    "FCS-MPCC of permanent magnet synchronous machines with a self-tuning PI speed loop.",
    "how would you rate his contribution to machine learning",
    "How many papers were published in 2023?",
]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)


def pool(hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Reduce token vectors to one sentence vector, then L2 normalise.

    This has to match what the PyTorch path does or parity measures the pooling
    rather than the export — and that is not hypothetical. The first version
    assumed mean pooling, which is the common default, and bge uses `cls`.
    Every probe came back at 0.973 against a *float32* export that should have
    agreed to five decimal places, and the check refused to ship it. Guessing
    would have produced an index subtly misaligned with its queries: the exact
    defect this project was built to replace, reintroduced at the last step.

    So the mode is read from the model's own configuration rather than assumed,
    and passed through to the adapter that serves it.
    """
    if mode == "cls":
        pooled = hidden[:, 0]
    else:
        expanded = mask[..., None].astype(hidden.dtype)
        counts = np.clip(expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = (hidden * expanded).sum(axis=1) / counts
    normalised = pooled / np.linalg.norm(pooled, axis=-1, keepdims=True)
    return np.asarray(normalised)


def pooling_mode(model_id: str) -> str:
    """What the sentence-transformers model actually does, not what is usual."""
    from sentence_transformers import SentenceTransformer

    for module in SentenceTransformer(model_id, device="cpu"):
        config: dict[str, object] = getattr(module, "get_config_dict", dict)()
        if mode := config.get("pooling_mode"):
            return str(mode)
    return "mean"


def export(model_id: str, out: Path) -> Path:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {model_id} -> {out}", flush=True)

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(out)
    # transformers ships types but leaves this factory unannotated, so a strict
    # run reports the call rather than the code. Narrow ignore, not a blanket
    # override on the package: the rest of transformers is typed and worth
    # checking — an earlier attempt to silence this by ignoring `transformers.*`
    # wholesale turned two correct annotations elsewhere into errors.
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)  # type: ignore[no-untyped-call]
    return out


def quantise(source: Path, out: Path) -> Path:
    """Dynamic int8. Weights are quantised, activations measured at run time.

    Dynamic rather than static because static needs a calibration set and buys
    little for a model this size, and because a calibration set drawn from one
    corpus is a way to overfit an artifact that should generalise.

    Per-channel, which is not the default and is worth the flag: one scale per
    output channel rather than one for the whole tensor. Measured on the probes,
    agreement with PyTorch went 0.9732 -> 0.9888 for 4.6MB more. ``reduce_range``
    was tried alongside and is much worse here (0.9128) — it exists for older
    AVX2 hardware that saturates on int8 accumulation, and costs a bit of
    precision to buy headroom this machine does not need.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        if (source / name).exists():
            shutil.copy(source / name, out / name)
    if (source / "config.json").exists():
        shutil.copy(source / "config.json", out / "config.json")

    print(f"quantising int8 -> {out}", flush=True)
    quantize_dynamic(
        model_input=source / "model.onnx",
        model_output=out / "model.onnx",
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return out


def _onnx_embed(directory: Path, texts: list[str], mode: str) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    encoded = tokenizer.encode_batch(texts)

    session = ort.InferenceSession(
        str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    inputs = {i.name for i in session.get_inputs()}
    feed: dict[str, Any] = {
        "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
    }
    if "token_type_ids" in inputs:
        feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

    hidden = session.run(None, feed)[0]
    return pool(hidden, feed["attention_mask"], mode)


def _torch_embed(model_id: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device="cpu")
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def verify(model_id: str, directory: Path, label: str, mode: str) -> float:
    """Agreement with PyTorch on real corpus text, per sentence."""
    reference = _torch_embed(model_id, PROBES)
    candidate = _onnx_embed(directory, PROBES, mode)
    scores = cosine(reference, candidate)

    print(f"\n{label}: cosine against PyTorch")
    for text, score in zip(PROBES, scores, strict=True):
        mark = "ok " if score >= MIN_COSINE else "LOW"
        print(f"  {mark} {score:.5f}  {text[:64]}")
    return float(scores.min())


def benchmark(
    float_dir: Path, int8_dir: Path, model_id: str, mode: str, runs: int = 3
) -> dict[str, float]:
    """Wall time for one batch, which is what a request actually pays.

    All three paths on CPU, because that is the comparison that decides the
    deployment. The GPU number is irrelevant to a host that has no GPU.
    """

    def timed(call: Any) -> float:
        call()  # Warm up. Session construction dominates the first call and is
        started = time.perf_counter()  # paid once at boot, not per request.
        for _ in range(runs):
            call()
        return (time.perf_counter() - started) * 1000 / runs

    return {
        "torch_cpu_ms": timed(lambda: _torch_embed(model_id, PROBES)),
        "onnx_float32_ms": timed(lambda: _onnx_embed(float_dir, PROBES, mode)),
        "onnx_int8_ms": timed(lambda: _onnx_embed(int8_dir, PROBES, mode)),
    }


def _size_mb(directory: Path) -> float:
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    float_dir = args.out / "embedder-onnx"
    int8_dir = args.out / "embedder-onnx-int8"

    mode = pooling_mode(args.model)
    print(f"pooling: {mode}")

    export(args.model, float_dir)
    quantise(float_dir, int8_dir)

    # Recorded beside the weights, so the adapter that serves them does not
    # have to guess either.
    for directory in (float_dir, int8_dir):
        (directory / "pooling.json").write_text(json.dumps({"mode": mode}), "utf-8")

    float_score = verify(args.model, float_dir, "float32", mode)
    int8_score = verify(args.model, int8_dir, "int8", mode)

    report: dict[str, Any] = {
        "model": args.model,
        "pooling": mode,
        "float32": {"min_cosine": float_score, "size_mb": round(_size_mb(float_dir), 1)},
        "int8": {"min_cosine": int8_score, "size_mb": round(_size_mb(int8_dir), 1)},
    }
    if not args.skip_benchmark:
        report["latency"] = {
            k: round(v, 1) for k, v in benchmark(float_dir, int8_dir, args.model, mode).items()
        }

    (args.out / "parity.json").write_text(json.dumps(report, indent=2), "utf-8")
    print("\n" + json.dumps(report, indent=2))

    # Only the float32 gate is fatal, because float32 is what ships. An export
    # that produces plausible vectors from a broken graph is the failure this
    # whole script exists to prevent, and shipping it would repeat the defect
    # the project was built to replace.
    if float_score < MIN_COSINE:
        print(f"\nFAILED: float32 agreement below {MIN_COSINE}. Do not ship this artifact.")
        return 1

    print(f"\nfloat32 agrees with PyTorch to {float_score:.5f} — safe to serve the GPU index")
    if int8_score < MIN_COSINE:
        # Not fatal, and not shipped. Kept because the size difference matters
        # on some hosts and because the comparison is part of the evaluation.
        print(f"int8 agrees only to {int8_score:.5f}, so it is not interchangeable with the")
        print("index. Serve float32 unless size forces otherwise; if it does, re-index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
