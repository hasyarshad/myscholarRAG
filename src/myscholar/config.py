"""Typed application settings.

One source of truth for every tunable. Settings are read from the environment
with the ``MYSCHOLAR_`` prefix; nested groups use a double underscore, so
``MYSCHOLAR_LLM__PROVIDER`` sets ``settings.llm.provider``.

Keeping provider choices here — rather than at call sites — is what allows the
same code to run against a local GPU in development and a hosted API in
production. See docs/learning/00-architecture-and-ports.md.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    GROQ = "groq"
    FAKE = "fake"


class EmbeddingProvider(StrEnum):
    TORCH = "torch"
    ONNX = "onnx"
    HASH = "hash"


class RerankerProvider(StrEnum):
    TORCH = "torch"
    ONNX = "onnx"
    NOOP = "noop"


class LLMSettings(BaseModel):
    """Which language model backend to use, and how to reach it."""

    provider: LLMProvider = LLMProvider.FAKE

    generator_model: str = "qwen3:14b"
    """Model for final answer synthesis. Larger, slower, called once or twice.

    A reasoning model. Ollama separates its reasoning trace into a distinct
    field, so the answer text stays clean, but the trace still consumes the
    token budget and adds latency.
    """

    utility_model: str = "qwen3:4b-instruct"
    """Model for classification and grading. Called many times per question.

    Deliberately the *instruct* variant, not plain ``qwen3:4b``. The latter is
    reasoning-only: it spends its whole token budget thinking before emitting
    anything, so a short grading call returns empty content. Measured on a live
    server — ``qwen3:4b`` used 16 tokens producing nothing, the instruct variant
    answered in 2.
    """

    suppress_reasoning: bool = True
    """Append Qwen3's ``/no_think`` directive to shorten reasoning traces.

    Harmless for instruct models and measurably cheaper for reasoning ones.
    Note this is *not* Ollama's ``think`` parameter, which was measured to make
    matters worse — see the Ollama adapter.
    """

    base_url: str = "http://localhost:11434"
    api_key: SecretStr | None = None

    temperature: float = 0.0
    """Zero by default. Grounded extraction wants determinism, not creativity."""

    timeout_seconds: float = 120.0

    @model_validator(mode="after")
    def _require_key_for_hosted(self) -> LLMSettings:
        if self.provider is LLMProvider.GROQ and self.api_key is None:
            msg = (
                "llm.provider is 'groq' but no api_key was supplied. "
                "Set MYSCHOLAR_LLM__API_KEY (free key at https://console.groq.com)."
            )
            raise ValueError(msg)
        return self


class EmbeddingSettings(BaseModel):
    """The embedder must be identical at index time and at query time.

    ``model_id`` is stored alongside every vector
    and checked at startup.
    """

    provider: EmbeddingProvider = EmbeddingProvider.HASH
    model_id: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    batch_size: int = 32
    device: str = "cuda"
    """Only consulted by the torch adapter; ONNX is CPU-only by design."""


class RerankerSettings(BaseModel):
    provider: RerankerProvider = RerankerProvider.NOOP
    model_id: str = "BAAI/bge-reranker-base"
    device: str = "cuda"


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://myscholar:myscholar@localhost:5433/myscholar"
    echo: bool = False


class RetrievalSettings(BaseModel):
    top_k: int = 50
    """Candidates fetched per branch before fusion."""

    rerank_k: int = 8
    """Candidates surviving the cross-encoder and passed to generation."""

    window: int = 2
    """Chunks of context pulled either side of a hit during parent expansion."""

    max_span_chunks: int = 8
    """Cap on how many chunks one expanded span may contain."""

    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    """Equal to the dense weight, which was measured rather than assumed.

    The initial 0.4 encoded a prior that meaning matters more than term
    overlap. Swept over the retrieval test set, that prior was wrong for this
    corpus: 0.4 scored MRR 0.887 while 1.0 scored 0.901, and 0.1-0.2 were worse
    than disabling the branch entirely. A corpus this dense in identifiers
    (FCS-MPCC, NSGA-II, T-Type VSI) rewards exact term matching more than
    general-purpose advice suggests.
    """
    rrf_k: int = 60
    """Reciprocal rank fusion constant. 60 is the value from the original paper."""


class ReasoningSettings(BaseModel):
    """Bounds guaranteeing the adaptive loop terminates."""

    max_steps: int = 24
    max_retrieval_rounds: int = 3
    max_regenerations: int = 2

    deadline_seconds: float = 90.0
    """Wall clock for one question.

    45s was too tight: a worst-case path is three generations plus grounding
    checks, and a 14b model takes 8-10s per generation, so the deadline tripped
    on answers that were already valid.
    """


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MYSCHOLAR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)

    papers_dir: Path = Path("papers")
    data_dir: Path = Path("data")


def load_settings() -> Settings:
    """Build settings from the environment and .env."""
    return Settings()
