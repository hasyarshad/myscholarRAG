"""Typed application settings.

Read from the environment with the ``MYSCHOLAR_`` prefix. Nested groups use a double
underscore, so ``MYSCHOLAR_LLM__PROVIDER`` sets ``settings.llm.provider``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OLLAMA = "ollama"

    OPENAI_COMPATIBLE = "openai_compatible"
    """Any endpoint speaking the OpenAI chat-completions format."""

    FAKE = "fake"


class EmbeddingProvider(StrEnum):
    TORCH = "torch"
    ONNX = "onnx"
    HASH = "hash"


class LLMSettings(BaseModel):
    """Which language model backend to use, and how to reach it."""

    provider: LLMProvider = LLMProvider.FAKE

    generator_model: str = "qwen3:14b"
    """Model for answer synthesis.

    A reasoning model: Ollama returns the trace in a separate field, but it still consumes
    the token budget and adds latency.
    """

    utility_model: str = "qwen3:4b-instruct"
    """Model for routing, rewriting and extraction, called many times per question.

    The instruct variant, because plain ``qwen3:4b`` is reasoning-only and spends a short
    call's whole token budget thinking, returning empty content.
    """

    suppress_reasoning: bool = True
    """Append Qwen3's ``/no_think`` directive to shorten reasoning traces.

    Harmless for instruct models. This is not Ollama's ``think`` parameter, which moves the
    reasoning trace into the answer text (see the Ollama adapter).
    """

    reasoning_effort: str | None = None
    """Reasoning effort for a hosted reasoning model that supports it.

    The OpenAI-compatible counterpart to ``suppress_reasoning``. Sent only when set, so a
    model that rejects the parameter is unaffected by the default. ``"low"`` cut reasoning
    tokens and latency with no visible change in the answer.
    """

    seed: int | None = 42
    """Fixed sampling seed, so repeated evaluation runs are comparable.

    Disable it for an endpoint that rejects unknown parameters: Google's OpenAI-compatible
    endpoint returns 400 ``Unknown name "seed"``. ``""``, ``none``, ``null`` and ``off`` all
    mean "do not send it".
    """

    base_url: str = "http://localhost:11434"
    api_key: SecretStr | None = None

    @field_validator("seed", "reasoning_effort", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """Let an environment variable switch an optional setting off."""
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off"}:
            return None
        return value

    temperature: float = 0.0
    """Zero, for deterministic grounded answers."""

    timeout_seconds: float = 120.0

    @model_validator(mode="after")
    def _require_key_for_hosted(self) -> LLMSettings:
        if self.provider is LLMProvider.OPENAI_COMPATIBLE and self.api_key is None:
            msg = (
                "llm.provider is 'openai_compatible' but no api_key was supplied. "
                "Set MYSCHOLAR_LLM__API_KEY (free key at https://aistudio.google.com/apikey)."
            )
            raise ValueError(msg)
        return self


class EmbeddingSettings(BaseModel):
    """The embedder must be identical at index time and at query time.

    ``model_id`` is stored alongside every vector and checked at startup.
    """

    provider: EmbeddingProvider = EmbeddingProvider.HASH
    model_id: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    batch_size: int = 32
    device: str = "cuda"
    """Only used by the torch adapter; ONNX runs on CPU."""

    onnx_path: Path = Path("artifacts/embedder-onnx")
    """Where ``scripts/export_onnx.py`` wrote the graph this serves.

    The float32 export agrees with PyTorch at 1.00000 cosine and gives identical retrieval
    metrics, so it can serve an index built with PyTorch. The int8 export agrees at 0.9888,
    enough to reorder near neighbours; use it only if a host forces the size, and re-index.
    """

    onnx_model_id: str | None = None
    """Recorded with every vector this produces. None means "same as model_id".

    Sharing the identifier is safe for the float32 export, which matches PyTorch exactly, so
    vectors embedded on a GPU can be served from CPU. Set a distinct value for a graph that
    is not equivalent (int8), and the startup check will refuse the mismatch.
    """

    onnx_threads: int = 0
    """0 leaves ONNX Runtime's default. Oversubscribing threads on a two-core host is slower."""

    @model_validator(mode="after")
    def _default_onnx_id(self) -> EmbeddingSettings:
        if self.onnx_model_id is None:
            self.onnx_model_id = self.model_id
        return self


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://myscholar:myscholar@localhost:5433/myscholar"
    echo: bool = False

    migrate_on_startup: bool = True
    """Whether the API creates its own tables when it boots.

    Convenient locally. In production the serving role should hold no DDL rights, and
    replicas racing to create the same tables can deadlock: set
    MYSCHOLAR_DATABASE__MIGRATE_ON_STARTUP=false and run ``myscholar db migrate`` once, as
    an admin, before deploying.
    """


class RetrievalSettings(BaseModel):
    top_k: int = 50
    """Candidates fetched per branch before fusion."""

    context_blocks: int = 8
    """How many context blocks reach the generator."""

    window: int = 2
    """Chunks of context pulled either side of a hit during parent expansion."""

    max_span_chunks: int = 8
    """Cap on how many chunks one expanded span may contain."""

    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    """Equal to the dense weight, as measured on the retrieval test set.

    0.4 scored MRR 0.887 and 1.0 scored 0.901; 0.1-0.2 were worse than disabling the branch.
    The corpus is dense in identifiers (FCS-MPCC, NSGA-II, T-Type VSI), which favours exact
    term matching.
    """
    rrf_k: int = 60
    """Reciprocal rank fusion constant, the value from the original paper."""


class ReasoningSettings(BaseModel):
    """Bounds guaranteeing the adaptive loop terminates."""

    max_steps: int = 24
    max_retrieval_rounds: int = 3

    deadline_seconds: float = 90.0
    """Wall clock for one question.

    45s cut off answers that were already valid with the local ``qwen3:14b`` generator,
    which takes 8-10s per generation.
    """


class PersonSettings(BaseModel):
    """Whose work this is. The prompts, canned replies and widget copy all name the person."""

    name: str = "Muhammad Haseeb Arshad"
    short_name: str = "Haseeb"
    site_url: str = "https://haseeb-arshad.me"


class RegistrySettings(BaseModel):
    """Where bibliographic facts are resolved from.

    A DOI is matched by regex against a paper's first page, so it cannot be invented.
    Resolving it replaces every other bibliographic field with the publisher's record.
    """

    enabled: bool = True
    """Turn off to fall back to extraction plus the review file, so a corpus builds offline."""

    mailto: str = "haseeb.arshad.ee@gmail.com"
    """Contact address sent to Crossref, which routes requests to its polite pool.

    Not authentication. Anonymous traffic goes to a slower, rate-limited pool.
    """

    base_url: str = "https://api.crossref.org"
    timeout_seconds: float = 20.0
    publications_url: str = "https://haseeb-arshad.me/research"
    """The page listing which papers are his.

    His own page, because OpenAlex's profile for this name merges other authors and ORCID
    lists only half the corpus. It agrees with the index, 20 for 20.
    """


class ApiSettings(BaseModel):
    """HTTP service settings."""

    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3100",
            "http://127.0.0.1:3100",
            "https://haseeb-arshad.me",
            "https://www.haseeb-arshad.me",
        ]
    )
    """Browser origins permitted to call the API.

    Exact origins, never a wildcard. Several localhost ports are listed because a dev server
    takes whichever port is free. CORS only restricts browsers; ``shared_secret`` is the
    access control.
    """

    shared_secret: SecretStr | None = None
    """Required on every /v1 request when set. Empty in development.

    Held by the Next.js route handler on the server, so it never reaches a browser bundle.
    It authenticates the site, not the visitor. Generate with ``openssl rand -hex 32``.
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
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reasoning: ReasoningSettings = Field(default_factory=ReasoningSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    person: PersonSettings = Field(default_factory=PersonSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)

    papers_dir: Path = Path("papers")
    data_dir: Path = Path("data")


def load_settings() -> Settings:
    """Build settings from the environment and .env."""
    return Settings()
