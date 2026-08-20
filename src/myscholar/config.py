"""Typed application settings.

One source of truth for every tunable. Settings are read from the environment
with the ``MYSCHOLAR_`` prefix; nested groups use a double underscore, so
``MYSCHOLAR_LLM__PROVIDER`` sets ``settings.llm.provider``.

Keeping provider choices here — rather than at call sites — is what allows the
same code to run against a local GPU in development and a hosted API in
production.
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

    onnx_path: Path = Path("artifacts/embedder-onnx")
    """Where `scripts/export_onnx.py` wrote the graph this serves.

    The float32 export, not the quantised one, and the reason is that there is
    nothing to trade. float32 agrees with PyTorch at 1.00000 cosine and returns
    identical retrieval metrics to three decimals across all five evaluated
    configurations, so serving it is indistinguishable from serving the model
    the index was built with. int8 agrees at 0.9888, which is close enough to
    look right and far enough to reorder near neighbours. It costs 306MB and
    126ms per batch less — neither of which matters beside a hosted generation
    call taking seconds. Point this at `embedder-onnx-int8` only if a host
    forces the size, and re-index the corpus with it if you do.
    """

    onnx_model_id: str | None = None
    """Recorded with every vector this produces. None means "same as model_id".

    Sharing the identifier is a claim, and the parity check is what earns it:
    the float32 export agrees with PyTorch to 1.00000 on every probe, so the
    two are the same embedder in every way the index can observe. That is what
    lets ingestion run on the GPUs here and the container serve those vectors
    from CPU without re-embedding anything.

    Set it explicitly to something distinct when the graph is *not* equivalent
    — int8 is the live example. The startup assertion then refuses the mismatch
    instead of letting it degrade quietly, which is the old two-embedder bug.
    """

    onnx_threads: int = 0
    """0 leaves ONNX Runtime's default. On a two-core host, oversubscribing
    threads makes latency worse."""

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

    Convenient locally, where the database is a user process owned by the same
    person and the conversation table would otherwise only exist on a machine
    that had run an ingest. Wrong in production for two reasons: the serving
    role should hold no DDL rights at all, so the attempt either fails or proves
    the role is over-privileged; and every replica racing to create the same
    tables is a deadlock waiting for a second instance.

    Set MYSCHOLAR_DATABASE__MIGRATE_ON_STARTUP=false there and run
    `myscholar db migrate` once, as an admin, before deploying.
    """


class RetrievalSettings(BaseModel):
    top_k: int = 50
    """Candidates fetched per branch before fusion."""

    context_blocks: int = 8
    """How many context blocks reach the generator.

    Named for what it does. It was `rerank_k` — the number surviving a
    cross-encoder — but the cross-encoder was measured to make retrieval worse
    on this corpus and has been removed, so the name described a stage that no
    longer exists.
    """

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

    deadline_seconds: float = 90.0
    """Wall clock for one question.

    45s was too tight: a worst-case path is three generations plus grounding
    checks, and a 14b model takes 8-10s per generation, so the deadline tripped
    on answers that were already valid.
    """


class PersonSettings(BaseModel):
    """Whose work this is.

    Configurable rather than hardcoded, because the prompts, the canned
    replies and the widget copy all name the person, and "this researcher"
    reads like a template nobody finished filling in.
    """

    name: str = "Muhammad Haseeb Arshad"
    short_name: str = "Haseeb"
    site_url: str = "https://haseeb-arshad.me"


class RegistrySettings(BaseModel):
    """Where bibliographic facts are resolved from.

    A DOI found on a paper's first page is matched by regex against the
    literal text, so it cannot be invented. Resolving it replaces every other
    bibliographic field with the record the publisher deposited, which removes
    both the review step and the errors that survived it.
    """

    enabled: bool = True
    """Turn off to fall back to extraction plus the review file.

    Kept switchable because it makes ingestion depend on a network service,
    and a corpus must remain buildable offline.
    """

    mailto: str = "haseeb.arshad.ee@gmail.com"
    """Contact address sent to Crossref, which routes to its polite pool.

    Not authentication. Crossref serves anonymous traffic from a slower,
    rate-limited pool, so omitting this looks like the service being flaky.
    """

    base_url: str = "https://api.crossref.org"
    timeout_seconds: float = 20.0
    publications_url: str = "https://haseeb-arshad.me/research"
    """The page listing which papers are actually his.

    Deliberately his own page rather than an author profile. OpenAlex's
    profile for this name is merged with other people called Arshad and
    credits him with a cangrelor neurointervention review and a 1989
    air-conditioning paper; ORCID is clean but lists only half the corpus.
    A page he maintains is the only source that is both correct and complete —
    checked against the index, the two agree exactly, 20 for 20.
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

    Exact origins, never a wildcard. Several localhost ports are listed
    because a dev server lands wherever the port is free — hardcoding one and
    then starting the site on another produced a preflight 400 and a widget
    reporting "could not reach the assistant", with the API perfectly healthy.

    Worth restating that CORS is a browser-only control: it stops a page on
    another origin, and stops nothing from curl. The real gate in production is
    the shared secret the Next.js proxy injects.
    """

    shared_secret: SecretStr | None = None
    """Required on every /v1 request when set. Empty in development.

    This is the control that actually restricts who reaches the endpoint. It is
    held by the Next.js route handler, which runs on a server, so it never
    enters a browser bundle — and it must not, since a secret shipped to the
    client is decoration.

    It authenticates *the site*, not the visitor, which is the right granularity
    here: anyone may ask questions through the page, nobody may point a load
    generator at a free-tier container directly. Generate with
    `openssl rand -hex 32`.
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
