"""Composition root.

This is the only module permitted to import from ``adapters``. Everything else
depends on ``ports``. Concentrating the choice here is what lets the same
pipeline run against a local GPU in development and a hosted API in production
without touching application code.

See docs/learning/00-architecture-and-ports.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from myscholar.config import (
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
    load_settings,
)
from myscholar.ports import ChatModel, Embedder, Reranker


class ConfigurationError(RuntimeError):
    """Settings describe a system that cannot be built or would misbehave."""


@dataclass(frozen=True, slots=True)
class Container:
    """Assembled components, ready to use."""

    settings: Settings
    generator: ChatModel
    """Answer synthesis. Larger model, called once or twice per question."""

    utility: ChatModel
    """Classification and grading. Smaller model, called many times per question."""

    embedder: Embedder
    reranker: Reranker


def _build_chat(settings: Settings, *, model: str) -> ChatModel:
    match settings.llm.provider:
        case LLMProvider.FAKE:
            from myscholar.adapters.fakes import FakeChatModel

            return FakeChatModel()

        case LLMProvider.OLLAMA:
            from myscholar.adapters.llm.ollama import OllamaChatModel

            return OllamaChatModel(
                model,
                base_url=settings.llm.base_url,
                temperature=settings.llm.temperature,
                timeout_seconds=settings.llm.timeout_seconds,
                suppress_reasoning=settings.llm.suppress_reasoning,
            )

        case LLMProvider.GROQ:
            from myscholar.adapters.llm.groq import GroqChatModel

            if settings.llm.api_key is None:  # pragma: no cover - validated in settings
                msg = "groq provider requires MYSCHOLAR_LLM__API_KEY"
                raise ConfigurationError(msg)
            return GroqChatModel(
                model,
                api_key=settings.llm.api_key.get_secret_value(),
                temperature=settings.llm.temperature,
                timeout_seconds=settings.llm.timeout_seconds,
            )


def _build_embedder(settings: Settings) -> Embedder:
    match settings.embedding.provider:
        case EmbeddingProvider.HASH:
            from myscholar.adapters.fakes import HashEmbedder

            return HashEmbedder(dim=settings.embedding.dim)

        case EmbeddingProvider.TORCH:
            msg = "The torch embedder arrives in Phase 1 (ingestion)."
            raise ConfigurationError(msg)

        case EmbeddingProvider.ONNX:
            msg = "The ONNX embedder arrives in Phase 7 (deployment)."
            raise ConfigurationError(msg)


def _build_reranker(settings: Settings) -> Reranker:
    match settings.reranker.provider:
        case RerankerProvider.NOOP:
            from myscholar.adapters.fakes import NoopReranker

            return NoopReranker()

        case RerankerProvider.TORCH:
            msg = "The torch reranker arrives in Phase 2 (retrieval)."
            raise ConfigurationError(msg)

        case RerankerProvider.ONNX:
            msg = "The ONNX reranker arrives in Phase 7 (deployment)."
            raise ConfigurationError(msg)


def build_container(settings: Settings | None = None) -> Container:
    """Assemble the application from settings.

    Adapter modules are imported lazily inside each branch so that running with
    fakes needs neither torch nor onnxruntime installed. That keeps the unit
    test suite and CI free of heavyweight dependencies.
    """
    settings = settings or load_settings()

    embedder = _build_embedder(settings)
    if embedder.dim != settings.embedding.dim:
        msg = (
            f"Embedder {embedder.model_id} produces {embedder.dim}-dimensional vectors "
            f"but embedding.dim is configured as {settings.embedding.dim}."
        )
        raise ConfigurationError(msg)

    return Container(
        settings=settings,
        generator=_build_chat(settings, model=settings.llm.generator_model),
        utility=_build_chat(settings, model=settings.llm.utility_model),
        embedder=embedder,
        reranker=_build_reranker(settings),
    )
