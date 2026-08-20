"""Composition root.

This is the only module permitted to import from ``adapters``. Everything else
depends on ``ports``. Concentrating the choice here is what lets the same
pipeline run against a local GPU in development and a hosted API in production
without touching application code.
"""

from __future__ import annotations

from dataclasses import dataclass

from myscholar.config import (
    EmbeddingProvider,
    LLMProvider,
    Settings,
    load_settings,
)
from myscholar.ports import ChatModel, Embedder
from myscholar.ports.metadata import MetadataResolver


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
    resolver: MetadataResolver | None
    """Bibliographic registry, or None when the corpus is built offline."""


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
                # Set to talk to Ollama Cloud instead of a local server; the
                # base_url changes to https://ollama.com and nothing else does.
                api_key=(settings.llm.api_key.get_secret_value() if settings.llm.api_key else None),
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
            from myscholar.adapters.embedding.torch_embedder import TorchEmbedder

            return TorchEmbedder(
                settings.embedding.model_id,
                device=settings.embedding.device,
                batch_size=settings.embedding.batch_size,
            )

        case EmbeddingProvider.ONNX:
            from myscholar.adapters.embedding.onnx_embedder import OnnxEmbedder

            if not (settings.embedding.onnx_path / "model.onnx").exists():
                msg = (
                    f"No ONNX embedder at {settings.embedding.onnx_path}. "
                    f"Run: uv run --extra export python scripts/export_onnx.py"
                )
                raise ConfigurationError(msg)

            return OnnxEmbedder(
                settings.embedding.onnx_path,
                model_id=settings.embedding.onnx_model_id or settings.embedding.model_id,
                dim=settings.embedding.dim,
                batch_size=settings.embedding.batch_size,
                threads=settings.embedding.onnx_threads,
            )


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
        resolver=_build_resolver(settings),
    )


def _build_resolver(settings: Settings) -> MetadataResolver | None:
    """None when disabled, so ingestion still runs with no network at all."""
    if not settings.registry.enabled:
        return None

    from myscholar.adapters.metadata import CrossrefResolver

    return CrossrefResolver(
        mailto=settings.registry.mailto,
        base_url=settings.registry.base_url,
        timeout_seconds=settings.registry.timeout_seconds,
    )
