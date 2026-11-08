"""Composition root.

The only module permitted to import from ``adapters``; everything else depends on ``ports``.
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
    """Answer synthesis. The larger model."""

    utility: ChatModel
    """Routing, rewriting and extraction. Smaller model, called many times per question."""

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
                # Set for Ollama Cloud, with base_url https://ollama.com.
                api_key=(settings.llm.api_key.get_secret_value() if settings.llm.api_key else None),
            )

        case LLMProvider.OPENAI_COMPATIBLE:
            from myscholar.adapters.llm.openai_compatible import (
                DEFAULT_BASE_URL,
                OpenAICompatibleChatModel,
            )

            if settings.llm.api_key is None:  # pragma: no cover - validated in settings
                msg = "openai_compatible provider requires MYSCHOLAR_LLM__API_KEY"
                raise ConfigurationError(msg)
            # The default base_url is Ollama's local address; a hosted provider
            # falls back to the adapter's own default endpoint.
            local_default = settings.llm.base_url.startswith(("http://localhost", "http://127."))
            return OpenAICompatibleChatModel(
                model,
                api_key=settings.llm.api_key.get_secret_value(),
                base_url=DEFAULT_BASE_URL if local_default else settings.llm.base_url,
                temperature=settings.llm.temperature,
                timeout_seconds=settings.llm.timeout_seconds,
                reasoning_effort=settings.llm.reasoning_effort,
                seed=settings.llm.seed,
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

    Adapters are imported lazily, so running with fakes needs neither torch nor onnxruntime.
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
