"""Settings resolve from the environment and reject impossible combinations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from myscholar.config import EmbeddingProvider, LLMProvider
from tests.conftest import make_settings


def test_defaults_select_fakes() -> None:
    """Out of the box the system needs no models, no GPU and no network."""
    settings = make_settings()

    assert settings.llm.provider is LLMProvider.FAKE
    assert settings.embedding.provider is EmbeddingProvider.HASH
    assert settings.llm.temperature == 0.0


def test_nested_env_vars_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYSCHOLAR_LLM__PROVIDER", "ollama")
    monkeypatch.setenv("MYSCHOLAR_LLM__GENERATOR_MODEL", "qwen3:32b")
    monkeypatch.setenv("MYSCHOLAR_RETRIEVAL__TOP_K", "77")

    settings = make_settings()

    assert settings.llm.provider is LLMProvider.OLLAMA
    assert settings.llm.generator_model == "qwen3:32b"
    assert settings.retrieval.top_k == 77


def test_hosted_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting Groq without a key fails at startup, not at first request.

    A configuration error surfacing on the first visitor question rather than at
    boot is the kind of thing that only shows up in production.
    """
    monkeypatch.setenv("MYSCHOLAR_LLM__PROVIDER", "groq")
    monkeypatch.delenv("MYSCHOLAR_LLM__API_KEY", raising=False)

    with pytest.raises(ValidationError, match="api_key"):
        make_settings()


def test_hosted_provider_accepts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYSCHOLAR_LLM__PROVIDER", "groq")
    monkeypatch.setenv("MYSCHOLAR_LLM__API_KEY", "gsk-not-a-real-key")

    settings = make_settings()

    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "gsk-not-a-real-key"


def test_api_key_is_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets must not leak into logs or tracebacks via repr."""
    monkeypatch.setenv("MYSCHOLAR_LLM__PROVIDER", "groq")
    monkeypatch.setenv("MYSCHOLAR_LLM__API_KEY", "gsk-super-secret")

    assert "gsk-super-secret" not in repr(make_settings())


def test_the_suite_is_isolated_from_a_developers_env_file() -> None:
    """A local .env must never decide whether the tests pass.

    Found the hard way: filling in MYSCHOLAR_API__SHARED_SECRET for a real
    deployment made seventeen unrelated tests fail, because `load_settings()`
    reads `.env` and the API then demanded a secret the tests do not send. The
    suite had been quietly coupled to one machine's configuration.
    """
    import pathlib
    import re

    offenders = [
        path.name
        for path in pathlib.Path("tests").rglob("*.py")
        if re.search(r"^from myscholar\.config import .*load_settings", path.read_text(), re.M)
    ]
    assert offenders == [], (
        f"{offenders} call load_settings(), which reads .env. "
        f"Use make_settings() from tests/conftest.py instead."
    )
