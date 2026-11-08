"""Shared test fixtures and helpers."""

from __future__ import annotations

import os
from typing import Any

import pytest

from myscholar.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build settings that ignore any ``.env`` in the working tree."""
    # ``_env_file`` is a pydantic-settings runtime parameter, not a declared field.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide every MYSCHOLAR_* variable from the process environment.

    ``make_settings`` skips ``.env``, but pydantic-settings still reads ``os.environ``.
    """
    for name in list(os.environ):
        if name.startswith("MYSCHOLAR_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Default settings: fake components, no models, no network."""
    return make_settings()
