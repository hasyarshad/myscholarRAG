"""Shared test fixtures and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from myscholar.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build settings ignoring any ``.env`` in the working tree.

    Without this, a developer's local ``.env`` leaks into the suite and tests
    pass or fail depending on whose machine they run on.

    ``_env_file`` is a pydantic-settings runtime parameter rather than a
    declared field, so it is invisible to the type checker.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def settings() -> Settings:
    """Default settings: fake components, no models, no network."""
    return make_settings()
