"""Shared test fixtures and helpers."""

from __future__ import annotations

import os
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


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide every MYSCHOLAR_* variable from the process environment.

    ``make_settings`` passes ``_env_file=None``, which stops pydantic reading
    ``.env`` — but pydantic-settings also reads ``os.environ``, and that is not
    covered by it. So a developer who exports ``MYSCHOLAR_API__SHARED_SECRET``
    for a deployment still changes how the suite behaves, and a CI runner with
    a different environment gets different results again.

    Autouse, because the failure mode is tests passing on one machine and
    failing on another, which is exactly the sort of thing nobody remembers to
    opt into.
    """
    for name in list(os.environ):
        if name.startswith("MYSCHOLAR_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Default settings: fake components, no models, no network."""
    return make_settings()
