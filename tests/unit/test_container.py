"""The container assembles components and catches misconfiguration at startup."""

from __future__ import annotations

import pytest

from myscholar.container import ConfigurationError, build_container
from tests.conftest import make_settings


def test_builds_fake_components_by_default() -> None:
    container = build_container(make_settings())

    assert container.generator.model_id == "fake"
    assert container.embedder.dim == 768


def test_generator_and_utility_are_separate_instances() -> None:
    container = build_container(make_settings())

    assert container.generator is not container.utility


def test_embedder_dimension_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    from myscholar.adapters.fakes import HashEmbedder

    settings = make_settings()  # embedding.dim defaults to 768
    # Real embedders take their dimension from the loaded weights. HashEmbedder follows
    # the configuration, so a stub with a different dimension stands in for one.
    monkeypatch.setattr(
        "myscholar.container._build_embedder",
        lambda _settings: HashEmbedder(dim=1024, model_id="stub/1024d"),
    )

    with pytest.raises(ConfigurationError, match="1024-dimensional"):
        build_container(settings)


def test_no_heavy_adapters_are_imported_for_fakes() -> None:
    import sys

    # Checks our adapter modules, not torch: whether torch is in sys.modules depends
    # on which tests ran earlier in the same process.
    heavy = [name for name in sys.modules if name.startswith("myscholar.adapters.embedding")]
    for name in heavy:
        del sys.modules[name]

    build_container(make_settings())

    assert not [name for name in sys.modules if name.startswith("myscholar.adapters.embedding")]
