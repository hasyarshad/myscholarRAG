"""The container assembles components and catches misconfiguration at startup."""

from __future__ import annotations

import pytest

from myscholar.container import ConfigurationError, build_container
from tests.conftest import make_settings


def test_builds_fake_components_by_default() -> None:
    container = build_container(make_settings())

    assert container.generator.model_id == "fake"
    assert container.embedder.dim == 768
    assert container.reranker.model_id == "noop"


def test_generator_and_utility_are_separate_instances() -> None:
    """Two models, because grading is called many times and synthesis once or twice."""
    container = build_container(make_settings())

    assert container.generator is not container.utility


def test_embedder_dimension_mismatch_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose dimension contradicts the configuration fails immediately.

    This is the structural guard against the defect in the code this project
    replaces, where indexing and querying used different models with different
    dimensions and nothing noticed.

    The real embedders take their dimension from the loaded weights, so the two
    can genuinely disagree. HashEmbedder is built to whatever the configuration
    asks for and cannot, hence the substitute here.
    """
    from myscholar.adapters.fakes import HashEmbedder

    settings = make_settings()  # embedding.dim defaults to 768
    monkeypatch.setattr(
        "myscholar.container._build_embedder",
        lambda _settings: HashEmbedder(dim=1024, model_id="stub/1024d"),
    )

    with pytest.raises(ConfigurationError, match="1024-dimensional"):
        build_container(settings)


def test_no_heavy_adapters_are_imported_for_fakes() -> None:
    """Running with fakes must not import the GPU or ONNX adapters.

    Adapter imports are lazy so the unit suite and CI stay installable without
    multi-gigabyte machine learning dependencies.

    This checks *our* modules rather than ``torch`` itself: whether torch is in
    sys.modules depends on what else ran first in the same process, so the
    obvious assertion passes alone and fails alongside the integration tests.
    Our adapter modules are only ever imported by the container.
    """
    import sys

    heavy = [
        name
        for name in sys.modules
        if name.startswith(("myscholar.adapters.embedding", "myscholar.adapters.reranking"))
    ]
    for name in heavy:
        del sys.modules[name]

    build_container(make_settings())

    assert not [
        name
        for name in sys.modules
        if name.startswith(("myscholar.adapters.embedding", "myscholar.adapters.reranking"))
    ]
