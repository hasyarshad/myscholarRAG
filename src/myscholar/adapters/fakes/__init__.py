"""Fake implementations for tests and for the Phase 0 walking skeleton.

These are fakes, not mocks. A mock asserts how it was called and couples tests
to implementation details; a fake is a working implementation with a shortcut.
Tests here assert on behaviour, so they survive refactoring.

Everything in this package runs with no GPU, no network and no database, which
is what allows the suite to run in under a second and to work on free CI
runners.
"""

from myscholar.adapters.fakes.chat import FakeChatModel
from myscholar.adapters.fakes.embedding import HashEmbedder
from myscholar.adapters.fakes.metadata import FakeMetadataResolver
from myscholar.adapters.fakes.reranker import NoopReranker
from myscholar.adapters.fakes.store import InMemoryVectorStore

__all__ = [
    "FakeChatModel",
    "FakeMetadataResolver",
    "HashEmbedder",
    "InMemoryVectorStore",
    "NoopReranker",
]
