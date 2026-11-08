"""Working fake implementations of the ports, for tests.

They need no GPU, network or database.
"""

from myscholar.adapters.fakes.chat import FakeChatModel
from myscholar.adapters.fakes.embedding import HashEmbedder
from myscholar.adapters.fakes.metadata import FakeMetadataResolver
from myscholar.adapters.fakes.store import InMemoryVectorStore

__all__ = [
    "FakeChatModel",
    "FakeMetadataResolver",
    "HashEmbedder",
    "InMemoryVectorStore",
]
