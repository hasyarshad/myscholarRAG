"""Interfaces the application depends on.

Application code imports from here and never from ``adapters``. Only
``container.py`` knows which concrete implementation is in use — that is what
makes swapping local Ollama for a hosted API a configuration change.

See docs/learning/00-architecture-and-ports.md.
"""

from myscholar.ports.embedding import Embedder
from myscholar.ports.llm import ChatModel, Message, Role
from myscholar.ports.reranker import Reranker
from myscholar.ports.store import VectorStore

__all__ = ["ChatModel", "Embedder", "Message", "Reranker", "Role", "VectorStore"]
