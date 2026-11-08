"""Interfaces the application depends on.

Application code imports from here, never from ``adapters``; only ``container.py``
chooses the concrete implementations.
"""

from myscholar.ports.embedding import Embedder
from myscholar.ports.llm import ChatModel, Message, Role
from myscholar.ports.store import VectorStore

__all__ = ["ChatModel", "Embedder", "Message", "Role", "VectorStore"]
