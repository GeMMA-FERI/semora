"""Embedding interfaces, backend selection, batching, and serialization."""

from semora.embeddings.base import BaseEmbedder
from semora.embeddings.registry import get_embedder
from semora.embeddings.serialization import (
    build_embedding,
    build_embedding_id,
    tensor_from_float32blob,
)

__all__ = [
    "BaseEmbedder",
    "build_embedding",
    "build_embedding_id",
    "get_embedder",
    "tensor_from_float32blob",
]
