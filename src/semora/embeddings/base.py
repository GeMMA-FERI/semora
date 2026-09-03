"""Shared embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseEmbedder(ABC):
    """Interface implemented by all local embedding backends."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def load(self) -> BaseEmbedder:
        """Load backend resources and return this embedder."""

    @abstractmethod
    def embed_query(self, text: str) -> torch.Tensor:
        """Embed a query."""

    @abstractmethod
    def embed_sentence(self, text: str) -> torch.Tensor:
        """Embed one sentence or document."""

    def embed_document(self, text: str) -> torch.Tensor:
        return self.embed_sentence(text)

    def embed_documents(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0))
        return torch.stack([self.embed_document(text) for text in texts], dim=0)
