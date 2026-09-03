"""Embedding backend selection."""

from __future__ import annotations

import os
from typing import Any

from semora.embeddings.backends import (
    CausalEmbedder,
    FlagEmbedder,
    HFAutoEmbedder,
    NvidiaLlamaEmbedder,
    SentenceEmbedder,
)
from semora.embeddings.base import BaseEmbedder


def get_embedder(
    model_id: str,
    kind: str | None = None,
    transformer_kwargs: dict[str, Any] | None = None,
) -> BaseEmbedder:
    """Create an embedder for a model identifier and optional backend kind."""
    if kind == "causal":
        return CausalEmbedder(model_id)
    if model_id == "nvidia/llama-embed-nemotron-8b":
        return NvidiaLlamaEmbedder(model_id)
    if model_id.startswith("BAAI/"):
        return FlagEmbedder(model_id)

    peft_config = os.path.join(model_id, "adapter_config.json")
    if os.path.isdir(model_id) and os.path.exists(peft_config):
        return HFAutoEmbedder(model_id)

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HFAutoEmbedder(model_id)
    return SentenceEmbedder(model_id, transformer_kwargs=transformer_kwargs)
