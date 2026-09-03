from __future__ import annotations

import hashlib
import json

import torch

from semora.storage import Embedding


def tensor_from_float32blob(
    blob: bytes,
    *,
    expected_dimension: int | None = None,
) -> torch.Tensor:
    """Decode contiguous float32 bytes into an owned tensor."""
    element_size = torch.empty((), dtype=torch.float32).element_size()
    if len(blob) % element_size:
        raise ValueError(f"Embedding blob size is not divisible by {element_size} bytes.")

    tensor = torch.frombuffer(bytearray(blob), dtype=torch.float32).clone()
    if expected_dimension is not None and tensor.numel() != expected_dimension:
        raise ValueError(
            f"Expected embedding dimension {expected_dimension}, "
            f"but decoded {tensor.numel()} values."
        )
    return tensor


def build_embedding(
    *,
    embedding_run_id: str,
    chunk_id: str,
    vector,
) -> Embedding:
    """Build the canonical float32 embedding record for one chunk."""
    tensor = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().contiguous().reshape(-1)
    return Embedding(
        embedding_id=build_embedding_id(
            embedding_run_id=embedding_run_id,
            chunk_id=chunk_id,
        ),
        embedding_run_id=embedding_run_id,
        chunk_id=chunk_id,
        tensor_blob=tensor.numpy().tobytes(),
    )


def build_embedding_id(*, embedding_run_id: str, chunk_id: str) -> str:
    value = {
        "embedding_run_id": embedding_run_id,
        "chunk_id": chunk_id,
    }
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    return f"embedding_{digest[:24]}"
