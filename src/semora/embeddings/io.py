"""Read legacy file-based embedding batches."""

from __future__ import annotations

import os

import torch
from tqdm import tqdm


def load_embeddings(path: str) -> tuple[list[str], torch.Tensor]:
    """Load path-keyed Torch batches into one ordered tensor."""
    files = (
        sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".pt"))
        if os.path.isdir(path)
        else [path]
    )
    paths: list[str] = []
    embeddings: list[torch.Tensor] = []
    for file_path in tqdm(files, desc="Loading embeddings"):
        data = torch.load(file_path, map_location="cpu")
        for source_path, embedding in data.items():
            paths.append(source_path)
            embeddings.append(embedding.detach().float())
    if not embeddings:
        return paths, torch.empty((0, 0))
    return paths, torch.stack(embeddings, dim=0)
