"""Records accepted and returned by Semora's storage layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Run:
    run_id: str
    run_type: str


@dataclass
class Log:
    run_id: str
    level: str
    message: str


@dataclass
class Newspaper:
    newspaper_id: str
    run_id: str
    content: str
    metadata: dict | None = None


@dataclass
class Article:
    article_id: str
    run_id: str
    newspaper_id: str
    title: str | None
    content: str
    metadata: dict | None = None


@dataclass
class ChunkingRun:
    chunking_run_id: str
    run_id: str
    method: str
    config: dict | None = None


@dataclass
class Chunk:
    chunk_id: str
    run_id: str
    article_id: str
    chunking_run_id: str
    chunk_index: int
    method: str
    text: str


@dataclass
class EmbeddingRun:
    embedding_run_id: str
    run_id: str
    model_id: str
    config: dict | None = None


@dataclass
class Embedding:
    embedding_id: str
    embedding_run_id: str
    chunk_id: str | None
    tensor_blob: bytes


@dataclass
class EmbeddingProjection:
    embedding_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class EmbeddingValidationCounts:
    total: int
    valid: int
    invalid: int
