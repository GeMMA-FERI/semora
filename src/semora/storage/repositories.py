"""Domain-oriented access to Semora's SQLite operations."""

from __future__ import annotations

import builtins
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from semora.storage.models import (
    Article,
    Chunk,
    ChunkingRun,
    Embedding,
    EmbeddingProjection,
    EmbeddingRun,
    EmbeddingValidationCounts,
    Newspaper,
    Run,
)

if TYPE_CHECKING:
    from semora.storage.database import Database


@dataclass(frozen=True)
class RunRepository:
    database: Database

    def add(self, run: Run) -> None:
        self.database.insert_run(run)

    def log(self, run_id: str, level: str, message: str) -> None:
        self.database.log(run_id, level, message)


@dataclass(frozen=True)
class DocumentRepository:
    database: Database

    def add_newspaper(self, newspaper: Newspaper) -> None:
        self.database.insert_newspaper(newspaper)

    def add_article(self, article: Article) -> None:
        self.database.insert_article(article)

    def list_newspapers(self) -> list[sqlite3.Row]:
        return self.database.get_newspapers()

    def list_articles(self, *, valid_only: bool = False) -> list[sqlite3.Row]:
        return self.database.get_valid_articles() if valid_only else self.database.get_articles()

    def set_article_validity(
        self,
        article_id: str,
        *,
        is_valid: bool,
        reason: str | None = None,
    ) -> None:
        self.database.update_article_validation(
            article_id,
            is_valid=is_valid,
            reason=reason,
        )


@dataclass(frozen=True)
class ChunkRepository:
    database: Database

    def add_run(self, run: ChunkingRun) -> None:
        self.database.insert_chunking_run(run)

    def add(self, chunk: Chunk) -> None:
        self.database.insert_chunk(chunk)

    def add_many(self, chunks: builtins.list[Chunk]) -> None:
        self.database.insert_chunks(chunks)

    def list(
        self,
        *,
        chunking_run_id: str | None = None,
        limit: int | None = None,
    ) -> builtins.list[sqlite3.Row]:
        return self.database.get_chunks(
            chunking_run_id=chunking_run_id,
            limit=limit,
        )

    def list_runs(self) -> builtins.list[sqlite3.Row]:
        return self.database.get_chunking_runs()


@dataclass(frozen=True)
class EmbeddingRepository:
    database: Database

    def add_run(self, run: EmbeddingRun) -> None:
        self.database.insert_embedding_run(run)

    def add_many(self, embeddings: list[Embedding]) -> None:
        self.database.insert_embeddings(embeddings)

    def count(self) -> int:
        return self.database.count_embeddings()

    def list_runs(self) -> list[sqlite3.Row]:
        return self.database.get_embedding_runs()

    def propagate_validation(
        self,
        *,
        batch_size: int = 10_000,
        on_batch: Callable[[int], None] | None = None,
    ) -> EmbeddingValidationCounts:
        return self.database.propagate_embedding_validation(
            batch_size=batch_size,
            on_batch=on_batch,
        )


@dataclass(frozen=True)
class ProjectionRepository:
    database: Database

    def count(self, embedding_run_ids: list[str]) -> int:
        return self.database.count_embeddings_for_projection(embedding_run_ids)

    def iter_embeddings(self, embedding_run_ids: list[str]) -> sqlite3.Cursor:
        return self.database.iter_embeddings_for_projection(embedding_run_ids)

    def update(self, projections: list[EmbeddingProjection]) -> None:
        self.database.update_embedding_projections(projections)
