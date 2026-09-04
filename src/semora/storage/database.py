from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from semora.storage.connection import open_connection
from semora.storage.metadata import normalize_newspaper_metadata
from semora.storage.migrations import apply_migrations
from semora.storage.models import (
    Article,
    Chunk,
    ChunkingRun,
    Embedding,
    EmbeddingProjection,
    EmbeddingRun,
    EmbeddingValidationCounts,
    Log,
    Newspaper,
    Run,
)
from semora.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingRepository,
    ProjectionRepository,
    RunRepository,
)


class Database:
    """SQLite database connection and migration helper."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = self._connect()
        self.runs = RunRepository(self)
        self.documents = DocumentRepository(self)
        self.chunks = ChunkRepository(self)
        self.embeddings = EmbeddingRepository(self)
        self.projections = ProjectionRepository(self)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def initialize(self) -> None:
        """Create or migrate the database to the latest schema."""
        self._apply_migrations()

    def insert_run(self, run: Run) -> None:
        """Insert one pipeline run."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (
                    run_id,
                    run_type
                ) VALUES (?, ?)
                """,
                (
                    run.run_id,
                    run.run_type,
                ),
            )

    def insert_log(self, log: Log) -> None:
        """Insert one log entry for a pipeline run."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO logs (
                    run_id,
                    level,
                    message
                ) VALUES (?, ?, ?)
                """,
                (
                    log.run_id,
                    log.level,
                    log.message,
                ),
            )

    def log(self, run_id: str, level: str, message: str) -> None:
        """Write one log entry for a pipeline run."""
        self.insert_log(Log(run_id=run_id, level=level, message=message))

    def insert_newspaper(self, newspaper: Newspaper) -> None:
        """Insert one newspaper issue and its source metadata."""
        self.insert_newspapers([newspaper])

    def insert_newspapers(self, newspapers: list[Newspaper]) -> None:
        """Insert multiple newspaper issues in one transaction."""
        if not newspapers:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO newspapers (
                    newspaper_id,
                    run_id,
                    content,
                    metadata_json,
                    date,
                    publisher,
                    source,
                    rights,
                    title,
                    urn,
                    issue,
                    volume,
                    relative_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_newspaper_values(newspaper) for newspaper in newspapers],
            )

    def insert_article(self, article: Article) -> None:
        """Insert one article extracted from a newspaper issue."""
        metadata_json = json.dumps(article.metadata or {}, ensure_ascii=False)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO articles (
                    article_id,
                    run_id,
                    newspaper_id,
                    title,
                    content,
                    metadata_json,
                    char_start,
                    char_end,
                    line_start,
                    line_end,
                    is_valid,
                    cleaning_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _article_values(article, metadata_json),
            )

    def insert_articles(self, articles: list[Article]) -> None:
        """Insert multiple articles in one transaction."""
        if not articles:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO articles (
                    article_id,
                    run_id,
                    newspaper_id,
                    title,
                    content,
                    metadata_json,
                    char_start,
                    char_end,
                    line_start,
                    line_end,
                    is_valid,
                    cleaning_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    _article_values(
                        article,
                        json.dumps(article.metadata or {}, ensure_ascii=False),
                    )
                    for article in articles
                ],
            )

    def insert_chunking_run(self, chunking_run: ChunkingRun) -> None:
        """Insert one chunking configuration run."""
        config_json = json.dumps(chunking_run.config or {}, ensure_ascii=False)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chunking_runs (
                    chunking_run_id,
                    run_id,
                    method,
                    config_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    chunking_run.chunking_run_id,
                    chunking_run.run_id,
                    chunking_run.method,
                    config_json,
                ),
            )

    def insert_chunk(self, chunk: Chunk) -> None:
        """Insert one text chunk extracted from an article."""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    run_id,
                    article_id,
                    chunking_run_id,
                    chunk_index,
                    method,
                    text,
                    char_start,
                    char_end,
                    line_start,
                    line_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _chunk_values(chunk),
            )

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert multiple text chunks in one transaction."""
        if not chunks:
            return

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    run_id,
                    article_id,
                    chunking_run_id,
                    chunk_index,
                    method,
                    text,
                    char_start,
                    char_end,
                    line_start,
                    line_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_chunk_values(chunk) for chunk in chunks],
            )

    def insert_embedding_run(self, embedding_run: EmbeddingRun) -> None:
        """Insert one embedding configuration run."""
        config_json = json.dumps(embedding_run.config or {}, ensure_ascii=False)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embedding_runs (
                    embedding_run_id,
                    run_id,
                    model_id,
                    config_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    embedding_run.embedding_run_id,
                    embedding_run.run_id,
                    embedding_run.model_id,
                    config_json,
                ),
            )

    def insert_embeddings(self, embeddings: list[Embedding]) -> None:
        """Insert multiple vector embeddings in one transaction."""
        if not embeddings:
            return

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO embeddings (
                    embedding_id,
                    embedding_run_id,
                    chunk_id,
                    tensor_blob,
                    is_valid
                ) VALUES (
                    ?, ?, ?, ?,
                    COALESCE(
                        (
                            SELECT articles.is_valid
                            FROM chunks
                            JOIN articles ON articles.article_id = chunks.article_id
                            WHERE chunks.chunk_id = ?
                        ),
                        0
                    )
                )
                """,
                [_embedding_values(embedding) for embedding in embeddings],
            )

    def count_embeddings(self) -> int:
        """Return the total number of stored embeddings."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS embedding_count FROM embeddings"
        ).fetchone()
        return int(row["embedding_count"])

    def propagate_embedding_validation(
        self,
        *,
        batch_size: int = 10_000,
        on_batch: Callable[[int], None] | None = None,
    ) -> EmbeddingValidationCounts:
        """Copy article validation in bounded, independently committed batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        last_embedding_id: str | None = None
        while True:
            if last_embedding_id is None:
                batch = self.conn.execute(
                    """
                    SELECT
                        MIN(embedding_id) AS first_embedding_id,
                        MAX(embedding_id) AS last_embedding_id,
                        COUNT(*) AS batch_count
                    FROM (
                        SELECT embedding_id
                        FROM embeddings
                        ORDER BY embedding_id
                        LIMIT ?
                    )
                    """,
                    (batch_size,),
                ).fetchone()
            else:
                batch = self.conn.execute(
                    """
                    SELECT
                        MIN(embedding_id) AS first_embedding_id,
                        MAX(embedding_id) AS last_embedding_id,
                        COUNT(*) AS batch_count
                    FROM (
                        SELECT embedding_id
                        FROM embeddings
                        WHERE embedding_id > ?
                        ORDER BY embedding_id
                        LIMIT ?
                    )
                    """,
                    (last_embedding_id, batch_size),
                ).fetchone()

            batch_count = int(batch["batch_count"])
            if batch_count == 0:
                break

            first_embedding_id = str(batch["first_embedding_id"])
            next_embedding_id = str(batch["last_embedding_id"])
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE embeddings
                    SET is_valid = COALESCE(
                        (
                            SELECT articles.is_valid
                            FROM chunks
                            JOIN articles ON articles.article_id = chunks.article_id
                            WHERE chunks.chunk_id = embeddings.chunk_id
                        ),
                        0
                    )
                    WHERE embedding_id >= ?
                      AND embedding_id <= ?
                    """,
                    (first_embedding_id, next_embedding_id),
                )

            if on_batch is not None:
                on_batch(batch_count)
            last_embedding_id = next_embedding_id

        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                COUNT(*) FILTER (WHERE is_valid = 1) AS valid_count,
                COUNT(*) FILTER (WHERE is_valid = 0) AS invalid_count
            FROM embeddings
            """
        ).fetchone()
        return EmbeddingValidationCounts(
            total=int(row["total_count"]),
            valid=int(row["valid_count"]),
            invalid=int(row["invalid_count"]),
        )

    def optimize(self) -> None:
        """Refresh planner statistics when SQLite considers it useful."""
        self.conn.execute("PRAGMA optimize")

    def get_embedding_runs_by_model(self, model_id: str) -> list[sqlite3.Row]:
        """Return embedding runs that use an exact model id."""
        return self.conn.execute(
            """
            SELECT embedding_run_id, model_id, config_json
            FROM embedding_runs
            WHERE model_id = ?
            ORDER BY embedding_run_id
            """,
            (model_id,),
        ).fetchall()

    def get_embedding_runs_by_model_and_chunking_run(
        self,
        model_id: str,
        chunking_run_id: str,
    ) -> list[sqlite3.Row]:
        """Return runs matching exact model and configured chunking run ids."""
        return self.conn.execute(
            """
            SELECT embedding_run_id, model_id, config_json
            FROM embedding_runs
            WHERE model_id = ?
              AND json_extract(config_json, '$.chunking_run_id') = ?
            ORDER BY embedding_run_id
            """,
            (model_id, chunking_run_id),
        ).fetchall()

    def get_embedding_runs_by_ids(self, embedding_run_ids: list[str]) -> list[sqlite3.Row]:
        """Return requested embedding runs with their resolved chunking run IDs."""
        if not embedding_run_ids:
            return []
        placeholders = ", ".join("?" for _ in embedding_run_ids)
        return self.conn.execute(
            f"""
            SELECT
                er.embedding_run_id,
                er.model_id,
                er.config_json,
                COALESCE(
                    json_extract(er.config_json, '$.chunking_run_id'),
                    (
                        SELECT c.chunking_run_id
                        FROM embeddings e
                        JOIN chunks c ON c.chunk_id = e.chunk_id
                        WHERE e.embedding_run_id = er.embedding_run_id
                        LIMIT 1
                    )
                ) AS chunking_run_id
            FROM embedding_runs er
            WHERE er.embedding_run_id IN ({placeholders})
            ORDER BY er.embedding_run_id
            """,
            embedding_run_ids,
        ).fetchall()

    def get_embeddings_for_projection(
        self,
        embedding_run_ids: list[str],
    ) -> list[sqlite3.Row]:
        """Return vectors for embedding runs that will share one projection."""
        return self.iter_embeddings_for_projection(embedding_run_ids).fetchall()

    def iter_embeddings_for_projection(
        self,
        embedding_run_ids: list[str],
    ) -> sqlite3.Cursor:
        """Iterate vectors for embedding runs in a deterministic order."""
        if not embedding_run_ids:
            return self.conn.execute("SELECT NULL WHERE 0")
        placeholders = ", ".join("?" for _ in embedding_run_ids)
        return self.conn.execute(
            f"""
            SELECT
                embedding_id,
                embedding_run_id,
                tensor_blob,
                projection_x,
                projection_y,
                projection_z
            FROM embeddings
            WHERE embedding_run_id IN ({placeholders})
              AND is_valid = 1
            ORDER BY embedding_run_id, embedding_id
            """,
            embedding_run_ids,
        )

    def iter_embeddings_for_run_fast(
        self,
        embedding_run_id: str,
    ) -> sqlite3.Cursor:
        """Iterate only vector blobs for one run without joining other tables."""
        return self.conn.execute(
            """
            SELECT tensor_blob
            FROM embeddings
            WHERE embedding_run_id = ?
              AND is_valid = 1
            """,
            (embedding_run_id,),
        )

    def count_embeddings_for_projection(self, embedding_run_ids: list[str]) -> int:
        """Count embeddings in the selected projection space."""
        if not embedding_run_ids:
            return 0
        placeholders = ", ".join("?" for _ in embedding_run_ids)
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) AS embedding_count
            FROM embeddings
            WHERE embedding_run_id IN ({placeholders})
              AND is_valid = 1
            """,
            embedding_run_ids,
        ).fetchone()
        return int(row["embedding_count"])

    def prepare_embedding_projection_staging(self) -> None:
        """Create an empty temporary table for an atomic projection replacement."""
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS temp.embedding_projection_staging")
            self.conn.execute(
                """
                CREATE TEMP TABLE embedding_projection_staging (
                    embedding_id TEXT PRIMARY KEY,
                    projection_x REAL NOT NULL,
                    projection_y REAL NOT NULL,
                    projection_z REAL NOT NULL
                )
                """
            )

    def stage_embedding_projections(
        self,
        projections: list[EmbeddingProjection],
    ) -> None:
        """Stage one batch without changing existing embedding coordinates."""
        if not projections:
            return
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO embedding_projection_staging (
                    embedding_id, projection_x, projection_y, projection_z
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (projection.embedding_id, projection.x, projection.y, projection.z)
                    for projection in projections
                ],
            )

    def apply_staged_embedding_projections(self, *, expected_count: int) -> None:
        """Atomically replace coordinates after verifying the staged row count."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS projection_count FROM embedding_projection_staging"
        ).fetchone()
        actual_count = int(row["projection_count"])
        if actual_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} staged projections, found {actual_count}"
            )

        with self.conn:
            self.conn.execute(
                """
                UPDATE embeddings
                SET
                    projection_x = staging.projection_x,
                    projection_y = staging.projection_y,
                    projection_z = staging.projection_z
                FROM embedding_projection_staging AS staging
                WHERE embeddings.embedding_id = staging.embedding_id
                """
            )
            self.conn.execute("DROP TABLE embedding_projection_staging")

    def update_embedding_projections(
        self,
        projections: list[EmbeddingProjection],
    ) -> None:
        """Update 3D coordinates atomically for a collection of embeddings."""
        if not projections:
            return

        with self.conn:
            self.conn.executemany(
                """
                UPDATE embeddings
                SET projection_x = ?, projection_y = ?, projection_z = ?
                WHERE embedding_id = ?
                """,
                [
                    (projection.x, projection.y, projection.z, projection.embedding_id)
                    for projection in projections
                ],
            )

    def get_embedding_runs(self) -> list[sqlite3.Row]:
        """Return embedding runs, newest first."""
        return self.conn.execute(
            """
            WITH runs AS (
                SELECT
                    er.*,
                    COALESCE(
                        json_extract(er.config_json, '$.chunking_run_id'),
                        (
                            SELECT c.chunking_run_id
                            FROM embeddings e
                            JOIN chunks c ON c.chunk_id = e.chunk_id
                            WHERE e.embedding_run_id = er.embedding_run_id
                              AND e.is_valid = 1
                            LIMIT 1
                        )
                    ) AS resolved_chunking_run_id
                FROM embedding_runs er
            )
            SELECT
                runs.embedding_run_id,
                runs.model_id,
                runs.created_at,
                (
                    SELECT COUNT(*)
                    FROM embeddings e
                    WHERE e.embedding_run_id = runs.embedding_run_id
                      AND e.is_valid = 1
                ) AS embedding_count,
                chunking_runs.method AS chunking_methods,
                chunking_runs.config_json AS chunking_config_json,
                runs.resolved_chunking_run_id AS chunking_run_ids
            FROM runs
            LEFT JOIN chunking_runs
                ON chunking_runs.chunking_run_id = runs.resolved_chunking_run_id
            ORDER BY runs.created_at DESC, runs.embedding_run_id DESC
            """
        ).fetchall()

    def get_non_noop_embedding_run_ids(self) -> list[str]:
        """Return runs whose configured chunking method is not noop."""
        rows = self.conn.execute(
            """
            SELECT er.embedding_run_id
            FROM embedding_runs er
            WHERE EXISTS (
                  SELECT 1
                  FROM embeddings e
                  JOIN chunks c ON c.chunk_id = e.chunk_id
                  JOIN chunking_runs cr
                    ON cr.chunking_run_id = c.chunking_run_id
                  WHERE e.embedding_run_id = er.embedding_run_id
                    AND e.is_valid = 1
                    AND cr.method != 'noop'
                  LIMIT 1
              )
            ORDER BY er.created_at DESC, er.embedding_run_id DESC
            """
        ).fetchall()
        return [row["embedding_run_id"] for row in rows]

    def iter_embeddings_for_run(
        self,
        embedding_run_id: str,
        *,
        chunking_run_id: str | None = None,
    ) -> sqlite3.Cursor:
        """Iterate embeddings ordered for article-by-article processing."""
        params: list[object] = [embedding_run_id]
        sql = """
            SELECT
                chunks.article_id,
                chunks.chunk_id,
                chunks.chunk_index,
                chunks.chunking_run_id,
                chunking_runs.method AS chunking_method,
                chunking_runs.config_json AS chunking_config_json,
                embeddings.tensor_blob
            FROM embeddings
            JOIN chunks ON chunks.chunk_id = embeddings.chunk_id
            JOIN chunking_runs
                ON chunking_runs.chunking_run_id = chunks.chunking_run_id
            WHERE embeddings.embedding_run_id = ?
              AND embeddings.is_valid = 1
        """
        if chunking_run_id:
            sql += " AND chunks.chunking_run_id = ?"
            params.append(chunking_run_id)
        sql += " ORDER BY chunks.chunking_run_id, chunks.article_id, chunks.chunk_index"
        return self.conn.execute(sql, params)

    def get_embeddings_for_run(
        self,
        embedding_run_id: str,
        *,
        chunking_run_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """Return embeddings ordered by article and chunk index."""
        return self.iter_embeddings_for_run(
            embedding_run_id,
            chunking_run_id=chunking_run_id,
        ).fetchall()

    def get_shared_eligible_embedding_article_ids(
        self,
        embedding_run_ids: list[str],
        *,
        min_chunks: int,
    ) -> list[str]:
        """Return articles eligible for every run's distinct chunking strategy."""
        if not embedding_run_ids:
            raise ValueError("At least one embedding run ID is required.")
        if min_chunks < 1:
            raise ValueError("Minimum chunks must be at least 1.")
        unique_run_ids = list(dict.fromkeys(embedding_run_ids))
        run_rows = self.get_embedding_runs_by_ids(unique_run_ids)
        known_run_ids = {str(row["embedding_run_id"]) for row in run_rows}
        unknown_run_ids = sorted(set(unique_run_ids) - known_run_ids)
        if unknown_run_ids:
            raise ValueError(
                "Unknown embedding run ID(s): " + ", ".join(unknown_run_ids)
            )
        chunking_run_ids = [
            str(row["chunking_run_id"])
            for row in run_rows
            if row["chunking_run_id"] is not None
        ]
        if len(chunking_run_ids) != len(run_rows):
            raise ValueError("Every embedding run must resolve to a chunking run.")
        return self.get_shared_eligible_chunk_article_ids(
            chunking_run_ids,
            min_chunks=min_chunks,
        )

    def get_shared_eligible_chunk_article_ids(
        self,
        chunking_run_ids: list[str],
        *,
        min_chunks: int,
    ) -> list[str]:
        """Intersect valid articles meeting a chunk count for every strategy."""
        if not chunking_run_ids:
            raise ValueError("At least one chunking run ID is required.")
        if min_chunks < 1:
            raise ValueError("Minimum chunks must be at least 1.")

        shared_article_ids: set[str] | None = None
        for chunking_run_id in dict.fromkeys(chunking_run_ids):
            rows = self.conn.execute(
                """
                SELECT grouped.article_id
                FROM (
                    SELECT article_id
                    FROM chunks INDEXED BY idx_chunks_chunking_run_article
                    WHERE chunking_run_id = ?
                    GROUP BY article_id
                    HAVING COUNT(*) >= ?
                ) AS grouped
                JOIN articles ON articles.article_id = grouped.article_id
                WHERE articles.is_valid = 1
                ORDER BY grouped.article_id
                """,
                (chunking_run_id, min_chunks),
            ).fetchall()
            eligible = {str(row["article_id"]) for row in rows}
            shared_article_ids = (
                eligible
                if shared_article_ids is None
                else shared_article_ids & eligible
            )
            if not shared_article_ids:
                return []
        return sorted(shared_article_ids or ())

    def get_full_article_embeddings(
        self,
        *,
        embedding_run_id: str,
    ) -> list[sqlite3.Row]:
        """Return full-article embeddings stored as noop chunks."""
        return self.conn.execute(
            """
            SELECT
                chunks.article_id,
                chunks.chunk_id,
                chunks.chunking_run_id,
                embeddings.tensor_blob
            FROM embeddings
            JOIN chunks ON chunks.chunk_id = embeddings.chunk_id
            JOIN chunking_runs
                ON chunking_runs.chunking_run_id = chunks.chunking_run_id
            WHERE embeddings.embedding_run_id = ?
              AND embeddings.is_valid = 1
              AND chunking_runs.method = 'noop'
            ORDER BY chunks.article_id
            """,
            (embedding_run_id,),
        ).fetchall()

    def get_chunk_embeddings_for_run(
        self,
        *,
        embedding_run_id: str,
        chunking_run_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """Return non-noop chunk embeddings for representation metrics."""
        params: list[object] = [embedding_run_id]
        sql = """
            SELECT
                chunks.article_id,
                chunks.chunk_id,
                chunks.chunk_index,
                chunks.chunking_run_id,
                chunking_runs.method AS chunking_method,
                chunking_runs.config_json AS chunking_config_json,
                length(chunks.text) AS chunk_length,
                embeddings.tensor_blob
            FROM embeddings
            JOIN chunks ON chunks.chunk_id = embeddings.chunk_id
            JOIN chunking_runs
                ON chunking_runs.chunking_run_id = chunks.chunking_run_id
            WHERE embeddings.embedding_run_id = ?
              AND embeddings.is_valid = 1
              AND chunking_runs.method != 'noop'
        """
        if chunking_run_id:
            sql += " AND chunks.chunking_run_id = ?"
            params.append(chunking_run_id)
        sql += " ORDER BY chunks.chunking_run_id, chunks.article_id, chunks.chunk_index"
        return self.conn.execute(sql, params).fetchall()

    def get_newspapers(self) -> list[sqlite3.Row]:
        """Return stored newspaper rows ordered by id."""
        return self.conn.execute(
            """
            SELECT
                newspaper_id,
                content,
                metadata_json,
                date,
                publisher,
                source,
                rights,
                title,
                urn,
                issue,
                volume
            FROM newspapers
            ORDER BY newspaper_id
            """
        ).fetchall()

    def get_chunks(
        self,
        *,
        chunking_run_id: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """Return stored chunk rows ordered by article and chunk index."""
        where = []
        params: list[object] = []

        if chunking_run_id:
            where.append("chunking_run_id = ?")
            params.append(chunking_run_id)

        sql = """
            SELECT chunk_id, article_id, chunking_run_id, chunk_index, text
            FROM chunks
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY article_id, chunk_index"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        return self.conn.execute(sql, params).fetchall()

    def get_chunking_runs(self) -> list[sqlite3.Row]:
        """Return all chunking runs ordered by ID."""
        return self.conn.execute(
            """
            SELECT chunking_run_id, method, config_json
            FROM chunking_runs
            ORDER BY chunking_run_id
            """
        ).fetchall()

    def get_chunking_runs_by_ids(
        self,
        chunking_run_ids: list[str],
    ) -> list[sqlite3.Row]:
        """Return metadata for the requested chunking runs."""
        if not chunking_run_ids:
            return []
        placeholders = ", ".join("?" for _ in chunking_run_ids)
        return self.conn.execute(
            f"""
            SELECT chunking_run_id, method, config_json
            FROM chunking_runs
            WHERE chunking_run_id IN ({placeholders})
            ORDER BY chunking_run_id
            """,
            chunking_run_ids,
        ).fetchall()

    def iter_chunk_texts_for_runs(
        self,
        chunking_run_ids: list[str],
    ) -> sqlite3.Cursor:
        """Iterate chunk texts for several chunking runs without materializing rows."""
        if not chunking_run_ids:
            raise ValueError("At least one chunking run ID is required.")
        placeholders = ", ".join("?" for _ in chunking_run_ids)
        return self.conn.execute(
            f"""
            SELECT chunking_run_id, text
            FROM chunks
            WHERE chunking_run_id IN ({placeholders})
            ORDER BY chunking_run_id, article_id, chunk_index
            """,
            chunking_run_ids,
        )

    def get_unembedded_chunks(
        self,
        *,
        embedding_run_id: str,
        chunking_run_id: str | None = None,
        max_characters: int | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """Return chunks without an embedding for the given embedding run."""
        params: list[object] = [embedding_run_id]
        where = ["e.embedding_id IS NULL"]

        if chunking_run_id:
            where.append("c.chunking_run_id = ?")
            params.append(chunking_run_id)

        if max_characters is not None:
            where.append("length(c.text) <= ?")
            params.append(int(max_characters))

        sql = """
            SELECT c.chunk_id, c.article_id, c.chunking_run_id, c.chunk_index, c.text
            FROM chunks c
            LEFT JOIN embeddings e
                ON e.embedding_run_id = ?
                AND e.chunk_id = c.chunk_id
        """
        sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY c.article_id, c.chunk_index"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        return self.conn.execute(sql, params).fetchall()

    def get_oversized_unembedded_chunks(
        self,
        *,
        embedding_run_id: str,
        max_characters: int,
        chunking_run_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """Return unembedded chunks that exceed the character limit."""
        params: list[object] = [embedding_run_id, int(max_characters)]
        where = ["e.embedding_id IS NULL", "length(c.text) > ?"]

        if chunking_run_id:
            where.append("c.chunking_run_id = ?")
            params.append(chunking_run_id)

        sql = """
            SELECT c.chunk_id, c.article_id, length(c.text) AS character_count
            FROM chunks c
            LEFT JOIN embeddings e
                ON e.embedding_run_id = ?
                AND e.chunk_id = c.chunk_id
        """
        sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY c.article_id, c.chunk_index"
        return self.conn.execute(sql, params).fetchall()

    def get_articles(self) -> list[sqlite3.Row]:
        """Return stored article rows ordered by id."""
        return self.conn.execute(
            """
            SELECT article_id, title, content
            FROM articles
            ORDER BY article_id
            """
        ).fetchall()

    def get_valid_articles(self) -> list[sqlite3.Row]:
        """Return stored article rows ordered by id."""
        return self.conn.execute(
            """
            SELECT article_id, title, content
            FROM articles
            WHERE is_valid = 1
            ORDER BY article_id
            """
        ).fetchall()

    def count_articles_for_analysis(self, *, include_invalid: bool) -> int:
        """Count non-empty articles selected for read-only text analysis."""
        where = "LENGTH(content) > 0"
        if not include_invalid:
            where += " AND is_valid = 1"
        row = self.conn.execute(
            f"SELECT COUNT(*) AS article_count FROM articles WHERE {where}"
        ).fetchone()
        return int(row["article_count"])

    def iter_articles_for_analysis(
        self,
        *,
        include_invalid: bool,
        fetch_size: int = 1_000,
    ):
        """Yield non-empty article rows without materializing the full corpus."""
        if fetch_size < 1:
            raise ValueError("Article fetch size must be at least 1.")
        where = "LENGTH(content) > 0"
        if not include_invalid:
            where += " AND is_valid = 1"
        cursor = self.conn.execute(
            f"""
            SELECT article_id, title, content, is_valid, cleaning_reason
            FROM articles
            WHERE {where}
            """
        )
        while batch := cursor.fetchmany(fetch_size):
            yield from batch

    def update_article_validation(
        self,
        article_id: str,
        *,
        is_valid: bool,
        reason: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE articles
                SET is_valid = ?, cleaning_reason = ?
                WHERE article_id = ?
                """,
                (
                    int(is_valid),
                    reason,
                    article_id,
                ),
            )

    def invalidate_articles(self, rows: list[tuple[str, str]]) -> None:
        """Mark articles invalid in one transaction using (reason, article_id) rows."""
        if not rows:
            return
        with self.conn:
            self.conn.executemany(
                """
                UPDATE articles
                SET is_valid = 0, cleaning_reason = ?
                WHERE article_id = ? AND is_valid = 1
                """,
                rows,
            )

    def get_duplicate_article_ids(self) -> list[str]:
        """Return non-canonical exact-content duplicates needing invalidation."""
        rows = self.conn.execute(
            """
            SELECT article_id
            FROM (
                SELECT
                    article_id,
                    is_valid,
                    cleaning_reason,
                    ROW_NUMBER() OVER (
                        PARTITION BY content
                        ORDER BY article_id
                    ) AS duplicate_rank
                FROM articles
                WHERE content <> ''
            ) AS ranked
            WHERE duplicate_rank > 1
              AND (
                  is_valid IS NOT 0
                  OR cleaning_reason IS NOT 'duplicate'
              )
            ORDER BY article_id
            """
        ).fetchall()
        return [str(row["article_id"]) for row in rows]

    def mark_articles_duplicate(self, article_ids: list[str]) -> int:
        """Mark exact-content duplicates invalid in one transaction."""
        if not article_ids:
            return 0
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(
                """
                UPDATE articles
                SET is_valid = 0, cleaning_reason = 'duplicate'
                WHERE article_id = ?
                  AND (
                      is_valid IS NOT 0
                      OR cleaning_reason IS NOT 'duplicate'
                  )
                """,
                [(article_id,) for article_id in article_ids],
            )
        return self.conn.total_changes - before

    def _apply_migrations(self) -> None:
        """Apply numbered SQL migrations that have not been applied yet."""
        apply_migrations(self.conn)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with project-specific settings applied."""
        return open_connection(self.path)


def _chunk_values(chunk: Chunk) -> tuple:
    return (
        chunk.chunk_id,
        chunk.run_id,
        chunk.article_id,
        chunk.chunking_run_id,
        chunk.chunk_index,
        chunk.method,
        chunk.text,
        chunk.char_start,
        chunk.char_end,
        chunk.line_start,
        chunk.line_end,
    )


def _newspaper_values(newspaper: Newspaper) -> tuple:
    normalized = normalize_newspaper_metadata(newspaper.metadata)
    return (
        newspaper.newspaper_id,
        newspaper.run_id,
        newspaper.content,
        json.dumps(newspaper.metadata or {}, ensure_ascii=False),
        normalized["date"],
        normalized["publisher"],
        normalized["source"],
        normalized["rights"],
        normalized["title"],
        normalized["urn"],
        normalized["issue"],
        normalized["volume"],
        newspaper.relative_path,
    )


def _article_values(article: Article, metadata_json: str) -> tuple:
    return (
        article.article_id,
        article.run_id,
        article.newspaper_id,
        article.title,
        article.content,
        metadata_json,
        article.char_start,
        article.char_end,
        article.line_start,
        article.line_end,
        article.is_valid,
        article.cleaning_reason,
    )


def _embedding_values(embedding: Embedding) -> tuple:
    return (
        embedding.embedding_id,
        embedding.embedding_run_id,
        embedding.chunk_id,
        embedding.tensor_blob,
        embedding.chunk_id,
    )

