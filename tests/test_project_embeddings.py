from __future__ import annotations

import pathlib
import sqlite3
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import semora.projection.projector as projection_module
from semora.embeddings.serialization import tensor_from_float32blob
from semora.projection.projector import (
    _embedding_matrix_from_rows,
    _fit_projector,
    _fit_torch_mlp,
    _reservoir_sample,
    _resolve_embedding_runs,
    _transform_batches,
    _validate_coordinates,
)
from semora.storage import Database, EmbeddingProjection


def test_projection_migration_adds_nullable_real_columns(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "projection.sqlite")
    try:
        db.initialize()
        columns = {
            row["name"]: row["type"]
            for row in db.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        assert columns["projection_x"] == "REAL"
        assert columns["projection_y"] == "REAL"
        assert columns["projection_z"] == "REAL"
    finally:
        db.close()


def test_projection_updates_are_stored_as_sqlite_real(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "projection.sqlite")
    try:
        db.initialize()
        _insert_minimal_embedding(db.conn)
        db.update_embedding_projections(
            [EmbeddingProjection(embedding_id="embedding-1", x=1.25, y=-2.5, z=3.75)]
        )

        row = db.conn.execute(
            """
            SELECT projection_x, projection_y, projection_z,
                   typeof(projection_x) AS x_type
            FROM embeddings
            WHERE embedding_id = 'embedding-1'
            """
        ).fetchone()
        assert tuple(row[name] for name in ("projection_x", "projection_y", "projection_z")) == (
            1.25,
            -2.5,
            3.75,
        )
        assert row["x_type"] == "real"

        db.update_embedding_projections(
            [EmbeddingProjection(embedding_id="embedding-1", x=9.0, y=8.0, z=7.0)]
        )
        overwritten = db.conn.execute(
            "SELECT projection_x, projection_y, projection_z FROM embeddings"
        ).fetchone()
        assert tuple(overwritten) == (9.0, 8.0, 7.0)
    finally:
        db.close()


def test_resolve_embedding_runs_by_model(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "projection.sqlite")
    try:
        db.initialize()
        _insert_minimal_embedding(db.conn)
        args = SimpleNamespace(model_id="model", embedding_run_ids=None)
        rows = _resolve_embedding_runs(db, args)
        assert [row["embedding_run_id"] for row in rows] == ["embeddings-1"]
    finally:
        db.close()


def test_resolve_embedding_runs_rejects_different_models(tmp_path: pathlib.Path) -> None:
    db = Database(tmp_path / "projection.sqlite")
    try:
        db.initialize()
        _insert_minimal_embedding(db.conn)
        with db.conn:
            db.conn.execute(
                """
                INSERT INTO embedding_runs (embedding_run_id, run_id, model_id)
                VALUES ('embeddings-2', 'run-1', 'other-model')
                """
            )
        args = SimpleNamespace(
            model_id=None,
            embedding_run_ids=["embeddings-1", "embeddings-2"],
        )
        with pytest.raises(ValueError, match="different models"):
            _resolve_embedding_runs(db, args)
    finally:
        db.close()


def test_embedding_matrix_rejects_mixed_dimensions() -> None:
    rows = [
        {"tensor_blob": np.array([1, 2], dtype=np.float32).tobytes()},
        {"tensor_blob": np.array([1, 2, 3], dtype=np.float32).tobytes()},
    ]
    with pytest.raises(ValueError, match="inconsistent"):
        _embedding_matrix_from_rows(rows)


def test_tensor_from_float32blob_returns_owned_torch_tensor() -> None:
    blob = np.array([1.25, -2.5, 3.75], dtype=np.float32).tobytes()
    tensor = tensor_from_float32blob(blob, expected_dimension=3)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [1.25, -2.5, 3.75]


def test_validate_coordinates_rejects_non_finite_values() -> None:
    coordinates = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or infinite"):
        _validate_coordinates(coordinates, expected_count=1)


def test_reservoir_sample_is_reproducible_and_bounded() -> None:
    first = _reservoir_sample(iter(range(100)), 10, random_state=42)
    second = _reservoir_sample(iter(range(100)), 10, random_state=42)

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_staged_projection_replaces_coordinates_only_when_complete(
    tmp_path: pathlib.Path,
) -> None:
    db = Database(tmp_path / "projection.sqlite")
    try:
        db.initialize()
        _insert_minimal_embedding(db.conn)
        db.update_embedding_projections(
            [EmbeddingProjection(embedding_id="embedding-1", x=1.0, y=2.0, z=3.0)]
        )
        db.prepare_embedding_projection_staging()
        db.stage_embedding_projections(
            [EmbeddingProjection(embedding_id="embedding-1", x=4.0, y=5.0, z=6.0)]
        )

        unchanged = db.conn.execute(
            "SELECT projection_x, projection_y, projection_z FROM embeddings"
        ).fetchone()
        assert tuple(unchanged) == (1.0, 2.0, 3.0)

        with pytest.raises(ValueError, match="Expected 2"):
            db.apply_staged_embedding_projections(expected_count=2)
        still_unchanged = db.conn.execute(
            "SELECT projection_x, projection_y, projection_z FROM embeddings"
        ).fetchone()
        assert tuple(still_unchanged) == (1.0, 2.0, 3.0)

        db.apply_staged_embedding_projections(expected_count=1)
        replaced = db.conn.execute(
            "SELECT projection_x, projection_y, projection_z FROM embeddings"
        ).fetchone()
        assert tuple(replaced) == (4.0, 5.0, 6.0)
    finally:
        db.close()


def test_parallel_umap_disables_fit_seed_and_passes_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeUmap:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, vectors):
            return self

    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUmap))
    args = SimpleNamespace(
        method="umap",
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=4,
    )

    _fit_projector(np.ones((20, 4), dtype=np.float32), args)

    assert captured["n_jobs"] == 4
    assert captured["random_state"] is None


def test_transform_batches_runs_projection_on_multiple_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "embedding_id": f"embedding-{index}",
            "tensor_blob": np.full(4, index + 1, dtype=np.float32).tobytes(),
        }
        for index in range(8)
    ]

    class FakeCursor:
        def __init__(self):
            self.offset = 0

        def fetchmany(self, size):
            batch = rows[self.offset : self.offset + size]
            self.offset += len(batch)
            return batch

    class FakeProjector:
        def __init__(self):
            self.thread_ids = set()
            self.lock = threading.Lock()

        def transform(self, matrix):
            with self.lock:
                self.thread_ids.add(threading.get_ident())
            time.sleep(0.01)
            return matrix[:, :3]

    monkeypatch.setattr(projection_module, "DEFAULT_TRANSFORM_BATCH_SIZE", 1)
    projector = FakeProjector()
    batches = list(_transform_batches(FakeCursor(), projector, worker_count=4))

    assert sum(len(batch) for batch in batches) == len(rows)
    assert len(projector.thread_ids) > 1


def test_torch_mlp_projector_trains_and_transforms_on_cpu() -> None:
    generator = np.random.default_rng(42)
    vectors = generator.normal(size=(40, 4)).astype(np.float32)
    targets = vectors[:, :3].copy()
    args = SimpleNamespace(
        mlp_device="cpu",
        random_state=42,
        mlp_learning_rate=0.01,
        mlp_epochs=2,
        mlp_batch_size=8,
        mlp_patience=2,
    )

    projector = _fit_torch_mlp(vectors, targets, args)
    coordinates = projector.transform(vectors[:5])

    assert coordinates.shape == (5, 3)
    assert np.isfinite(coordinates).all()


def _insert_minimal_embedding(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("INSERT INTO runs (run_id, run_type) VALUES ('run-1', 'test')")
        conn.execute(
            "INSERT INTO newspapers (newspaper_id, run_id, content) VALUES ('paper-1', 'run-1', '')"
        )
        conn.execute(
            """
            INSERT INTO articles (article_id, run_id, newspaper_id, content)
            VALUES ('article-1', 'run-1', 'paper-1', '')
            """
        )
        conn.execute(
            """
            INSERT INTO chunking_runs (chunking_run_id, run_id, method)
            VALUES ('chunks-1', 'run-1', 'noop')
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, run_id, article_id, chunking_run_id, chunk_index, text, method
            ) VALUES ('chunk-1', 'run-1', 'article-1', 'chunks-1', 0, '', 'noop')
            """
        )
        conn.execute(
            """
            INSERT INTO embedding_runs (embedding_run_id, run_id, model_id)
            VALUES ('embeddings-1', 'run-1', 'model')
            """
        )
        conn.execute(
            """
            INSERT INTO embeddings (embedding_id, embedding_run_id, chunk_id, tensor_blob)
            VALUES ('embedding-1', 'embeddings-1', 'chunk-1', ?)
            """,
            (np.array([1, 2, 3], dtype=np.float32).tobytes(),),
        )
