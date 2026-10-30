from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.ner.base import Entity, Prediction
from semora.pipeline import ner
from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.ner import index_batch_ner, model_identity


class Model:
    def __init__(self, *, fail_at=None, bad_span=False, metadata=None):
        self.metadata = copy.deepcopy(metadata or {
            "adapter": "classla", "package": "default", "processors": "tokenize,ner",
            "classla_version": "2.2.2", "model_sha256": "a" * 64, "source_sha256": "b" * 64,
            "ner_config": {"model_path": "C:/old/resources/model.pt", "batch_size": 128, "char": True},
        })
        self.calls = 0
        self.texts = []
        self.fail_at = fail_at
        self.bad_span = bad_span
        self.closed = False

    def prepare(self, texts):
        return texts

    def predict_prepared(self, texts, _batch_size):
        self.calls += 1
        self.texts.extend(texts)
        if self.calls == self.fail_at:
            raise RuntimeError("simulated NER failure")
        return Prediction([
            [Entity(0, 3, "PER", "bad" if self.bad_span else "Ana")] if text.startswith("Ana") else []
            for text in texts
        ])

    def close(self):
        self.closed = True


def ingest(root: Path, batch: str, entries: dict[str, str]) -> Path:
    source = root / "inputs"
    records = []
    for name, content in entries.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# Article\n" + content + "\n").encode("utf-8"))
        metadata = path.with_name(path.stem + "_metadata.json")
        metadata.write_text(json.dumps({"Record": {"source": "Paper"}}), encoding="utf-8")
        records.append(BatchIssue(
            relative_path=name, markdown_path=name, metadata_path=metadata.relative_to(source).as_posix(),
            markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        ))
    database = root / "indexes" / "semora.sqlite"
    run_batch(BatchManifest(batch, tuple(records)), source, database, stages=("newspapers", "articles"))
    return database


def state(database: Path) -> tuple[list[tuple], list[tuple]]:
    with closing(sqlite3.connect(database)) as connection:
        return (
            connection.execute("SELECT * FROM article_ner_state ORDER BY article_id").fetchall(),
            connection.execute("SELECT * FROM article_entities ORDER BY article_id,entity_index").fetchall(),
        )


def test_two_batches_and_retry_are_scoped_and_include_zero_entities(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak.", "Paper/b.md": "No names here."})
    ingest(tmp_path, "unselected", {"Paper/unselected.md": "Ana outside this batch."})
    model = Model()

    first = index_batch_ner(database, "first", model=model, batch_articles=1, pipeline_depth=2)
    before = state(database)
    repeated = index_batch_ner(database, "first", model=model)

    assert first["completed_articles"] == first["added_articles"] == 2
    assert first["added_entities"] == 1 and first["complete"]
    assert repeated["added_articles"] == 0 and repeated["skipped_articles"] == 2
    assert state(database) == before
    assert len(before[0]) == 2 and sorted(row[3] for row in before[0]) == [0, 1]
    assert model.calls == 2 and not model.closed
    assert all("outside" not in text for text in model.texts)
    ingest(tmp_path, "second", {"Paper/c.md": "Ana arrives."})
    second = index_batch_ner(database, "second", model=model, batch_articles=10)
    assert second["added_articles"] == 1
    assert set(before[0]).issubset(state(database)[0])
    assert not second["retrieval_ready"] and not second["publication_ready"]


def test_metadata_only_overlap_does_not_repeat_ner(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    index_batch_ner(database, "first", model=Model())
    ingest(tmp_path, "overlap", {"Paper/a.md": "Ana Novak."})
    model = Model(fail_at=1)

    report = index_batch_ner(database, "overlap", model=model)

    assert report["added_articles"] == 0 and model.calls == 0


def test_failure_keeps_only_committed_checkpoints_and_resumes(tmp_path):
    database = ingest(tmp_path, "first", {
        "Paper/a.md": "Ana first.", "Paper/b.md": "Ana second.", "Paper/c.md": "Ana third.",
    })
    with pytest.raises(RuntimeError, match="simulated NER failure"):
        index_batch_ner(database, "first", model=Model(fail_at=2), batch_articles=1)
    assert len(state(database)[0]) == len(state(database)[1]) == 1
    model = Model()

    report = index_batch_ner(database, "first", model=model, batch_articles=1)

    assert report["added_articles"] == model.calls == 2
    assert report["skipped_articles"] == 1 and report["pending_articles"] == 0
    assert len(state(database)[0]) == len(state(database)[1]) == 3


def test_invalid_entity_span_commits_neither_mentions_nor_completion(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})

    with pytest.raises(ValueError, match="does not match source"):
        index_batch_ner(database, "first", model=Model(bad_span=True))

    assert state(database) == ([], [])


def test_relocation_and_runtime_tuning_do_not_change_model_identity(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    original = Model()
    index_batch_ner(database, "first", model=original)
    ingest(tmp_path, "second", {"Paper/b.md": "Ana arrives."})
    moved = tmp_path / "moved.sqlite"
    database.rename(moved)
    metadata = copy.deepcopy(original.metadata)
    metadata["ner_config"].update({
        "model_path": "D:/relocated/model.pt", "batch_size": 16, "pipeline_depth": 4, "decoder": "numba",
    })
    metadata["tokenizer_workers"] = 4
    model = Model(metadata=metadata)

    report = index_batch_ner(
        moved, "second", model=model, device="cuda", batch_articles=1, batch_size=8,
        tokenizer_workers=4, ner_pipeline_depth=2, ner_decoder="numba",
    )

    assert model_identity(metadata) == model_identity(original.metadata)
    assert report["added_articles"] == model.calls == 1


@pytest.mark.parametrize("field", ["model_sha256", "source_sha256", "classla_version"])
def test_model_or_code_change_is_rejected_before_any_write(tmp_path, field):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    original = Model()
    index_batch_ner(database, "first", model=original)
    ingest(tmp_path, "second", {"Paper/b.md": "Ana arrives."})
    changed = copy.deepcopy(original.metadata)
    changed[field] = "c" * 64 if field.endswith("sha256") else "3.0"
    before = database.read_bytes()
    model = Model(metadata=changed)

    with pytest.raises(ValueError, match="model identity changed"):
        index_batch_ner(database, "second", model=model)

    assert database.read_bytes() == before and model.calls == 0


def test_models_load_lazily_and_owned_adapter_closes(tmp_path, monkeypatch):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    model = Model()
    calls = []

    def load(name, **options):
        calls.append((name, options))
        return model

    monkeypatch.setattr(ner, "load_adapter", load)
    index_batch_ner(database, "first")
    assert model.closed and len(calls) == 1
    assert calls[0][1]["sentence_order"] == "input"
    monkeypatch.setattr(ner, "load_adapter", lambda *_a, **_k: pytest.fail("completed work loaded a model"))
    report = index_batch_ner(database, "first", batch_articles=99, device="cuda", ner_decoder="numba")
    assert report["added_articles"] == 0


def test_historical_outputs_require_matching_explicit_adoption(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    model = Model()
    index_batch_ner(database, "first", model=model)
    # Reproduce a historical run's recorded metadata without pipeline ownership.
    legacy = {"model": model.metadata, "device": "cuda", "batch_articles": 32, "batch_size": 16,
              "ner_decoder": "numpy", "sentence_order": "input"}
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE ner_runs SET config_json=?", (json.dumps(legacy),))
    ingest(tmp_path, "second", {"Paper/b.md": "Ana arrives."})
    before = state(database)
    with pytest.raises(ValueError, match="explicit audited adoption"):
        index_batch_ner(database, "second", model=model)
    assert state(database) == before
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_ner_adoptions (
                ner_run_id TEXT PRIMARY KEY REFERENCES ner_runs(ner_run_id),
                evidence_json TEXT NOT NULL,model_identity_json TEXT NOT NULL,
                adopted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) "
                           "VALUES ('classla-sl','{}',?)", (json.dumps(model_identity(model.metadata)),))

    report = index_batch_ner(database, "second", model=Model())

    assert report["added_articles"] == 1


def test_changed_article_during_prediction_is_rejected_atomically(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})

    class ChangesSource(Model):
        def predict_prepared(self, texts, batch_size):
            prediction = super().predict_prepared(texts, batch_size)
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("UPDATE articles SET content='Changed source'")
            return prediction

    with pytest.raises(RuntimeError, match="Article changed during NER"):
        index_batch_ner(database, "first", model=ChangesSource())

    assert state(database) == ([], [])


def test_missing_batch_does_not_load_models_or_create_database(tmp_path, monkeypatch):
    path = tmp_path / "missing" / "archive.sqlite"
    monkeypatch.setattr(ner, "load_adapter", lambda *_a, **_k: pytest.fail("model loaded"))
    with pytest.raises(FileNotFoundError):
        index_batch_ner(path, "missing")
    assert not path.parent.exists()


def test_empty_article_batch_completes_without_models(tmp_path, monkeypatch):
    database = ingest(tmp_path, "empty", {"Paper/a.md": ""})
    monkeypatch.setattr(ner, "load_adapter", lambda *_a, **_k: pytest.fail("model loaded"))

    report = index_batch_ner(database, "empty")

    assert report["complete"] and report["eligible_articles"] == report["added_articles"] == 0


def test_unused_legacy_run_becomes_resumable_pipeline_run(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana Novak."})
    model = Model()
    old_config = {"model": model.metadata, "device": "cuda", "batch_articles": 32, "batch_size": 16,
                  "ner_decoder": "numpy", "sentence_order": "input"}
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("INSERT INTO ner_runs(ner_run_id,config_json) VALUES ('classla-sl',?)", (
            json.dumps(old_config),
        ))

    first = index_batch_ner(database, "first", model=model)
    retry = index_batch_ner(database, "first", model=model)

    assert first["added_articles"] == 1 and retry["added_articles"] == 0
    assert model.calls == 1
    with closing(sqlite3.connect(database)) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM ner_runs").fetchone()[0])
    assert config["format"] == ner.NER_FORMAT
    assert config["model"] == old_config["model"]
