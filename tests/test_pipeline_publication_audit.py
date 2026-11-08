"""Read-only publication checks over a small committed batch."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from semora.pipeline import publication_audit
from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.publication_audit import audit_publication
from tests.test_pipeline_lexical import CountingLemmatizer
from tests.test_pipeline_semantic import ingest


def test_audit_reports_incomplete_stages_without_publishing(tmp_path: Path) -> None:
    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    semantic = tmp_path / "indexes" / "semantic"
    entities = tmp_path / "indexes" / "entities"
    semantic.mkdir(exist_ok=True)
    entities.mkdir(exist_ok=True)

    first = audit_publication(source, "batch-a", semantic, "a" * 64, entities)
    assert first["stages"]["ingestion"]["complete"] is True
    assert first["stages"]["surface"]["complete"] is False
    assert first["candidate"] is None
    assert not first["eligible"] and not first["publication_ready"]

    index_batch_lexical(source, "batch-a")
    index_batch_lexical(source, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    second = audit_publication(source, "batch-a", semantic, "a" * 64, entities)
    assert second["stages"]["surface"]["complete"] is True
    assert second["stages"]["lemma"]["complete"] is True
    assert second["stages"]["semantic"]["complete"] is False
    assert second["stages"]["entities"]["complete"] is False
    assert second["candidate"] is None


def test_audit_rejects_ambiguous_faiss_generation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64-character"):
        audit_publication(tmp_path, "batch-a", tmp_path, "../unsafe", tmp_path)


def test_audit_checks_lexical_coverage_outside_selected_batch(tmp_path: Path) -> None:
    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    ingest(tmp_path, "batch-b", "Straza/b.md")
    index_batch_lexical(source, "batch-b")
    index_batch_lexical(source, "batch-b", stage="lemma", lemmatizer=CountingLemmatizer())
    semantic = tmp_path / "indexes" / "semantic"
    entities = tmp_path / "indexes" / "entities"
    semantic.mkdir(exist_ok=True)
    entities.mkdir(exist_ok=True)

    report = audit_publication(source, "batch-b", semantic, "a" * 64, entities)

    assert report["stages"]["surface"]["complete"] is False
    assert report["stages"]["lemma"]["complete"] is False
    assert any("outside the selected batch" in item["reason"] for item in report["errors"])


def test_audit_rejects_an_unfinished_other_batch(tmp_path: Path) -> None:
    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    ingest(tmp_path, "batch-b", "Straza/b.md")
    with sqlite3.connect(source) as corpus:
        corpus.execute("UPDATE pipeline_batches SET state='failed' WHERE batch_id='batch-a'")
    semantic = tmp_path / "indexes" / "semantic"
    entities = tmp_path / "indexes" / "entities"
    semantic.mkdir(exist_ok=True)
    entities.mkdir(exist_ok=True)

    report = audit_publication(source, "batch-b", semantic, "a" * 64, entities)

    assert report["stages"]["ingestion"]["complete"] is False
    assert any("batch-a" in item["reason"] for item in report["errors"])


def test_historical_ner_mode_does_not_require_new_batch_outputs(tmp_path: Path, monkeypatch) -> None:
    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    with sqlite3.connect(source) as corpus:
        corpus.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_ner_adoptions "
            "(ner_run_id TEXT PRIMARY KEY,evidence_json TEXT,model_identity_json TEXT)"
        )
        corpus.execute(
            "INSERT INTO pipeline_ner_adoptions(ner_run_id,evidence_json,model_identity_json) "
            "VALUES ('classla-sl','{}','{}')"
        )
    monkeypatch.setattr(publication_audit, "_recorded_identity", lambda *_args: ({}, {"adapter": "classla"}))
    monkeypatch.setattr(publication_audit, "ner_coverage", lambda *_args: (1, 0))
    with sqlite3.connect(source) as corpus:
        assert publication_audit._historical_ner(corpus, "classla-sl")["historical_only"]
        with pytest.raises(ValueError, match="NER is incomplete"):
            publication_audit._ner(corpus, "batch-a", "classla-sl")


def test_partial_publication_requires_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    index_batch_lexical(source, "batch-a")
    index_batch_lexical(source, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    with sqlite3.connect(source) as corpus:
        chunks = corpus.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        corpus_id = corpus.execute("SELECT corpus_id FROM pipeline_corpus_identity").fetchone()[0]
    semantic = tmp_path / "indexes" / "semantic"
    generation = "a" * 64
    private = semantic / "generations" / generation
    private.mkdir(parents=True)
    (semantic / "mapping.sqlite").touch()
    (private / "manifest.json").write_text(json.dumps({"faiss": {"index_type": "flat"}}), encoding="utf-8")
    monkeypatch.setattr(publication_audit, "read_store_config", lambda *_args: {"dimensions": 2, "model_id": "test"})
    monkeypatch.setattr(publication_audit, "store_corpus_identity", lambda *_args: corpus_id)
    monkeypatch.setattr(publication_audit, "validate_corpus_membership", lambda *_args: None)
    monkeypatch.setattr(publication_audit, "vector_coverage", lambda *_args: (chunks, chunks))
    monkeypatch.setattr(publication_audit, "_shards", lambda *_args: [SimpleNamespace(vector_count=chunks)])
    monkeypatch.setattr(
        publication_audit, "_load_generation",
        lambda *_args: (None, {"chunks": chunks, "index_sha256": "b" * 64}),
    )
    monkeypatch.setattr(
        publication_audit, "_historical_ner",
        lambda *_args: {"complete": False, "historical_only": True},
    )

    def historical_entities(_corpus, _root, _run_id, _identity, report):
        report["entity_generation"] = "c" * 32
        report["entity_manifest_sha256"] = "d" * 64
        return {"complete": False, "historical_only": True}

    monkeypatch.setattr(publication_audit, "_historical_entities", historical_entities)
    entity = tmp_path / "indexes" / "entities"
    strict = audit_publication(source, "batch-a", semantic, generation, entity, faiss_module=object())
    partial = audit_publication(
        source, "batch-a", semantic, generation, entity,
        allow_historical_entities=True, faiss_module=object(),
    )

    assert not strict["eligible"] and strict["candidate"] is None
    assert partial["eligible"] and partial["candidate"]["entity_scope"] == "historical_only"


@pytest.mark.parametrize("stage", ["surface", "lemma"])
def test_lexical_audit_uses_active_table_policy_and_rejects_stale_hash(tmp_path, stage):
    from contextlib import closing

    from semora.storage.connection import open_connection

    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    with closing(open_connection(source)) as db, db:
        db.execute("UPDATE articles SET content=content || '<table><td>hidden</td></table>'")
    index_batch_lexical(source, "batch-a")
    index_batch_lexical(source, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    with closing(open_connection(source)) as db:
        assert publication_audit._lexical(db, "batch-a", stage)["complete"]
        with db:
            db.execute("UPDATE pipeline_lexical_state SET input_sha256='stale' WHERE stage=?", (stage,))
        with pytest.raises(ValueError, match="input/configuration differs"):
            publication_audit._lexical(db, "batch-a", stage)


@pytest.mark.parametrize("trust_completed", [False, True])
def test_lexical_audit_rejects_incomplete_table_repair(tmp_path, trust_completed):
    from contextlib import closing

    from semora.storage.connection import open_connection

    source = ingest(tmp_path, "batch-a", "Straza/a.md")
    index_batch_lexical(source, "batch-a")
    with closing(open_connection(source)) as db:
        with db:
            db.execute("UPDATE pipeline_batch_table_policies SET state='repairing'")
        with pytest.raises(ValueError, match="table repair is incomplete"):
            publication_audit._lexical(db, "batch-a", "surface", trust_completed=trust_completed)
