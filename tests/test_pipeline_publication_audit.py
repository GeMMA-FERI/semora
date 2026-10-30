"""Read-only publication checks over a small committed batch."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.test_pipeline_lexical import CountingLemmatizer
from tests.test_pipeline_semantic import ingest

from semora.pipeline.lexical import index_batch_lexical
from semora.pipeline.publication_audit import audit_publication


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
