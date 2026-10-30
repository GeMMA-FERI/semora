"""Read-only publication checks over a small committed batch."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_pipeline_lexical import CountingLemmatizer
from test_pipeline_semantic import ingest

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
