import sqlite3
from contextlib import closing

import pytest

from semora.pipeline import ner
from semora.pipeline.ner import index_batch_ner
from tests.test_pipeline_ner import Model, ingest, state


def test_trusted_resume_uses_durable_counts_and_validates_pending_rows(tmp_path, monkeypatch):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana first.", "Paper/b.md": "Ana second."})
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        index_batch_ner(
            database,
            "first",
            model=Model(),
            batch_articles=1,
            progress=lambda event: (
                (_ for _ in ()).throw(RuntimeError("stop after checkpoint"))
                if event["event"] == "ner_checkpoint"
                else None
            ),
        )
    assert len(state(database)[0]) == 1
    monkeypatch.setattr(ner, "_coverage", lambda *_: pytest.fail("trusted resume performed full validation"))
    monkeypatch.setattr(ner, "_trusted_completed_count", lambda *_: pytest.fail("trusted count was rescanned"))
    result = index_batch_ner(
        database,
        "first",
        model=Model(),
        trust_completed_state=True,
        trusted_eligible_articles=2,
        trusted_completed_articles=1,
    )
    assert result["complete"] and result["added_articles"] == result["skipped_articles"] == 1
    assert result["trusted_completed_state"] is True


def test_trusted_resume_rejects_inconsistent_counts_and_retains_write_validation(tmp_path, monkeypatch):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana first."})
    with pytest.raises(ValueError, match="pending articles"):
        index_batch_ner(
            database,
            "first",
            model=Model(),
            trust_completed_state=True,
            trusted_eligible_articles=1,
            trusted_completed_articles=1,
        )

    class ChangesSource(Model):
        def predict_prepared(self, texts, batch_size):
            prediction = super().predict_prepared(texts, batch_size)
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("UPDATE articles SET content='Changed source'")
            return prediction

    with pytest.raises(RuntimeError, match="Article changed during NER"):
        index_batch_ner(
            database,
            "first",
            model=ChangesSource(),
            trust_completed_state=True,
            trusted_eligible_articles=1,
            trusted_completed_articles=0,
        )
    assert state(database) == ([], [])


def test_trusted_resume_requires_explicit_valid_counts(tmp_path):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana first."})
    with pytest.raises(ValueError, match="eligible count"):
        index_batch_ner(database, "first", trust_completed_state=True)
    with pytest.raises(ValueError, match="between zero"):
        index_batch_ner(
            database, "first", trust_completed_state=True, trusted_eligible_articles=1, trusted_completed_articles=2
        )
    with pytest.raises(ValueError, match="require trust"):
        index_batch_ner(database, "first", trusted_eligible_articles=1)


def test_trusted_resume_counts_batch_markers_when_no_checkpoint_total_is_supplied(tmp_path, monkeypatch):
    database = ingest(tmp_path, "first", {"Paper/a.md": "Ana first.", "Paper/b.md": "Ana second."})
    index_batch_ner(database, "first", model=Model())
    monkeypatch.setattr(ner, "_coverage", lambda *_: pytest.fail("trusted resume performed full validation"))
    result = index_batch_ner(database, "first", trust_completed_state=True, trusted_eligible_articles=2)
    assert result["complete"] and result["skipped_articles"] == 2 and result["added_articles"] == 0
