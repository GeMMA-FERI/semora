"""Read-only legacy entity dictionary audit against a disposable NER corpus."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.ner.base import Entity, Prediction
from semora.ner.indexing import build_ner_index
from semora.ner.names import build_entity_names
from semora.ner.spellfix import install_spellfix
from semora.pipeline.entities import index_batch_entities
from semora.pipeline.entity_adoption import adopt_entity_baseline
from semora.pipeline.entity_baseline import audit_entity_baseline
from semora.pipeline.ner import index_batch_ner
from semora.pipeline.ner_adoption import adopt_ner_baseline
from semora.storage import Database


class Model:
    metadata = {"adapter": "fixture", "version": 1}

    def prepare(self, texts):
        return texts

    def predict_prepared(self, texts, _batch_size):
        def entities(text):
            if text.startswith("Ana"):
                return [Entity(0, 3, "PER", "Ana"), Entity(0, 3, "ORG", "Ana")]
            if text.startswith("Zürich"):
                return [Entity(0, 6, "LOC", "Zürich")]
            return []

        return Prediction([entities(text) for text in texts])


@pytest.fixture
def legacy(tmp_path: Path, monkeypatch):
    library = os.environ.get("SEMORA_TEST_SPELLFIX")
    if not library:
        pytest.skip("Set SEMORA_TEST_SPELLFIX to a trusted local spellfix library")
    monkeypatch.delenv("SEMORA_SPELLFIX_PATH", raising=False)
    install_spellfix(tmp_path, library)
    path = tmp_path / "indexes" / "semora.sqlite"
    db = Database(path)
    try:
        db.initialize()
        with db.conn:
            db.conn.execute("INSERT INTO runs(run_id,run_type) VALUES ('source','test')")
            db.conn.execute("INSERT INTO newspapers(newspaper_id,run_id,content) VALUES ('n','source','')")
            db.conn.executemany(
                "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid) VALUES (?,'source','n',?,1)",
                [("a", "Ana gre domov."), ("b", "No names here.")],
            )
    finally:
        db.close()
    build_ner_index(path, model=Model(), device="cpu", pipeline_depth=1)
    manifest = build_entity_names(tmp_path, batch_size=1)
    dictionary = tmp_path / "indexes" / "entities" / f"{manifest['generation']}.sqlite"
    return path, dictionary


def test_complete_dictionary_audit_is_read_only_and_preserves_duplicate_labels(legacy):
    source, dictionary = legacy
    before = source.read_bytes(), dictionary.read_bytes()

    report = audit_entity_baseline(source, dictionary.parent)

    assert report["eligible"] and report["complete"], report["errors"]
    assert report["counts"]["expected_memberships"] == 1
    assert report["counts"]["zero_name_articles"] == 1
    assert report["counts"]["pending_memberships"] == 0
    assert (source.read_bytes(), dictionary.read_bytes()) == before


def test_missing_expected_membership_is_pending_not_invalid(legacy):
    source, dictionary = legacy
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(
            "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid) "
            "VALUES ('c','source','n','Ana again.',1)"
        )
        conn.execute(
            "INSERT INTO article_ner_state(ner_run_id,article_id,content_sha256,entity_count) VALUES (?,?,?,?)",
            ("classla-sl", "c", hashlib.sha256(b"Ana again.").hexdigest(), 1),
        )
        conn.execute(
            "INSERT INTO article_entities(ner_run_id,article_id,entity_index,text,label,char_start,char_end) "
            "VALUES ('classla-sl','c',0,'Ana','PER',0,3)"
        )

    report = audit_entity_baseline(source, dictionary.parent)

    assert report["eligible"], report["errors"]
    assert not report["complete"]
    assert report["pending_memberships"] == 1


def test_wrong_existing_dictionary_membership_fails_closed(legacy):
    source, dictionary = legacy
    with closing(sqlite3.connect(dictionary)) as conn, conn:
        conn.execute("UPDATE entity_articles SET content_sha256='wrong'")

    report = audit_entity_baseline(source, dictionary.parent)

    assert not report["eligible"]
    assert any("hash/earliest offset" in row["reason"] for row in report["errors"])


def test_entity_adoption_adds_control_schema_without_rewriting_memberships(legacy, monkeypatch):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    with closing(sqlite3.connect(dictionary)) as conn:
        before = (
            conn.execute("SELECT * FROM names").fetchall(),
            conn.execute("SELECT * FROM entity_articles").fetchall(),
        )

    first = adopt_entity_baseline(source, dictionary.parent)
    from semora.pipeline import entity_adoption

    monkeypatch.setattr(entity_adoption, "audit_entity_baseline", lambda *_a, **_k: pytest.fail("No repeat full scan"))
    second = adopt_entity_baseline(source, dictionary.parent)

    assert first["adoption"]["already_adopted"] is False
    assert second["adoption"]["already_adopted"] is True
    with closing(sqlite3.connect(dictionary)) as conn:
        assert before == (
            conn.execute("SELECT * FROM names").fetchall(),
            conn.execute("SELECT * FROM entity_articles").fetchall(),
        )
        assert conn.execute("SELECT COUNT(*) FROM pipeline_entity_profile").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM pipeline_entity_articles").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='pipeline_entity_articles_lookup'"
        ).fetchone()


def test_entity_adoption_can_trust_prior_report_without_rescan(legacy, monkeypatch):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    report = audit_entity_baseline(source, dictionary.parent)
    assert report["eligible"] and report["complete"]
    monkeypatch.setattr(
        "semora.pipeline.entity_adoption.audit_entity_baseline",
        lambda *_a, **_k: pytest.fail("repeat entity audit"),
    )

    adopted = adopt_entity_baseline(source, dictionary.parent, audit_report=report)

    assert adopted["adoption"]["prior_audit_trusted_without_rescan"] is True
    assert adopted["evidence_fingerprint"] == report["evidence_fingerprint"]


def test_entity_adoption_requires_ner_adoption_and_valid_dictionary(legacy):
    source, dictionary = legacy
    with pytest.raises(ValueError, match="adopt the historical NER"):
        adopt_entity_baseline(source, dictionary.parent)
    adopt_ner_baseline(source)
    with closing(sqlite3.connect(dictionary)) as conn, conn:
        conn.execute("UPDATE entity_articles SET content_sha256='wrong'")

    with pytest.raises(ValueError, match="incomplete or invalid"):
        adopt_entity_baseline(source, dictionary.parent)

    with closing(sqlite3.connect(dictionary)) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='pipeline_entity_profile'").fetchone() is None


def _batch(source: Path, batch_id: str, name: str, body: str) -> None:
    # The dictionary writer depends on the committed batch ledger. This small
    # fixture inserts that ledger directly because its legacy IDs predate the
    # canonical corpus-baseline fixture used by the ingestion tests.
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(
            "INSERT INTO newspapers(newspaper_id,run_id,content,relative_path) VALUES (?,'source',?,?)",
            (name, body, f"Straza/{name}.md"),
        )
        conn.execute(
            "INSERT INTO articles(article_id,run_id,newspaper_id,content,is_valid) VALUES (?,'source',?,?,1)",
            (name, name, body),
        )
        conn.execute(
            "INSERT INTO pipeline_batches(batch_id,manifest_fingerprint,config_json,issue_count,state) "
            "VALUES (?,'fixture','{}',1,'ingested')",
            (batch_id,),
        )
        conn.execute(
            "INSERT INTO pipeline_batch_issues(batch_id,relative_path,newspaper_id,action,content_sha256,"
            "metadata_sha256) VALUES (?,?,?,'new',?,?)",
            (batch_id, f"Straza/{name}.md", name, hashlib.sha256(body.encode()).hexdigest(), "0" * 64),
        )
        conn.execute(
            "INSERT INTO pipeline_issue_stages(newspaper_id,stage,input_sha256,config_json,output_count) "
            "VALUES (?,'articles',?,'{}',1)",
            (name, hashlib.sha256(body.encode()).hexdigest()),
        )
    index_batch_ner(source, batch_id, model=Model())


def test_additive_dictionary_updates_only_new_memberships_and_retries(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-a", "issue-a", "Ana arrives.")

    first = index_batch_entities(source, "batch-a", dictionary.parent)
    with closing(sqlite3.connect(dictionary)) as conn:
        token = conn.execute("SELECT advancement_token FROM pipeline_entity_profile").fetchone()[0]
    second = index_batch_entities(source, "batch-a", dictionary.parent)

    assert first["complete"] and first["added_memberships"] == 1
    assert second["added_articles"] == second["added_memberships"] == 0
    with closing(sqlite3.connect(dictionary)) as conn:
        assert conn.execute("SELECT article_count FROM names WHERE normalized_name='ana'").fetchone()[0] == 2
        assert conn.execute("SELECT rank FROM name_spellfix_vocab WHERE word='ana'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM pipeline_entity_articles").fetchone()[0] == 1
        assert conn.execute("SELECT advancement_token FROM pipeline_entity_profile").fetchone()[0] == token


def test_new_unicode_name_and_zero_name_article_are_checkpointed(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-new", "issue-new", "Zürich works.")
    _batch(source, "batch-zero", "issue-zero", "No entity here.")

    named = index_batch_entities(source, "batch-new", dictionary.parent)
    empty = index_batch_entities(source, "batch-zero", dictionary.parent)

    assert named["added_names"] == named["added_memberships"] == 1
    assert empty["added_names"] == empty["added_memberships"] == 0
    with closing(sqlite3.connect(dictionary)) as conn:
        assert conn.execute("SELECT name FROM names WHERE normalized_name='zürich'").fetchone()[0] == "Zürich"
        assert (
            conn.execute("SELECT name_count FROM pipeline_entity_articles WHERE article_id='issue-zero'").fetchone()[0]
            == 0
        )


def test_dictionary_checkpoint_rolls_back_memberships_on_late_failure(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-new", "issue-new", "Zürich works.")
    with closing(sqlite3.connect(dictionary)) as conn, conn:
        conn.execute("""
            CREATE TRIGGER reject_completion BEFORE INSERT ON pipeline_entity_articles
            BEGIN SELECT RAISE(ABORT, 'late dictionary failure'); END
        """)

    with pytest.raises(sqlite3.IntegrityError, match="late dictionary failure"):
        index_batch_entities(source, "batch-new", dictionary.parent)

    with closing(sqlite3.connect(dictionary)) as conn, conn:
        assert conn.execute("SELECT 1 FROM names WHERE normalized_name='zürich'").fetchone() is None
        assert conn.execute("SELECT COUNT(*) FROM pipeline_entity_articles").fetchone()[0] == 0
        conn.execute("DROP TRIGGER reject_completion")
    assert index_batch_entities(source, "batch-new", dictionary.parent)["added_memberships"] == 1


def test_dictionary_binding_rejects_wrong_corpus_identity(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-a", "issue-a", "Ana arrives.")
    with closing(sqlite3.connect(dictionary)) as conn, conn:
        conn.execute("UPDATE pipeline_entity_profile SET corpus_id='0'")

    with pytest.raises(ValueError, match="not adopted for this corpus"):
        index_batch_entities(source, "batch-a", dictionary.parent)


def test_restored_older_corpus_with_same_uuid_cannot_extend_newer_dictionary(legacy, tmp_path):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-a", "issue-a", "Ana arrives.")
    backup = tmp_path / "older.sqlite"
    with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(backup)) as saved:
        original.backup(saved)

    assert index_batch_entities(source, "batch-a", dictionary.parent)["added_memberships"] == 1
    with closing(sqlite3.connect(backup)) as saved, closing(sqlite3.connect(source)) as restored:
        saved.backup(restored)
    _batch(source, "batch-b", "issue-b", "Zürich works.")

    with pytest.raises(ValueError, match="advancement tokens differ"):
        index_batch_entities(source, "batch-b", dictionary.parent)
    with closing(sqlite3.connect(dictionary)) as conn:
        assert conn.execute("SELECT 1 FROM names WHERE normalized_name='zürich'").fetchone() is None


def test_interrupted_finalization_can_reconcile_committed_dictionary_checkpoint(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    _batch(source, "batch-a", "issue-a", "Ana arrives.")
    with closing(sqlite3.connect(source)) as conn:
        old_token = conn.execute(
            "SELECT advancement_token FROM pipeline_entity_bindings WHERE ner_run_id='classla-sl'"
        ).fetchone()[0]
    assert index_batch_entities(source, "batch-a", dictionary.parent)["added_memberships"] == 1
    with closing(sqlite3.connect(dictionary)) as conn:
        new_token = conn.execute(
            "SELECT advancement_token FROM pipeline_entity_profile WHERE profile_id=1"
        ).fetchone()[0]
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(
            "UPDATE pipeline_entity_bindings SET advancement_token=?,pending_token=? "
            "WHERE ner_run_id='classla-sl'", (old_token, new_token),
        )

    result = index_batch_entities(source, "batch-a", dictionary.parent)
    assert result["added_articles"] == result["added_memberships"] == 0
    with closing(sqlite3.connect(source)) as conn, closing(sqlite3.connect(dictionary)) as target:
        assert conn.execute(
            "SELECT advancement_token,pending_token FROM pipeline_entity_bindings "
            "WHERE ner_run_id='classla-sl'"
        ).fetchone() == (
            target.execute("SELECT advancement_token FROM pipeline_entity_profile").fetchone()[0], None
        )


def test_overlapping_batch_registers_existing_dictionary_pairs_without_incrementing_counts(legacy):
    source, dictionary = legacy
    adopt_ner_baseline(source)
    adopt_entity_baseline(source, dictionary.parent)
    with closing(sqlite3.connect(source)) as conn, conn:
        conn.execute(
            "INSERT INTO pipeline_batches(batch_id,manifest_fingerprint,config_json,issue_count,state) "
            "VALUES ('overlap','fixture','{}',1,'ingested')"
        )
        conn.execute(
            "INSERT INTO pipeline_batch_issues(batch_id,relative_path,newspaper_id,action,content_sha256,"
            "metadata_sha256) VALUES ('overlap','legacy.md','n','unchanged',?,?)",
            ("0" * 64, "0" * 64),
        )
        conn.execute(
            "INSERT INTO pipeline_issue_stages(newspaper_id,stage,input_sha256,config_json,output_count) "
            "VALUES ('n','articles',?,'{}',2)", ("0" * 64,),
        )

    first = index_batch_entities(source, "overlap", dictionary.parent)
    second = index_batch_entities(source, "overlap", dictionary.parent)

    assert first["eligible_articles"] == first["added_articles"] == 2
    assert first["added_memberships"] == first["added_names"] == 0
    assert second["added_articles"] == second["added_memberships"] == 0
    with closing(sqlite3.connect(dictionary)) as conn:
        assert conn.execute("SELECT article_count FROM names WHERE normalized_name='ana'").fetchone()[0] == 1
