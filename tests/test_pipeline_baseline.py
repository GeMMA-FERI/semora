from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.corpus import indexer
from semora.pipeline.baseline import adopt_baseline, audit_baseline
from semora.pipeline.lexical import lemma_config
from semora.retrieval.indexing import build_bm25_index, build_lemma_index


class Lemmatizer:
    def lemmatize_many(self, texts):
        return [text.casefold() for text in texts]


def _legacy(tmp_path: Path, *, lexical: bool = False) -> Path:
    corpus = tmp_path / "old-input" / "corpus"
    corpus.mkdir(parents=True)
    newspaper = corpus / "Paper" / "issue.md"
    newspaper.parent.mkdir()
    newspaper.write_text("# First\nOld text one.\n# Second\nOld text two.\n", encoding="utf-8")
    newspaper.with_name("issue_metadata.json").write_text(json.dumps({
        "Record": {"source": "Paper", "identifier": "URN:NBN:SI:doc-ABC"},
    }), encoding="utf-8")
    database = tmp_path / "indexes" / "semora.sqlite"
    indexer.ingest_corpus(corpus, database, stages=("newspapers", "articles"))
    if lexical:
        build_bm25_index(database)
        build_lemma_index(database, lemmatizer=Lemmatizer())
    return database


def _connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def test_audit_reads_stored_text_without_source_files_or_mutation(tmp_path: Path, monkeypatch) -> None:
    database = _legacy(tmp_path, lexical=True)
    before = database.read_bytes()
    # Adoption must not need an assembled source tree, tokenizer, or NLP models.
    (tmp_path / "old-input" / "corpus" / "Paper" / "issue.md").unlink()
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: pytest.fail("tokenizer loaded"))

    report = audit_baseline(database)

    assert database.read_bytes() == before
    assert report["corpus"]["eligible"]
    assert report["corpus"]["verified_newspapers"] == 1
    assert report["corpus"]["verified_articles"] == 2
    for stage in ("surface", "lemma"):
        assert report["lexical"][stage]["eligible"]
        assert report["lexical"][stage]["source_text_verified"] is False
    assert report["lexical"]["lemma"]["historical_model_fingerprint"] is None


def test_adoption_marks_verified_outputs_only_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn:
        before = {table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
                  for table in ("newspapers", "articles", "article_lemmas", "article_fts_map")}
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda *_a, **_k: pytest.fail("tokenizer loaded"))

    result = adopt_baseline(database)
    again = adopt_baseline(database)

    assert result["adoption"]["article_issue_markers_added"] == 1
    assert result["adoption"]["lexical_markers_added"] == {"surface": 2, "lemma": 2}
    assert again["adoption"]["article_issue_markers_added"] == 0
    assert again["adoption"]["lexical_markers_added"] == {"surface": 0, "lemma": 0}
    assert result["adoption"]["chunk_markers_added"] == 0
    with closing(_connect(database)) as conn:
        for table, original in before.items():
            assert original == [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
        assert conn.execute("SELECT COUNT(*) FROM pipeline_issue_stages WHERE stage='chunks'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_adoption_can_trust_existing_audit_without_rerunning_it(tmp_path: Path, monkeypatch) -> None:
    database = _legacy(tmp_path, lexical=True)
    report = audit_baseline(database)
    monkeypatch.setattr("semora.pipeline.baseline._audit", lambda *_args: pytest.fail("repeat audit"))

    adopted = adopt_baseline(database, audit_report=report)

    assert adopted["adoption"]["audit_source"] == "trusted_prior_report"
    assert adopted["adoption"]["lexical_markers_added"] == {"surface": 2, "lemma": 2}
    assert "adoption" not in report


def test_adoption_rejects_ineligible_prior_audit(tmp_path: Path) -> None:
    database = _legacy(tmp_path)
    report = audit_baseline(database)
    report["eligible"] = False

    with pytest.raises(ValueError, match="Supplied baseline audit report"):
        adopt_baseline(database, audit_report=report)


@pytest.mark.parametrize("column,value", [
    ("content", "Changed article"), ("title", "Wrong title"), ("char_start", 99),
    ("line_end", 99), ("article_id", "wrong-id"),
])
def test_exact_article_mismatch_prevents_adoption(tmp_path: Path, column: str, value: object) -> None:
    database = _legacy(tmp_path)
    with closing(_connect(database)) as conn, conn:
        conn.execute(f"UPDATE articles SET {column}=? WHERE rowid=(SELECT MIN(rowid) FROM articles)", (value,))

    assert not audit_baseline(database)["corpus"]["eligible"]
    with pytest.raises(ValueError, match="article audit failed"):
        adopt_baseline(database)
    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_issue_stages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 0


def test_missing_article_prevents_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path)
    with closing(_connect(database)) as conn, conn:
        conn.execute("DELETE FROM articles WHERE rowid=(SELECT MIN(rowid) FROM articles)")
    report = audit_baseline(database)
    assert not report["corpus"]["eligible"]
    assert any("Missing articles" in error["reason"] for error in report["corpus"]["errors"])


def test_incomplete_lexical_output_does_not_block_exact_article_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("UPDATE article_lemma_index_state SET complete=0")

    report = adopt_baseline(database)

    assert report["corpus"]["eligible"]
    assert report["lexical"]["surface"]["eligible"]
    assert not report["lexical"]["lemma"]["eligible"]
    assert report["adoption"]["lexical_markers_added"] == {"surface": 2, "lemma": 0}


def test_missing_surface_fts_row_rejects_legacy_lexical_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("INSERT INTO article_fts(article_fts) VALUES('delete-all')")

    report = adopt_baseline(database)

    assert report["corpus"]["eligible"]
    assert report["lexical"]["surface"]["missing_fts_rows"] == 2
    assert report["adoption"]["lexical_markers_added"] == {"surface": 0, "lemma": 0}


def test_missing_lemma_state_is_reported_not_fabricated(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("DELETE FROM article_lemma_index_state")

    report = adopt_baseline(database)

    assert report["lexical"]["lemma"]["state"] is None
    assert not report["lexical"]["lemma"]["eligible"]
    assert report["adoption"]["lexical_markers_added"]["lemma"] == 0


def test_conflicting_existing_marker_rolls_back_all_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("""
            INSERT INTO pipeline_issue_stages(newspaper_id,stage,input_sha256,config_json,output_count)
            SELECT newspaper_id,'articles','wrong','{}',2 FROM newspapers
        """)

    with pytest.raises(ValueError, match="completion conflicts"):
        adopt_baseline(database)

    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_lexical_state").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 0


def test_missing_database_audit_never_creates_it(tmp_path: Path) -> None:
    target = tmp_path / "absent" / "semora.sqlite"
    with pytest.raises(FileNotFoundError):
        audit_baseline(target)
    assert not target.parent.exists()


def test_schema_13_audit_does_not_apply_additive_migrations(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        names = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pipeline_%'"
        )]
        for name in reversed(names):
            conn.execute(f'DROP TABLE "{name}"')
        conn.execute("DELETE FROM schema_migrations WHERE version>13")
    before = database.read_bytes()

    report = audit_baseline(database)

    assert report["eligible"]
    assert report["lexical"]["surface"]["eligible"]
    assert database.read_bytes() == before
    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 13
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='pipeline_baselines'").fetchone()


def test_surface_rowid_mismatch_rejected_even_when_counts_match(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("INSERT INTO article_fts(article_fts) VALUES('delete-all')")
        conn.executemany("INSERT INTO article_fts(rowid,title,text) VALUES (?,'wrong','wrong')", [(98,), (99,)])

    report = audit_baseline(database)

    assert report["eligible"]
    surface = report["lexical"]["surface"]
    assert surface["counters_match"]
    assert surface["missing_fts_rows"] == surface["extra_fts_rows"] == 2
    assert not surface["eligible"]


def test_bad_lexical_marker_rolls_back_article_and_other_lexical_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    config = json.dumps(lemma_config(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    with closing(_connect(database)) as conn, conn:
        conn.execute("INSERT INTO pipeline_lexical_configs(stage,config_json) VALUES ('lemma',?)", (config,))
        conn.execute("""
            INSERT INTO pipeline_lexical_state(article_id,stage,input_sha256,config_json)
            SELECT article_id,'lemma','wrong',? FROM articles ORDER BY article_id LIMIT 1
        """, (config,))

    with pytest.raises(ValueError, match="Existing lexical input/configuration"):
        adopt_baseline(database)

    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_issue_stages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM pipeline_lexical_state").fetchone()[0] == 1


def test_receipt_preserves_evidence_and_distinct_snapshots(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    first = adopt_baseline(database)
    with closing(_connect(database)) as conn, conn:
        metadata = json.loads(conn.execute("SELECT metadata_json FROM newspapers").fetchone()[0])
        metadata["Record"]["source"] = "Renamed paper"
        conn.execute("UPDATE newspapers SET source=?,metadata_json=?", ("Renamed paper", json.dumps(metadata)))
    second = adopt_baseline(database)
    assert first["adoption"]["baseline_id"] != second["adoption"]["baseline_id"]
    with closing(_connect(database)) as conn:
        receipts = [json.loads(row[0]) for row in conn.execute("SELECT summary_json FROM pipeline_baselines")]
    assert len(receipts) == 2
    for receipt in receipts:
        assert receipt["lexical"]["surface"]["source_text_verified"] is False
        assert receipt["lexical"]["lemma"]["historical_model_fingerprint"] is None


def test_unknown_fts_schema_is_not_adopted_despite_matching_membership(tmp_path: Path) -> None:
    database = _legacy(tmp_path, lexical=True)
    with closing(_connect(database)) as conn, conn:
        conn.execute("DROP TABLE article_fts")
        conn.execute("CREATE VIRTUAL TABLE article_fts USING fts5(title,text,content='',tokenize='porter')")
        conn.execute("INSERT INTO article_fts(rowid,title,text) SELECT m.fts_id,a.title,a.content "
                     "FROM article_fts_map m JOIN articles a USING(article_id)")
    report = adopt_baseline(database)
    assert report["eligible"]
    assert not report["lexical"]["surface"]["schema_compatible"]
    assert report["adoption"]["lexical_markers_added"] == {"surface": 0, "lemma": 0}


def test_empty_article_output_can_be_adopted_without_inference(tmp_path: Path) -> None:
    database = _legacy(tmp_path)
    with closing(_connect(database)) as conn, conn:
        conn.execute("DELETE FROM articles")
        conn.execute("UPDATE newspapers SET content='# Heading only\\n'")

    report = adopt_baseline(database)

    assert report["eligible"]
    assert report["corpus"]["verified_articles"] == 0
    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT output_count FROM pipeline_issue_stages").fetchone()[0] == 0


@pytest.mark.parametrize("field", ["source", "date", "publisher", "rights", "title", "urn", "issue", "volume"])
def test_inconsistent_normalized_metadata_prevents_adoption(tmp_path: Path, field: str) -> None:
    database = _legacy(tmp_path)
    with closing(_connect(database)) as conn, conn:
        conn.execute(f"UPDATE newspapers SET {field}='Wrong stored value'")

    report = audit_baseline(database)

    assert not report["eligible"]
    assert any(
        error["reason"] == f"Stored normalized metadata differs in {field}"
        for error in report["corpus"]["errors"]
    )
    with pytest.raises(ValueError, match="Historical article audit failed"):
        adopt_baseline(database)
    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_issue_stages").fetchone()[0] == 0


def test_missing_newspaper_run_prevents_adoption(tmp_path: Path) -> None:
    database = _legacy(tmp_path)
    with closing(_connect(database)) as conn, conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("UPDATE newspapers SET run_id='missing-run'")
        conn.execute("UPDATE articles SET run_id='missing-run'")

    report = audit_baseline(database)

    assert not report["eligible"]
    assert report["corpus"]["orphan_newspaper_runs"] == 1
    assert any("Newspapers with no corpus run" in error["reason"] for error in report["corpus"]["errors"])
    with pytest.raises(ValueError, match="Historical article audit failed"):
        adopt_baseline(database)
    with closing(_connect(database)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pipeline_baselines").fetchone()[0] == 0
