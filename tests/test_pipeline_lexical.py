from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.corpus import indexer
from semora.pipeline import lexical
from semora.pipeline.batches import run_batch
from semora.pipeline.lexical import (
    adopt_legacy_lexical,
    index_batch_lexical,
    lemma_config,
    lexical_input_sha256,
    lexical_schema_compatible,
    surface_config,
)
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.storage.connection import open_connection


class CountingLemmatizer:
    def __init__(self, *, empty: bool = False, fail_at: int | None = None):
        self.calls: list[list[str]] = []
        self.empty = empty
        self.fail_at = fail_at

    def lemmatize_many(self, texts):
        self.calls.append(list(texts))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected model failure")
        return ["" if self.empty else text.lower().replace("trains", "train") for text in texts]


def ingest(tmp_path: Path, database: Path, batch_id: str, *paths: str) -> list[str]:
    incoming = tmp_path / "incoming"
    issues = []
    article_ids = []
    for name in paths:
        path = incoming / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"# TRAINS {Path(name).stem}\nTrains arrive in Ljubljana.\n"
        metadata = json.dumps({"Record": {"source": "Straža", "date": "1925 02 18"}}).encode()
        sidecar = path.with_name(path.stem + "_metadata.json")
        path.write_text(content, encoding="utf-8")
        sidecar.write_bytes(metadata)
        issues.append(BatchIssue(
            relative_path=name, markdown_path=name, metadata_path=sidecar.relative_to(incoming).as_posix(),
            markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        ))
        article_ids.extend(
            indexer._article_record(
                article, newspaper_id=indexer._stable_id("newspaper", name), run_id="unused", relative_path=name
            ).article_id
            for article in indexer.split_articles(content)
        )
    run_batch(BatchManifest(batch_id, tuple(issues)), incoming, database, stages=("newspapers", "articles"))
    return article_ids


def snapshot(database: Path, table: str) -> list[tuple]:
    assert table in {"article_fts_map", "article_lemmas", "pipeline_lexical_state"}
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()


def test_two_batches_append_retry_and_old_hash_checkpoint_is_irrelevant(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    # Find names whose stable article IDs have the opposite lexical path order.
    names = [f"Straza/{number}.md" for number in range(20)]
    ranked = sorted(names, key=lambda name: indexer._article_record(
        indexer.split_articles(f"# TRAINS {Path(name).stem}\nTrains arrive in Ljubljana.\n")[0],
        newspaper_id="unused", run_id="unused", relative_path=name,
    ).article_id)
    old_ids = ingest(tmp_path, database, "batch-a", ranked[-1])
    first = index_batch_lexical(database, "batch-a")
    model = CountingLemmatizer()
    index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=model)
    before = {table: snapshot(database, table) for table in ("article_fts_map", "article_lemmas")}
    new_ids = ingest(tmp_path, database, "batch-b", ranked[0])
    assert new_ids[0] < old_ids[0]

    second = index_batch_lexical(database, "batch-b", batch_size=1)
    lemma = index_batch_lexical(database, "batch-b", stage="lemma", batch_size=1, lemmatizer=model)
    retry = index_batch_lexical(database, "batch-b", stage="lemma", lemmatizer=model)

    assert first["added_articles"] == second["added_articles"] == lemma["added_articles"] == 1
    assert second["complete"] and lemma["complete"] and retry["complete"]
    assert retry["added_articles"] == 0 and len(model.calls) == 2
    assert not retry["retrieval_ready"]
    assert not retry["publication_ready"]
    for table, rows in before.items():
        assert set(rows).issubset(snapshot(database, table))
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM article_fts WHERE article_fts MATCH 'Trains'"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM article_lemma_fts WHERE article_lemma_fts MATCH 'train'"
        ).fetchone()[0] == 2
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_overlapping_batch_does_not_repeat_existing_lemma_processing(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    model = CountingLemmatizer()
    index_batch_lexical(database, "batch-a")
    index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=model)
    ingest(tmp_path, database, "batch-b", "Straza/a.md", "Straza/b.md")

    surface = index_batch_lexical(database, "batch-b")
    lemma = index_batch_lexical(database, "batch-b", stage="lemma", lemmatizer=model)

    assert surface["added_articles"] == lemma["added_articles"] == 1
    assert surface["skipped_articles"] == lemma["skipped_articles"] == 1
    assert len(model.calls) == 2


@pytest.mark.parametrize("stage", ["surface", "lemma"])
def test_atomic_checkpoint_rollback_and_resume(tmp_path: Path, stage: str) -> None:
    database = tmp_path / "semora.sqlite"
    ids = sorted(ingest(tmp_path, database, "batch-a", "Straza/a.md", "Straza/b.md", "Straza/c.md"))
    if stage == "lemma":
        index_batch_lexical(database, "batch-a")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TRIGGER fail_lexical BEFORE INSERT ON pipeline_lexical_state "
            f"WHEN NEW.stage='{stage}' AND NEW.article_id='{ids[1]}' "
            "BEGIN SELECT RAISE(ABORT,'injected checkpoint failure'); END"
        )
    kwargs = {"lemmatizer": CountingLemmatizer()} if stage == "lemma" else {}

    with pytest.raises(sqlite3.IntegrityError, match="injected checkpoint failure"):
        index_batch_lexical(database, "batch-a", stage=stage, batch_size=1, **kwargs)

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM pipeline_lexical_state WHERE stage=?", (stage,)
        ).fetchone()[0] == 1
        table = "article_fts_map" if stage == "surface" else "article_lemmas"
        fts = "article_fts" if stage == "surface" else "article_lemma_fts"
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 1
        assert connection.execute(f"SELECT COUNT(*) FROM {fts}").fetchone()[0] == 1
        connection.execute("DROP TRIGGER fail_lexical")
    resumed = index_batch_lexical(database, "batch-a", stage=stage, batch_size=2, **kwargs)
    assert resumed["complete"] and resumed["added_articles"] == 2 and resumed["skipped_articles"] == 1


def test_failure_in_second_row_rolls_back_whole_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ids = sorted(ingest(tmp_path, database, "batch-a", "Straza/a.md", "Straza/b.md"))
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TRIGGER fail_lexical BEFORE INSERT ON pipeline_lexical_state "
            f"WHEN NEW.article_id='{ids[1]}' BEGIN SELECT RAISE(ABORT,'checkpoint failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="checkpoint failure"):
        index_batch_lexical(database, "batch-a", batch_size=2)

    assert snapshot(database, "article_fts_map") == []
    assert snapshot(database, "pipeline_lexical_state") == []


def test_empty_annotations_are_completed_once_with_legacy_fallback(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    model = CountingLemmatizer(empty=True)

    assert index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=model)["complete"]
    assert index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=model)["added_articles"] == 0

    assert len(model.calls) == 1
    assert snapshot(database, "article_lemmas")[0][2] == "Trains arrive in Ljubljana."


def test_default_model_not_loaded_when_no_pending_work(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())

    def fail_load(**_kwargs):
        pytest.fail("Unchanged output must not load the model")

    monkeypatch.setattr(lexical, "ClasslaLemmatizer", fail_load)
    assert index_batch_lexical(database, "batch-a", stage="lemma")["added_articles"] == 0


def test_lemma_requires_surface_and_unchanged_configuration(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    with pytest.raises(ValueError, match="surface"):
        index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    index_batch_lexical(database, "batch-a")
    index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    with pytest.raises(ValueError, match="configuration differs"):
        index_batch_lexical(database, "batch-a", stage="lemma", model_fingerprint="new-resources")


def test_changed_input_and_missing_completed_output_are_rejected(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE articles SET content=content || 'changed'")
        connection.commit()
    with pytest.raises(ValueError, match="input/configuration differs"):
        index_batch_lexical(database, "batch-a")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE articles SET content=substr(content,1,length(content)-7)")
        connection.execute("DELETE FROM article_fts_map")
        connection.commit()
    with pytest.raises(ValueError, match="output is missing"):
        index_batch_lexical(database, "batch-a")


def test_existing_rows_without_ledger_require_adoption(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    before = snapshot(database, "article_fts_map")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM pipeline_lexical_state")
        connection.commit()

    with pytest.raises(ValueError, match="baseline adoption"):
        index_batch_lexical(database, "batch-a")

    assert snapshot(database, "article_fts_map") == before
    with closing(open_connection(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert adopt_legacy_lexical(connection, {
            "surface": {"eligible": True, "config": surface_config()},
        }) == {"surface": 1, "lemma": 0}
        connection.commit()
    assert index_batch_lexical(database, "batch-a")["added_articles"] == 0


def test_adoption_requires_transaction_and_unknown_model_identity_is_explicit(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    with closing(open_connection(database)) as connection:
        with pytest.raises(ValueError, match="active audited transaction"):
            adopt_legacy_lexical(connection, {})
    assert lemma_config()["model_fingerprint"] is None
    assert lexical_input_sha256(None, "text") == lexical_input_sha256("", "text")
    assert lexical_input_sha256("Title", "text") != lexical_input_sha256("title", "text")


def test_failed_or_incomplete_article_batch_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM pipeline_issue_stages WHERE stage='articles'")
        connection.commit()
    with pytest.raises(ValueError, match="article extraction"):
        index_batch_lexical(database, "batch-a")


def test_model_failure_preserves_earlier_completed_batches(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md", "Straza/b.md")
    index_batch_lexical(database, "batch-a")
    with pytest.raises(RuntimeError, match="injected model failure"):
        index_batch_lexical(
            database, "batch-a", stage="lemma", batch_size=1, lemmatizer=CountingLemmatizer(fail_at=2)
        )
    assert len(snapshot(database, "article_lemmas")) == 1
    result = index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    assert result["complete"] and result["added_articles"] == 1


def test_progress_reports_only_committed_checkpoints(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md", "Straza/b.md")
    events = []

    index_batch_lexical(database, "batch-a", batch_size=1, progress=events.append)

    assert events == [
        {"event": "lexical_checkpoint", "batch_id": "batch-a", "stage": "surface",
         "completed_articles": 1, "total": 2},
        {"event": "lexical_checkpoint", "batch_id": "batch-a", "stage": "surface",
         "completed_articles": 2, "total": 2},
    ]


def test_zero_eligible_articles_do_not_load_model(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("UPDATE articles SET is_valid=0")
        connection.commit()

    def fail_load(**_kwargs):
        pytest.fail("Empty eligible set must not load the model")

    monkeypatch.setattr(lexical, "ClasslaLemmatizer", fail_load)
    result = index_batch_lexical(database, "batch-a", stage="lemma")

    assert result["complete"] and result["eligible_articles"] == result["completed_articles"] == 0
    assert not result["publication_ready"]


def test_stage_config_is_global_and_incompatible_new_batch_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    index_batch_lexical(database, "batch-a", stage="lemma", lemmatizer=CountingLemmatizer())
    ingest(tmp_path, database, "batch-b", "Straza/b.md")
    index_batch_lexical(database, "batch-b")

    with pytest.raises(ValueError, match="configuration differs"):
        index_batch_lexical(database, "batch-b", stage="lemma", pipeline_type="different-resources")

    with closing(open_connection(database)) as connection:
        configs = connection.execute("SELECT stage,config_json FROM pipeline_lexical_configs ORDER BY stage").fetchall()
        assert len(configs) == 2
        assert json.loads(configs[0]["config_json"]) == lemma_config()
        assert json.loads(configs[1]["config_json"]) == surface_config()


def test_adopted_state_config_is_enforced_by_foreign_key(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    index_batch_lexical(database, "batch-a")
    with closing(open_connection(database)) as connection:
        connection.execute("DELETE FROM pipeline_lexical_state")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        adopt_legacy_lexical(connection, {"surface": {"eligible": True, "config": surface_config()}})
        connection.commit()

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("UPDATE pipeline_lexical_state SET config_json='different'")
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("UPDATE pipeline_lexical_configs SET config_json='different' WHERE stage='surface'")
        connection.rollback()
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_config_registration_rolls_back_with_failed_checkpoint(tmp_path: Path) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TRIGGER fail_lexical BEFORE INSERT ON pipeline_lexical_state "
            "BEGIN SELECT RAISE(ABORT,'checkpoint failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="checkpoint failure"):
        index_batch_lexical(database, "batch-a")

    with closing(open_connection(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pipeline_lexical_configs").fetchone()[0] == 0


@pytest.mark.parametrize("stage", ["surface", "lemma"])
def test_custom_tokenizer_is_rejected_before_model_or_output_changes(tmp_path: Path, monkeypatch, stage: str) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "batch-a", "Straza/a.md")
    table = "article_fts" if stage == "surface" else "article_lemma_fts"
    with closing(open_connection(database)) as connection:
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"CREATE VIRTUAL TABLE {table} USING fts5(title,text,content='',tokenize='porter')")
        connection.commit()
        assert not lexical_schema_compatible(connection, stage)
    before = database.read_bytes()

    def fail_load(**_kwargs):
        pytest.fail("Unsupported schemas must be rejected before model loading")

    monkeypatch.setattr(lexical, "ClasslaLemmatizer", fail_load)
    with pytest.raises(ValueError, match="schema/tokenizer"):
        index_batch_lexical(database, "batch-a", stage=stage)

    assert database.read_bytes() == before


@pytest.mark.parametrize("stage", ["surface", "lemma"])
def test_unadopted_historical_outputs_outside_new_batch_are_rejected(tmp_path: Path, stage: str) -> None:
    database = tmp_path / "semora.sqlite"
    ingest(tmp_path, database, "historical", "Straza/old.md")
    index_batch_lexical(database, "historical")
    if stage == "lemma":
        index_batch_lexical(database, "historical", stage="lemma", lemmatizer=CountingLemmatizer())
    with closing(open_connection(database)) as connection:
        # Retain the historical output but remove pipeline provenance to model
        # a legacy stage that has not been audited/adopted yet.
        connection.execute("DELETE FROM pipeline_lexical_state WHERE stage=?", (stage,))
        connection.execute("DELETE FROM pipeline_lexical_configs WHERE stage=?", (stage,))
        connection.commit()
    old_outputs = snapshot(database, "article_fts_map" if stage == "surface" else "article_lemmas")
    ingest(tmp_path, database, "new-batch", "Straza/new.md")
    if stage == "lemma":
        index_batch_lexical(database, "new-batch")

    with pytest.raises(ValueError, match="baseline adoption"):
        index_batch_lexical(database, "new-batch", stage=stage, lemmatizer=CountingLemmatizer())

    assert snapshot(database, "article_fts_map" if stage == "surface" else "article_lemmas") == old_outputs


def test_schema_recognizes_supported_quoted_layout_and_rejects_extra_columns() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute(
            'CREATE VIRTUAL TABLE "article_fts" USING fts5("title",[text],'
            'tokenize="unicode61 remove_diacritics 2",content="")'
        )
        assert lexical_schema_compatible(connection, "surface")
        assert not lexical_schema_compatible(connection, "lemma")
        connection.execute(
            "CREATE VIRTUAL TABLE article_lemma_fts USING fts5(title,text,extra,"
            "content='',tokenize='unicode61 remove_diacritics 2')"
        )
        assert not lexical_schema_compatible(connection, "lemma")
