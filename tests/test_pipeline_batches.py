from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from semora.corpus import indexer
from semora.pipeline import batches
from semora.pipeline.batches import batch_status, plan_batch, run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.storage import Database


class CountingTokenizer:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self, text: str, **_kwargs) -> dict:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("simulated tokenizer interruption")
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


def issue(
    root: Path,
    relative_path: str,
    markdown: str = "# Article\nAlpha beta gamma delta.\n",
    *,
    date: str = "1925 02 18",
    identifier: str | None = None,
) -> BatchIssue:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_bytes = markdown.encode("utf-8")
    metadata = {"Record": {"source": "Straža", "date": date}}
    if identifier is not None:
        metadata["Record"]["identifier"] = identifier
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    path.write_bytes(markdown_bytes)
    metadata_path = path.with_name(path.stem + "_metadata.json")
    metadata_path.write_bytes(metadata_bytes)
    return BatchIssue(
        relative_path=relative_path,
        markdown_path=relative_path,
        metadata_path=metadata_path.relative_to(root).as_posix(),
        markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        source_identifier=identifier,
    )


def run(manifest: BatchManifest, root: Path, database: Path, tokenizer=None, **kwargs):
    return run_batch(
        manifest, root, database, token_count=3, token_overlap=1,
        tokenizer=tokenizer or CountingTokenizer(), **kwargs,
    )


def rows(database: Path, table: str) -> list[tuple]:
    assert table in {"newspapers", "articles", "chunks"}
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def test_plan_missing_database_does_not_create_it(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    manifest = BatchManifest("batch-a", (issue(source, "Straza/a.md"),))
    database = tmp_path / "not-created" / "semora.sqlite"

    plan = plan_batch(manifest, source, database)

    assert plan["batch_id"] == "batch-a"
    assert plan["manifest_fingerprint"] == manifest.fingerprint
    assert plan["counts"]["new"] == 1
    assert not database.parent.exists()


def test_two_batches_append_and_retry_without_reprocessing(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "indexes" / "semora.sqlite"
    first = BatchManifest("batch-a", (issue(source, "Straza/a.md"),))
    tokenizer = CountingTokenizer()
    run(first, source, database, tokenizer)
    before = {table: rows(database, table) for table in ("newspapers", "articles", "chunks")}
    first_calls = tokenizer.calls
    run(first, source, database, tokenizer)
    assert tokenizer.calls == first_calls
    assert before == {table: rows(database, table) for table in before}

    second = BatchManifest("batch-b", (issue(source, "Straza/b.md", "# New article\nNovel content here.\n"),))
    run(second, source, database, tokenizer)

    for table in before:
        after = rows(database, table)
        assert len(after) > len(before[table])
        assert set(before[table]).issubset(after)
    with closing(sqlite3.connect(database)) as connection:
        for newspaper_id, relative_path in connection.execute("SELECT newspaper_id,relative_path FROM newspapers"):
            assert newspaper_id == indexer._stable_id("newspaper", relative_path)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_new_batch_with_unchanged_issue_does_not_repeat_nlp(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    original = issue(source, "Straza/a.md")
    run(BatchManifest("batch-a", (original,)), source, database)
    before = rows(database, "chunks")
    tokenizer = CountingTokenizer(fail_on=1)

    run(BatchManifest("batch-b", (original,)), source, database, tokenizer)

    assert tokenizer.calls == 0
    assert rows(database, "chunks") == before


def test_metadata_update_keeps_text_ids_and_derived_records(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    original = issue(source, "Straza/a.md", identifier="GF-URN:example")
    run(BatchManifest("batch-a", (original,)), source, database)
    before = {table: rows(database, table) for table in ("articles", "chunks")}
    revised = issue(source, "Straza/a.md", date="1925 02 19", identifier="GF-URN:example")
    manifest = BatchManifest("metadata-correction", (revised,))
    assert plan_batch(manifest, source, database)["counts"]["metadata_changed"] == 1
    tokenizer = CountingTokenizer(fail_on=1)

    run(manifest, source, database, tokenizer)

    assert tokenizer.calls == 0
    assert before == {table: rows(database, table) for table in before}
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT date,urn FROM newspapers").fetchone() == ("1925-02-19", None)
        metadata = json.loads(connection.execute("SELECT metadata_json FROM newspapers").fetchone()[0])
        assert metadata["Record"]["identifier"] == "GF-URN:example"


def test_changed_text_is_rejected_before_new_issue_is_inserted(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    original = issue(source, "Straza/z.md")
    run(BatchManifest("batch-a", (original,)), source, database)
    before = {table: rows(database, table) for table in ("newspapers", "articles", "chunks")}
    manifest = BatchManifest("batch-b", (
        issue(source, "Straza/a-new.md"),
        issue(source, "Straza/z.md", "# Article\nReplaced OCR.\n"),
    ))
    assert plan_batch(manifest, source, database)["counts"]["conflict"] == 1

    with pytest.raises(ValueError):
        run(manifest, source, database)

    assert before == {table: rows(database, table) for table in before}


def test_invalid_later_input_prevents_all_corpus_writes(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("invalid-input", (
        issue(source, "Straza/a.md"), issue(source, "Straza/z.md"),
    ))
    (source / "Straza/z.md").write_text("Changed after manifest preparation", encoding="utf-8")

    with pytest.raises(ValueError):
        run(manifest, source, database)

    if database.exists():
        with closing(sqlite3.connect(database)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "newspapers" in tables:
                assert connection.execute("SELECT COUNT(*) FROM newspapers").fetchone()[0] == 0


def test_tokenizer_failure_rolls_back_whole_issue_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("interrupted", (issue(
        source, "Straza/a.md", "# First\nOne two three four.\n# Second\nFive six seven eight.\n",
    ),))

    with pytest.raises(RuntimeError, match="simulated tokenizer interruption"):
        run(manifest, source, database, CountingTokenizer(fail_on=2))

    assert rows(database, "newspapers") == []
    assert rows(database, "articles") == []
    assert rows(database, "chunks") == []
    assert batch_status(database, manifest.batch_id)["state"] == "failed"
    tokenizer = CountingTokenizer()
    run(manifest, source, database, tokenizer)
    assert tokenizer.calls == 2
    assert len(rows(database, "chunks")) == 4
    complete = rows(database, "chunks")
    run(manifest, source, database, CountingTokenizer(fail_on=1))
    assert rows(database, "chunks") == complete


def test_empty_article_output_has_durable_completion(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("heading-only", (issue(source, "Straza/a.md", "# Empty heading\n"),))
    tokenizer = CountingTokenizer(fail_on=1)
    run(manifest, source, database, tokenizer)
    assert rows(database, "articles") == []
    assert rows(database, "chunks") == []

    def unexpected_split(_text):
        raise AssertionError("completed article extraction must not run again")

    monkeypatch.setattr(batches, "split_articles", unexpected_split)
    run(manifest, source, database, tokenizer)
    assert tokenizer.calls == 0


def test_same_batch_rejects_changed_chunk_configuration(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("batch-a", (issue(source, "Straza/a.md"),))
    run(manifest, source, database)
    before = rows(database, "chunks")

    with pytest.raises(ValueError):
        run_batch(manifest, source, database, token_count=5, token_overlap=1, tokenizer=CountingTokenizer())

    assert rows(database, "chunks") == before


def test_plan_does_not_modify_existing_database(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("batch-a", (issue(source, "Straza/a.md"),))
    run(manifest, source, database)
    before = database.read_bytes()

    plan = plan_batch(manifest, source, database)

    assert plan["counts"]["unchanged"] == 1
    assert database.read_bytes() == before


def test_missing_batch_status_never_creates_database(tmp_path: Path) -> None:
    database = tmp_path / "missing" / "semora.sqlite"

    with pytest.raises((FileNotFoundError, ValueError)):
        batch_status(database, "unknown")

    assert not database.parent.exists()


def test_failure_retains_previous_issue_and_retries_only_pending_work(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    manifest = BatchManifest("interrupted", (
        issue(source, "Straza/a.md"), issue(source, "Straza/z.md"),
    ))

    with pytest.raises(RuntimeError, match="simulated tokenizer interruption"):
        run(manifest, source, database, CountingTokenizer(fail_on=2))

    assert len(rows(database, "newspapers")) == 1
    assert len(rows(database, "articles")) == 1
    assert len(rows(database, "chunks")) == 2
    previous_chunks = rows(database, "chunks")
    tokenizer = CountingTokenizer()
    run(manifest, source, database, tokenizer)
    assert tokenizer.calls == 1
    assert len(rows(database, "newspapers")) == 2
    assert set(previous_chunks).issubset(rows(database, "chunks"))
    status = batch_status(database, manifest.batch_id)
    assert status["state"] == "ingested"
    assert status["stages"] == {"articles": 2, "chunks": 2}
    assert status["retrieval_ready"] is False


def test_duplicate_metadata_urn_is_a_preflight_conflict(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    originals = (
        issue(source, "News/a.md", identifier="URN:NBN:SI:doc-SAME"),
        issue(source, "News/b.md", identifier="URN:NBN:SI:doc-SAME"),
    )
    # Source identifiers are optional; metadata still supplies a source identity.
    manifest = BatchManifest("duplicate-urn", tuple(
        BatchIssue(
            relative_path=record.relative_path,
            markdown_path=record.markdown_path,
            metadata_path=record.metadata_path,
            markdown_sha256=record.markdown_sha256,
            metadata_sha256=record.metadata_sha256,
        )
        for record in originals
    ))

    assert plan_batch(manifest, source, database)["counts"]["conflict"] > 0
    with pytest.raises(ValueError):
        run(manifest, source, database)
    if database.exists():
        with closing(sqlite3.connect(database)) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "newspapers" in tables:
                assert connection.execute("SELECT COUNT(*) FROM newspapers").fetchone()[0] == 0


def test_canonical_paths_cannot_collide_by_case_across_batches(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    run(BatchManifest("batch-a", (issue(source, "News/a.md"),)), source, database)
    before = rows(database, "newspapers")
    incoming = BatchManifest("batch-b", (issue(source, "News/A.md"),))

    assert plan_batch(incoming, source, database)["counts"]["conflict"] == 1
    with pytest.raises(ValueError):
        run(incoming, source, database)
    assert rows(database, "newspapers") == before


def test_selected_stages_resume_after_input_and_data_root_move(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    source = original_root / "incoming"
    database = original_root / "indexes" / "semora.sqlite"
    manifest = BatchManifest("portable", (issue(source, "Straza/a.md"),))
    run(manifest, source, database, stages=("newspapers",))
    assert len(rows(database, "newspapers")) == 1
    assert rows(database, "articles") == []
    newspaper_records = rows(database, "newspapers")

    moved = tmp_path / "relocated"
    shutil.move(str(original_root), str(moved))
    source = moved / "incoming"
    database = moved / "indexes" / "semora.sqlite"
    run(manifest, source, database, stages=("newspapers", "articles"))
    assert len(rows(database, "articles")) == 1
    assert rows(database, "chunks") == []
    run(manifest, source, database)
    assert rows(database, "newspapers") == newspaper_records
    assert len(rows(database, "chunks")) == 2
    assert batch_status(database, manifest.batch_id)["retrieval_ready"] is False


def test_stale_batch_replay_cannot_restore_superseded_metadata(tmp_path: Path) -> None:
    original_source = tmp_path / "original-input"
    correction_source = tmp_path / "correction-input"
    database = tmp_path / "semora.sqlite"
    original = BatchManifest("original", (issue(original_source, "Straza/a.md", date="1925 02 18"),))
    run(original, original_source, database)
    correction = BatchManifest("correction", (issue(correction_source, "Straza/a.md", date="1925 02 19"),))
    run(correction, correction_source, database)
    before = rows(database, "newspapers")

    plan = plan_batch(original, original_source, database)
    assert plan["counts"]["conflict"] == 1
    assert "superseded" in plan["issues"][0]["reason"]
    with pytest.raises(ValueError):
        run(original, original_source, database)

    assert rows(database, "newspapers") == before
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT date FROM newspapers").fetchone()[0] == "1925-02-19"


def test_metadata_derived_gf_identifier_stays_unique_across_batches(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    original = replace(issue(source, "Straza/a.md", identifier="GF-URN:example"), source_identifier=None)
    run(BatchManifest("batch-a", (original,)), source, database)
    duplicate = replace(issue(source, "Straza/b.md", identifier="gf-urn:EXAMPLE"), source_identifier=None)
    manifest = BatchManifest("batch-b", (duplicate,))
    before = rows(database, "newspapers")

    assert plan_batch(manifest, source, database)["counts"]["conflict"] == 1
    with pytest.raises(ValueError):
        run(manifest, source, database)
    assert rows(database, "newspapers") == before


def test_manifest_identifier_must_agree_with_metadata_before_writes(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    incoming = replace(issue(source, "Straza/a.md", identifier="GF-URN:actual"), source_identifier="GF-URN:wrong")
    manifest = BatchManifest("bad-identifier", (incoming,))

    with pytest.raises(ValueError, match="disagrees with metadata"):
        plan_batch(manifest, source, database)
    assert not database.exists()
    with pytest.raises(ValueError, match="disagrees with metadata"):
        run(manifest, source, database)
    assert not database.exists()


def test_manifest_only_source_identity_cannot_change_on_existing_issue(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    original = replace(issue(source, "News/a.md"), source_identifier="external:one")
    run(BatchManifest("one", (original,)), source, database)
    changed = replace(original, source_identifier="external:two")
    plan = plan_batch(BatchManifest("two", (changed,)), source, database)
    assert plan["counts"]["conflict"] == 1
    with pytest.raises(ValueError, match="source conflicts"):
        run(BatchManifest("two", (changed,)), source, database)


def test_new_batch_migration_can_resume_after_partial_schema_creation(tmp_path: Path) -> None:
    database = Database(tmp_path / "semora.sqlite")
    try:
        database.initialize()
        with database.conn:
            database.conn.execute("DROP TABLE pipeline_issue_stages")
            database.conn.execute("DROP TABLE pipeline_batch_issues")
            database.conn.execute("DELETE FROM schema_migrations WHERE version=14")
        database.initialize()
        assert database.conn.execute("SELECT version FROM schema_migrations WHERE version=14").fetchone()[0] == 14
        assert database.conn.execute("SELECT COUNT(*) FROM pipeline_issue_stages").fetchone()[0] == 0
    finally:
        database.close()


def test_additive_runner_does_not_apply_historical_migrations(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    target = tmp_path / "semora.sqlite"
    manifest = BatchManifest("batch", (issue(source, "News/a.md"),))
    database = Database(target)
    try:
        database.initialize()
        with database.conn:
            database.conn.execute("DELETE FROM schema_migrations WHERE version=13")
    finally:
        database.close()
    before = target.read_bytes()
    with pytest.raises(ValueError, match="explicitly migrate the legacy schema"):
        run(manifest, source, target)
    assert target.read_bytes() == before


def test_legacy_article_outputs_require_adoption_before_any_append(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    issue(source, "News/old.md")
    indexer.ingest_corpus(source, database, stages=("newspapers", "articles"))
    incoming = issue(source, "News/new.md")
    before = database.read_bytes()
    with pytest.raises(ValueError, match="Historical article outputs require baseline adoption"):
        run(BatchManifest("new", (incoming,)), source, database)
    assert database.read_bytes() == before


def test_new_batch_cannot_mix_incompatible_chunk_configuration(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    run(BatchManifest("old", (issue(source, "News/old.md"),)), source, database)
    incoming = issue(source, "News/new.md")
    before = database.read_bytes()
    with pytest.raises(ValueError, match="Existing chunks configuration differs"):
        run_batch(BatchManifest("new", (incoming,)), source, database,
                  token_count=5, token_overlap=1, tokenizer=CountingTokenizer())
    assert database.read_bytes() == before


def test_legacy_lf_storage_matches_hash_verified_crlf_input(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    database = tmp_path / "semora.sqlite"
    incoming = issue(source, "Straza/a.md", "# Article\r\nAlpha beta gamma.\r\n")
    indexer.ingest_corpus(source, database, stages=("newspapers",))
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT content FROM newspapers").fetchone()[0] == "# Article\nAlpha beta gamma.\n"
    before = database.read_bytes()

    plan = plan_batch(BatchManifest("legacy-match", (incoming,)), source, database)

    assert plan["counts"]["unchanged"] == 1
    assert database.read_bytes() == before
