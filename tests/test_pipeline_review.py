"""Independent source-boundary and recovery regressions for the additive pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.corpus.indexer import ingest_corpus
from semora.pipeline.baseline import adopt_baseline
from semora.pipeline.batches import plan_batch, run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest


def _write_issue(root: Path, relative_path: str, identifier: str) -> BatchIssue:
    markdown = b"# Article\nThe original newspaper text.\n"
    metadata = json.dumps({"Record": {"source": "Paper", "identifier": identifier}}).encode("utf-8")
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(markdown)
    metadata_path = path.with_name(path.stem + "_metadata.json")
    metadata_path.write_bytes(metadata)
    return BatchIssue(
        relative_path=relative_path, markdown_path=relative_path,
        metadata_path=metadata_path.relative_to(root).as_posix(),
        markdown_sha256=hashlib.sha256(markdown).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        source_identifier=identifier,
    )


@pytest.mark.parametrize("old_identifier,incoming_identifier", [
    ("GF-URN:example", "GF-URN:example"),
    ("URN:NBN:SI:doc-EXAMPLE", "urn:nbn:si:DOC-example"),
])
@pytest.mark.parametrize("adopted", [False, True], ids=["schema13-read-only-fallback", "adopted-registry"])
def test_adopted_legacy_source_identifier_cannot_reappear_at_new_path(
    tmp_path: Path, old_identifier: str, incoming_identifier: str, adopted: bool,
) -> None:
    historical_source = tmp_path / "legacy-input"
    database = tmp_path / "indexes" / "semora.sqlite"
    _write_issue(historical_source, "Paper/original.md", old_identifier)
    ingest_corpus(historical_source, database, stages=("newspapers", "articles"))
    if adopted:
        adopt_baseline(database)
    else:
        with closing(sqlite3.connect(database)) as connection, connection:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'pipeline_%' ORDER BY rowid DESC"
            )]
            for table in tables:
                connection.execute(f'DROP TABLE "{table}"')
            connection.execute("DELETE FROM schema_migrations WHERE version>13")
    incoming_source = tmp_path / "new-input"
    issue = _write_issue(incoming_source, "Alias/duplicate.md", incoming_identifier)
    manifest = BatchManifest("new-batch", (issue,))
    before = database.read_bytes()

    plan = plan_batch(manifest, incoming_source, database)

    assert plan["counts"]["conflict"] == 1
    assert database.read_bytes() == before
    with pytest.raises(ValueError):
        run_batch(manifest, incoming_source, database, stages=("newspapers",))
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM newspapers").fetchone()[0] == 1
        if adopted:
            assert connection.execute("SELECT COUNT(*) FROM pipeline_batches").fetchone()[0] == 0
