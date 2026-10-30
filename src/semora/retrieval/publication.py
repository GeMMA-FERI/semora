"""Read one atomic serving revision and reject stale mutable SQLite stores."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

FORMAT = "semora.pipeline.publication.v1"


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate publication key: {key}")
        result[key] = value
    return result


def read_publication(connection: sqlite3.Connection, database_path: Path) -> dict | None:
    """Pin a serving pointer for the lifetime of one search engine."""
    root = database_path.resolve().parent
    pointer = root / "publication.json"
    if pointer.is_symlink():
        raise ValueError("Serving publication pointer must be a regular file.")
    state_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_publication_state'"
    ).fetchone()
    epoch = None
    if state_table is not None:
        rows = connection.execute("SELECT epoch FROM pipeline_publication_state").fetchmany(2)
        if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] < 0:
            raise ValueError("Corpus publication epoch is invalid.")
        epoch = rows[0][0]
    if not pointer.exists():
        if epoch not in (None, 0):
            raise ValueError(
                "Corpus changed without a serving publication; finish publication before restarting readers."
            )
        return None
    if epoch is None or not pointer.is_file() or pointer.stat().st_size > 1024 * 1024:
        raise ValueError("Serving publication requires a migrated corpus and bounded pointer file.")
    value = json.loads(pointer.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {
        "format", "publication_id", "corpus_id", "epoch", "semantic", "entities"
    } or value["format"] != FORMAT or not _hex(value["publication_id"], 32) or not _hex(value["corpus_id"], 32):
        raise ValueError("Serving publication has an invalid format or identity.")
    if type(value["epoch"]) is not int or value["epoch"] != epoch:
        raise ValueError("Serving publication is stale for the corpus epoch.")
    identity = connection.execute(
        "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
    ).fetchone()
    if identity is None or identity[0] != value["corpus_id"]:
        raise ValueError("Serving publication belongs to a different corpus.")
    semantic, entities = value["semantic"], value["entities"]
    if (
        not isinstance(semantic, dict)
        or set(semantic) != {"generation", "chunks", "index_sha256"}
        or not _hex(semantic["generation"], 64)
        or type(semantic["chunks"]) is not int
        or semantic["chunks"] < 0
        or not _hex(semantic["index_sha256"], 64)
        or not isinstance(entities, dict)
        or set(entities) != {"generation", "ner_run_id", "manifest_sha256"}
        or not _hex(entities["generation"], 32)
        or not isinstance(entities["ner_run_id"], str)
        or not entities["ner_run_id"]
        or not _hex(entities["manifest_sha256"], 64)
    ):
        raise ValueError("Serving publication has invalid index references.")
    return value
