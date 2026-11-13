"""Offline frequency order for immutable entity dictionaries; never a discovery sort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from semora.retrieval.semantic_lock import SemanticBuildLock

FORMAT = "semora.curation.frequency.v1"
PAGE_SQL = "SELECT rank,entity_id,normalized_name,article_count FROM ranked_names WHERE rank>? ORDER BY rank LIMIT ?"


class FrequencyNotReady(ValueError):
    """An operator must prepare this generation before automatic discovery."""


def frequency_identity(generation: str, advancement_token: str = "") -> str:
    return hashlib.sha256(json.dumps([generation, advancement_token], separators=(",", ":")).encode()).hexdigest()


def frequency_path(folder: Path, identity: str) -> Path:
    return folder / "curation-frequency" / f"{identity}.sqlite"


def dictionary_advancement_token(connection: sqlite3.Connection) -> str:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_entity_profile'"
    ).fetchone():
        row = connection.execute("SELECT advancement_token FROM pipeline_entity_profile WHERE profile_id=1").fetchone()
        if row is None or not isinstance(row[0], str):
            raise ValueError("Entity dictionary has no valid advancement identity.")
        return row[0]
    return ""


def _open(path: Path, identity: str) -> sqlite3.Connection:
    if not path.is_file():
        raise FrequencyNotReady(
            "Common-first discovery is not ready. Run python -m semora.curation.frequency --index-dir INDEXES."
        )
    if path.is_symlink():
        raise ValueError("Frequency order must be a regular artifact file.")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = connection.execute("SELECT format,identity,total FROM metadata").fetchmany(2)
        if len(metadata) != 1 or metadata[0]["format"] != FORMAT or metadata[0]["identity"] != identity:
            raise ValueError("Frequency order does not match the published entity dictionary.")
        total = metadata[0]["total"]
        first = connection.execute("SELECT rank FROM ranked_names ORDER BY rank LIMIT 1").fetchone()
        last = connection.execute("SELECT rank FROM ranked_names ORDER BY rank DESC LIMIT 1").fetchone()
        if (
            type(total) is not int
            or total < 0
            or (
                (total == 0 and (first is not None or last is not None))
                or (total > 0 and (first is None or first[0] != 1 or last[0] != total))
            )
        ):
            raise ValueError("Frequency order is incomplete.")
        return connection
    except BaseException:
        connection.close()
        raise


def frequency_page(entities, after_rank: int, limit: int):
    path = getattr(entities, "frequency_path", None)
    identity = getattr(entities, "frequency_identity", None)
    if path is None or identity is None:
        raise FrequencyNotReady("Common-first discovery requires a prepared entity frequency order.")
    with closing(_open(Path(path), identity)) as connection:
        return connection.execute(PAGE_SQL, (after_rank, limit)).fetchall()


def prepare_frequency(
    source: sqlite3.Connection,
    folder: Path,
    generation: str,
    advancement_token: str = "",
    *,
    batch_size: int = 10000,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Sort name metadata once using a dedicated source connection without TEMP objects.

    Preparation runs outside curation transactions and atomically publishes a separate artifact.
    """
    if type(batch_size) is not int or batch_size < 1 or not generation:
        raise ValueError("Supply a source identity and positive frequency checkpoint size.")
    identity = frequency_identity(generation, advancement_token)
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    output = frequency_path(folder, identity)
    with SemanticBuildLock(output.with_suffix(".lock"), "curation frequency"):
        if dictionary_advancement_token(source) != advancement_token:
            raise ValueError("Entity dictionary advancement changed before frequency preparation.")
        if output.exists():
            with closing(_open(output, identity)) as cached:
                total = cached.execute("SELECT total FROM metadata").fetchone()[0]
            result = {"path": str(output), "identity": identity, "names": total, "reused": True}
            if progress:
                progress({"state": "completed", "completed": total, "total": total})
            return result
        if source.in_transaction:
            raise ValueError(
                "Frequency preparation requires a dedicated source connection without an active transaction."
            )
        # Changing temp_store destroys TEMP objects. Never borrow a caller's
        # scratch tables; the CLI and index builders supply dedicated connections.
        if source.execute("SELECT 1 FROM sqlite_temp_master LIMIT 1").fetchone():
            raise ValueError("Frequency preparation requires a dedicated source connection without TEMP objects.")
        if progress:
            progress({"state": "running", "completed": 0, "total": 0})
        total = source.execute("SELECT COUNT(*) FROM names").fetchone()[0]
        temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
        completed = 0
        if progress:
            progress({"state": "running", "completed": 0, "total": total})
        original_temp_store = source.execute("PRAGMA temp_store").fetchone()[0]
        source.execute("PRAGMA temp_store=FILE")
        try:
            with closing(sqlite3.connect(temporary)) as target:
                target.executescript("""
                    CREATE TABLE ranked_names(rank INTEGER PRIMARY KEY,entity_id INTEGER NOT NULL,
                        normalized_name TEXT NOT NULL,article_count INTEGER NOT NULL CHECK(article_count>=0));
                    CREATE TABLE metadata(format TEXT NOT NULL,identity TEXT NOT NULL,total INTEGER NOT NULL);
                """)
                rows = source.execute(
                    "SELECT entity_id,normalized_name,article_count FROM names ORDER BY article_count DESC,entity_id"
                )
                while batch := rows.fetchmany(batch_size):
                    with target:
                        target.executemany(
                            "INSERT INTO ranked_names VALUES(?,?,?,?)",
                            [(completed + index + 1, row[0], row[1], row[2]) for index, row in enumerate(batch)],
                        )
                    completed += len(batch)
                    if progress:
                        progress({"state": "running", "completed": completed, "total": total})
                if completed != total or dictionary_advancement_token(source) != advancement_token:
                    raise ValueError("Entity dictionary changed during frequency preparation.")
                with target:
                    target.execute("INSERT INTO metadata VALUES(?,?,?)", (FORMAT, identity, total))
            os.replace(temporary, output)
            if progress:
                progress({"state": "completed", "completed": total, "total": total})
            return {"path": str(output), "identity": identity, "names": total, "reused": False}
        except BaseException:
            temporary.unlink(missing_ok=True)
            if progress:
                progress({"state": "failed", "completed": 0, "total": total})
            raise
        finally:
            source.execute(f"PRAGMA temp_store={original_temp_store}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    from semora.retrieval import SearchEngine

    def report(value):
        if args.progress_file is None:
            return
        path = args.progress_file
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "format": "pebble.progress.v1",
            "title": "Prepare common entity order",
            "unit": "names",
            "message": "Building an auxiliary index; archive and curation groups stay unchanged.",
            **value,
        }
        temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        temporary.write_text(json.dumps(value), encoding="utf-8")
        os.replace(temporary, path)

    engine = SearchEngine(args.index_dir / "semora.sqlite")
    try:
        entities = engine._entities()
        result = prepare_frequency(
            entities.connection,
            args.index_dir / "entities",
            entities.dictionary_generation,
            entities.advancement_token,
            progress=report,
        )
        print(json.dumps(result))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
