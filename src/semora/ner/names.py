"""Rebuildable, label-independent entity-name dictionary and spellfix index."""

import json
import os
import sqlite3
import unicodedata
import uuid
from contextlib import closing
from pathlib import Path

from tqdm import tqdm

from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection

from .spellfix import extension_path, load_spellfix


def normalize_name(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def build_entity_names(root, run_id="classla-sl", batch_size=10000):
    """Build off to the side; expose only a complete index via an atomic manifest."""
    root = Path(root).resolve()
    folder = root / "indexes" / "entities"
    folder.mkdir(parents=True, exist_ok=True)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generation = uuid.uuid4().hex
    output = folder / f"{generation}.sqlite"
    with SemanticBuildLock(folder / ".build.lock", "entity dictionary"):
        try:
            with closing(open_connection(root / "indexes" / "semora.sqlite", read_only=True)) as source:
                if source.execute("SELECT 1 FROM ner_runs WHERE ner_run_id=?", (run_id,)).fetchone() is None:
                    raise ValueError(f"Unknown NER run: {run_id}")
                with closing(sqlite3.connect(output)) as target:
                    load_spellfix(target, extension_path(root))
                    target.executescript("""
                        CREATE TABLE names (
                            entity_id INTEGER PRIMARY KEY, normalized_name TEXT NOT NULL UNIQUE,
                            name TEXT NOT NULL, article_count INTEGER NOT NULL DEFAULT 0
                        );
                        CREATE TABLE entity_articles (
                            entity_id INTEGER NOT NULL, article_id TEXT NOT NULL,
                            char_start INTEGER NOT NULL, content_sha256 TEXT NOT NULL,
                            PRIMARY KEY(entity_id, article_id)
                        );
                        CREATE TABLE metadata (generation TEXT, ner_run_id TEXT);
                        CREATE VIRTUAL TABLE name_spellfix USING spellfix1;
                    """)
                    target.execute("INSERT INTO metadata VALUES (?,?)", (generation, run_id))
                    target.commit()
                    # One SQLite read snapshot; only distinct name/article pairs are materialized.
                    cursor = source.execute(
                        "SELECT e.article_id,e.text,e.char_start,s.content_sha256 "
                        "FROM article_entities e JOIN article_ner_state s "
                        "ON s.ner_run_id=e.ner_run_id AND s.article_id=e.article_id "
                        "JOIN articles a ON a.article_id=e.article_id "
                        "WHERE e.ner_run_id=? AND a.is_valid=1", (run_id,),
                    )
                    with tqdm(desc="Indexing entity names", unit="mention") as progress:
                        while rows := cursor.fetchmany(batch_size):
                            with target:
                                cache = {}
                                for row in rows:
                                    name = normalize_name(row["text"])
                                    if not name:
                                        continue
                                    if name not in cache:
                                        target.execute(
                                            "INSERT OR IGNORE INTO names(normalized_name,name) VALUES (?,?)",
                                            (name, row["text"]),
                                        )
                                        cache[name] = target.execute(
                                            "SELECT entity_id FROM names WHERE normalized_name=?", (name,)
                                        ).fetchone()[0]
                                    target.execute(
                                        "INSERT INTO entity_articles VALUES (?,?,?,?) "
                                        "ON CONFLICT(entity_id,article_id) DO UPDATE SET "
                                        "char_start=MIN(char_start,excluded.char_start)",
                                        (cache[name], row["article_id"], row["char_start"], row["content_sha256"]),
                                    )
                            progress.update(len(rows))
                    with target:
                        target.execute("UPDATE names SET article_count=(SELECT COUNT(*) FROM entity_articles a "
                                       "WHERE a.entity_id=names.entity_id)")
                        target.execute("INSERT INTO name_spellfix(rowid,word,rank) "
                                       "SELECT entity_id,normalized_name,article_count FROM names")
                    count = target.execute("SELECT COUNT(*) FROM names").fetchone()[0]
            manifest = {"generation": generation, "ner_run_id": run_id, "names": count}
            temporary = folder / "manifest.tmp"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            os.replace(temporary, folder / "manifest.json")
            return manifest
        except BaseException:
            output.unlink(missing_ok=True)
            raise
