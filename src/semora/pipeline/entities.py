"""Append entity names and article memberships to an adopted dictionary."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import ExitStack, closing
from pathlib import Path

from semora.ner.names import normalize_name
from semora.ner.spellfix import extension_path, load_spellfix
from semora.pipeline.entity_baseline import (
    NORMALIZATION,
    _manifest,
    expected_names_sha256,
    normalized_mentions,
)
from semora.pipeline.ner import _recorded_identity, _validate_batch
from semora.pipeline.publication_state import invalidate_publication
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expected(source: sqlite3.Connection, row: sqlite3.Row, run_id: str) -> tuple[dict[str, int], dict[str, str]]:
    text = row["content"]
    if row["content_sha256"] != _hash(text):
        raise ValueError(f"NER source hash changed for article {row['article_id']}.")
    mentions = source.execute(
        "SELECT entity_index,text,label,char_start,char_end,score FROM article_entities "
        "WHERE ner_run_id=? AND article_id=? ORDER BY entity_index",
        (run_id, row["article_id"]),
    ).fetchall()
    if len(mentions) != row["entity_count"]:
        raise ValueError(f"NER mention count changed for article {row['article_id']}.")
    for index, mention in enumerate(mentions):
        start, end = mention["char_start"], mention["char_end"]
        score = mention["score"]
        if (
            mention["entity_index"] != index
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or text[start:end] != mention["text"]
            or not isinstance(mention["label"], str)
            or not mention["label"].strip()
            or score is not None
            and (not isinstance(score, (int, float)) or not math.isfinite(score))
        ):
            raise ValueError(f"NER mention no longer matches article {row['article_id']}.")
    names = normalized_mentions(mentions)
    displays: dict[str, str] = {}
    for mention in mentions:
        name = normalize_name(mention["text"])
        if name and name not in displays:
            displays[name] = mention["text"]
    return names, displays


def _memberships(target: sqlite3.Connection, article_id: str) -> dict[str, tuple[int, str]]:
    rows = target.execute(
        "SELECT n.normalized_name,m.char_start,m.content_sha256 FROM entity_articles m "
        "JOIN names n USING(entity_id) WHERE m.article_id=?",
        (article_id,),
    )
    return {row["normalized_name"]: (row["char_start"], row["content_sha256"]) for row in rows}


def _checkpoint(
    target: sqlite3.Connection,
    rows: list[sqlite3.Row],
    expected: list[tuple[dict[str, int], dict[str, str]]],
    previous_token: str,
    next_token: str,
) -> tuple[int, int, int]:
    added_articles = added_memberships = added_names = 0
    with target:
        target.execute("BEGIN IMMEDIATE")
        profile = target.execute(
            "SELECT advancement_token FROM pipeline_entity_profile WHERE profile_id=1"
        ).fetchone()
        if profile is None or profile[0] != previous_token:
            raise ValueError("Entity dictionary advancement changed during checkpoint.")
        for row, (names, displays) in zip(rows, expected, strict=True):
            article_id, content_hash = row["article_id"], row["content_sha256"]
            digest = expected_names_sha256(names)
            previous = target.execute(
                "SELECT content_sha256,expected_names_sha256,name_count "
                "FROM pipeline_entity_articles WHERE article_id=?",
                (article_id,),
            ).fetchone()
            existing = _memberships(target, article_id)
            if previous is not None and (
                previous["content_sha256"] != content_hash
                or previous["expected_names_sha256"] != digest
                or previous["name_count"] != len(names)
            ):
                raise ValueError(f"Dictionary completion changed for article {article_id}.")
            if any(
                existing_name not in names or existing[existing_name] != (names[existing_name], content_hash)
                for existing_name in existing
            ):
                raise ValueError(f"Dictionary has incompatible memberships for article {article_id}.")
            if previous is not None:
                if len(existing) != len(names):
                    raise ValueError(f"Completed dictionary article {article_id} has missing memberships.")
                continue
            for name, start in sorted(names.items()):
                if name in existing:
                    continue
                inserted = target.execute(
                    "INSERT OR IGNORE INTO names(normalized_name,name) VALUES (?,?)",
                    (name, displays[name]),
                ).rowcount
                if inserted:
                    added_names += 1
                saved = target.execute(
                    "SELECT entity_id,article_count FROM names WHERE normalized_name=?",
                    (name,),
                ).fetchone()
                if saved is None or type(saved["article_count"]) is not int or saved["article_count"] < 0:
                    raise ValueError("Entity name count is invalid.")
                entity_id = saved["entity_id"]
                target.execute(
                    "INSERT INTO entity_articles(entity_id,article_id,char_start,content_sha256) VALUES (?,?,?,?)",
                    (entity_id, article_id, start, content_hash),
                )
                new_count = saved["article_count"] + 1
                target.execute("UPDATE names SET article_count=? WHERE entity_id=?", (new_count, entity_id))
                spell = target.execute("SELECT word,rank FROM name_spellfix WHERE rowid=?", (entity_id,)).fetchone()
                if spell is None:
                    if not inserted:
                        raise ValueError("Existing entity name is missing from spellfix.")
                    target.execute(
                        "INSERT INTO name_spellfix(rowid,word,rank) VALUES (?,?,?)",
                        (entity_id, name, new_count),
                    )
                elif spell["word"] != name or spell["rank"] != saved["article_count"]:
                    raise ValueError("Spellfix rank/word disagrees with the entity name before update.")
                else:
                    target.execute("UPDATE name_spellfix SET rank=? WHERE rowid=?", (new_count, entity_id))
                added_memberships += 1
            target.execute(
                "INSERT INTO pipeline_entity_articles(article_id,content_sha256,expected_names_sha256,name_count) "
                "VALUES (?,?,?,?)",
                (article_id, content_hash, digest, len(names)),
            )
            added_articles += 1
        if added_articles:
            target.execute(
                "UPDATE pipeline_entity_profile SET advancement_token=? WHERE profile_id=1",
                (next_token,),
            )
    return added_articles, added_memberships, added_names


def index_batch_entities(
    database_path: str | Path,
    batch_id: str,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    batch_articles: int = 1000,
    spellfix_path: str | Path | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Extend an adopted dictionary in place during a stopped-reader window.

    Per-article output and its completion ledger share a dictionary transaction.
    The serving manifest remains unchanged until separate publication checks.
    """
    if type(batch_articles) is not int or batch_articles < 1 or not batch_id or not run_id:
        raise ValueError("Supply a batch/run and positive article checkpoint size.")
    source = Path(database_path).resolve(strict=True)
    root = Path(entity_dir).resolve(strict=True)
    _manifest_value, dictionary, _manifest_hash = _manifest(root, run_id)
    if dictionary.samefile(source):
        raise ValueError("Entity dictionary cannot alias its corpus database.")
    library = Path(spellfix_path).resolve(strict=True) if spellfix_path else extension_path(source.parent.parent)
    with ExitStack() as stack:
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        stack.enter_context(SemanticBuildLock(source.with_name(source.name + ".ner.lock"), "NER"))
        stack.enter_context(SemanticBuildLock(root / ".build.lock", "entity dictionary"))
        with closing(open_connection(source)) as corpus, closing(open_connection(dictionary)) as target:
            corpus.execute("BEGIN IMMEDIATE")
            try:
                _validate_batch(corpus, batch_id)
                _recorded_identity(corpus, run_id)
                identity = corpus.execute(
                    "SELECT corpus_id FROM pipeline_corpus_identity WHERE identity_id=1"
                ).fetchone()
                profile = target.execute("SELECT * FROM pipeline_entity_profile WHERE profile_id=1").fetchone()
                if (
                    identity is None
                    or profile is None
                    or (profile["corpus_id"], profile["ner_run_id"], profile["normalization"])
                    != (identity[0], run_id, NORMALIZATION)
                ):
                    raise ValueError("Entity dictionary is not adopted for this corpus/NER profile.")
                binding = corpus.execute(
                    "SELECT * FROM pipeline_entity_bindings WHERE ner_run_id=?", (run_id,)
                ).fetchone()
                if binding is None or (
                    binding["corpus_id"], binding["generation"]
                ) != (identity[0], _manifest_value["generation"]):
                    raise ValueError("Corpus entity binding differs from the dictionary generation.")
                current_token = binding["advancement_token"]
                target_token = profile["advancement_token"]
                pending_token = binding["pending_token"]
                if pending_token is None:
                    if target_token != current_token:
                        raise ValueError("Corpus and dictionary advancement tokens differ; audit before appending.")
                elif target_token == current_token:
                    corpus.execute(
                        "UPDATE pipeline_entity_bindings SET pending_token=NULL WHERE ner_run_id=?", (run_id,)
                    )
                elif target_token == pending_token:
                    current_token = pending_token
                    corpus.execute(
                        "UPDATE pipeline_entity_bindings SET advancement_token=?,pending_token=NULL "
                        "WHERE ner_run_id=?", (current_token, run_id),
                    )
                else:
                    raise ValueError("Interrupted entity checkpoint has inconsistent advancement tokens.")
                if (
                    target.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='pipeline_entity_articles_lookup'"
                    ).fetchone()
                    is None
                ):
                    raise ValueError("Entity article lookup index is missing; audit and repair adoption.")
                load_spellfix(target, library)
                eligible = corpus.execute(
                    "SELECT COUNT(*) FROM articles a JOIN pipeline_batch_issues b USING(newspaper_id) "
                    "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>''",
                    (batch_id,),
                ).fetchone()[0]
                corpus.execute("COMMIT")
                invalidated = False
                cursor = ""
                added_articles = added_memberships = added_names = skipped = 0
                while True:
                    corpus.execute("BEGIN IMMEDIATE")
                    rows = corpus.execute(
                        "SELECT a.article_id,a.content,s.content_sha256,s.entity_count FROM articles a "
                        "JOIN pipeline_batch_issues b USING(newspaper_id) "
                        "LEFT JOIN article_ner_state s ON s.article_id=a.article_id AND s.ner_run_id=? "
                        "WHERE b.batch_id=? AND a.is_valid=1 AND TRIM(a.content)<>'' AND a.article_id>? "
                        "ORDER BY a.article_id LIMIT ?",
                        (run_id, batch_id, cursor, batch_articles),
                    ).fetchall()
                    if not rows:
                        corpus.execute("ROLLBACK")
                        break
                    if any(row["content_sha256"] is None for row in rows):
                        raise ValueError("Finish selected batch NER before updating the entity dictionary.")
                    expected = [_expected(corpus, row, run_id) for row in rows]
                    complete = all(
                        target.execute(
                            "SELECT 1 FROM pipeline_entity_articles WHERE article_id=?", (row["article_id"],)
                        ).fetchone() is not None for row in rows
                    )
                    corpus.execute("COMMIT")
                    if complete:
                        new_articles, new_memberships, new_names = _checkpoint(
                            target, rows, expected, current_token, current_token
                        )
                    else:
                        if not invalidated:
                            invalidate_publication(source)
                            invalidated = True
                        next_token = uuid.uuid4().hex
                        corpus.execute("BEGIN IMMEDIATE")
                        corpus.execute(
                            "UPDATE pipeline_entity_bindings SET pending_token=? WHERE ner_run_id=? "
                            "AND advancement_token=? AND pending_token IS NULL",
                            (next_token, run_id, current_token),
                        )
                        if corpus.execute("SELECT changes()").fetchone()[0] != 1:
                            raise ValueError("Corpus entity advancement changed during checkpoint.")
                        corpus.execute("COMMIT")
                        new_articles, new_memberships, new_names = _checkpoint(
                            target, rows, expected, current_token, next_token
                        )
                        corpus.execute("BEGIN IMMEDIATE")
                        corpus.execute(
                            "UPDATE pipeline_entity_bindings SET advancement_token=?,pending_token=NULL "
                            "WHERE ner_run_id=? AND advancement_token=? AND pending_token=?",
                            (next_token, run_id, current_token, next_token),
                        )
                        if corpus.execute("SELECT changes()").fetchone()[0] != 1:
                            raise ValueError("Corpus entity checkpoint finalization changed unexpectedly.")
                        corpus.execute("COMMIT")
                        current_token = next_token
                    added_articles += new_articles
                    added_memberships += new_memberships
                    added_names += new_names
                    skipped += len(rows) - new_articles
                    cursor = rows[-1]["article_id"]
                    if progress:
                        progress(
                            {
                                "event": "entity_checkpoint",
                                "batch_id": batch_id,
                                "completed_articles": added_articles + skipped,
                                "total": eligible,
                            }
                        )
                return {
                    "batch_id": batch_id,
                    "stage": "entities",
                    "ner_run_id": run_id,
                    "eligible_articles": eligible,
                    "completed_articles": added_articles + skipped,
                    "added_articles": added_articles,
                    "skipped_articles": skipped,
                    "added_memberships": added_memberships,
                    "added_names": added_names,
                    "complete": added_articles + skipped == eligible,
                    "retrieval_ready": False,
                    "publication_ready": False,
                }
            finally:
                if corpus.in_transaction:
                    corpus.execute("ROLLBACK")
