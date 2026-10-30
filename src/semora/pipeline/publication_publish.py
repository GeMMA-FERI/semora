"""Atomically select a fully audited additive serving revision."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import ExitStack, closing
from pathlib import Path

from semora.pipeline.publication_audit import audit_publication
from semora.retrieval.publication import read_publication
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage.connection import open_connection


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def publish_batch(
    database_path: str | Path,
    batch_id: str,
    semantic_dir: str | Path,
    semantic_generation: str,
    entity_dir: str | Path,
    *,
    run_id: str = "classla-sl",
    spellfix_path: str | Path | None = None,
    faiss_module=None,
    progress=None,
) -> dict:
    """Repeat the complete gate under writer exclusion and replace one pointer.

    Readers and legacy writers must already be stopped. The pointer is the last
    change in this operation; a failed audit leaves the previous file in place.
    A previously invalidated epoch prevents old readers from reopening it.
    """
    source = Path(database_path).resolve(strict=True)
    semantic = Path(semantic_dir).resolve(strict=True)
    entity = Path(entity_dir).resolve(strict=True)
    pointer = source.parent / "publication.json"
    with ExitStack() as locks:
        locks.enter_context(SemanticBuildLock(source.with_name(source.name + ".pipeline.lock"), "pipeline"))
        locks.enter_context(SemanticBuildLock(source.with_name(source.name + ".ner.lock"), "NER"))
        locks.enter_context(SemanticBuildLock(semantic / ".vectors.lock", "vector"))
        locks.enter_context(SemanticBuildLock(entity / ".build.lock", "entity dictionary"))
        report = audit_publication(
            source,
            batch_id,
            semantic,
            semantic_generation,
            entity,
            run_id=run_id,
            spellfix_path=spellfix_path,
            faiss_module=faiss_module,
            progress=progress,
        )
        if not report["eligible"] or report["candidate"] is None:
            return {**report, "published": False, "already_published": False, "restart_readers": False}
        candidate = report["candidate"]
        with closing(open_connection(source)) as corpus:
            corpus.execute("BEGIN IMMEDIATE")
            current_epoch = corpus.execute(
                "SELECT epoch FROM pipeline_publication_state WHERE state_id=1"
            ).fetchone()
            if current_epoch is None or current_epoch[0] != candidate["epoch"]:
                raise ValueError("Corpus changed during publication audit; retry under maintenance.")
            if pointer.is_symlink():
                raise ValueError("Serving publication pointer must not be a symlink.")
            if pointer.exists():
                try:
                    existing = read_publication(corpus, source)
                except (ValueError, OSError, json.JSONDecodeError):
                    existing = None  # An invalid old pointer is replaced only after the full audit.
                if existing is not None and {
                    key: value for key, value in existing.items() if key != "publication_id"
                } == candidate:
                    corpus.execute("ROLLBACK")
                    return {
                        **report,
                        "publication_id": existing["publication_id"],
                        "published": True,
                        "publication_ready": True,
                        "already_published": True,
                        "restart_readers": True,
                    }
            value = {**candidate, "publication_id": uuid.uuid4().hex}
            temporary = pointer.with_name(f".publication-{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, pointer)
                _sync_directory(pointer.parent)
            finally:
                temporary.unlink(missing_ok=True)
                corpus.execute("ROLLBACK")
        return {
            **report,
            "publication_id": value["publication_id"],
            "published": True,
            "publication_ready": True,
            "already_published": False,
            "restart_readers": True,
        }
