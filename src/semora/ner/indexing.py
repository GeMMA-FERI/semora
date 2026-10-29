"""Resumable NER extraction with bounded reads, inference and atomic writes."""

import json
import time
import uuid
from contextlib import closing
from pathlib import Path

from tqdm import tqdm

from semora.pipeline.guards import reject_pipeline_batches
from semora.retrieval.semantic_lock import SemanticBuildLock
from semora.storage import Database
from semora.storage.connection import open_connection
from semora.storage.models import Run

from .adapters import load_adapter
from .staging import iter_predictions
from .storage import completed_count, register_run, save_batch


def build_ner_index(
    database_path, *, run_id="classla-sl", adapter="classla", device="cuda",
    batch_articles=32, batch_size=16, pipeline_depth=3, max_articles=None,
    tokenizer_workers=2, ner_pipeline_depth=1, ner_decoder="numpy",
    resources=None, model=None,
):
    """Store valid article entities. max_articles is a total run target, not an increment.

    model is an injected adapter for applications/tests; CLI loads the shared adapter.
    Only completed batches are skipped. A failed batch is recomputed on resume.
    """
    reject_pipeline_batches(database_path)
    path = Path(database_path).resolve(strict=True)
    if not run_id.strip():
        raise ValueError("NER run ID cannot be empty")
    if min(batch_articles, batch_size, pipeline_depth, ner_pipeline_depth) < 1 or tokenizer_workers < 0:
        raise ValueError("Batch sizes/depths must be positive and tokenizer workers nonnegative")
    if max_articles is not None and max_articles < 1:
        raise ValueError("max_articles must be positive")
    with SemanticBuildLock(path.with_name(path.name + ".ner.lock"), "NER"):
        database = Database(path)
        attempt = "index-ner-" + uuid.uuid4().hex
        started = time.perf_counter()
        logged = False
        try:
            database.initialize()
            database.insert_run(Run(run_id=attempt, run_type="index_ner"))
            database.log(attempt, "INFO", json.dumps({"event": "index_started", "ner_run_id": run_id}))
            logged = True
            if model is None:
                model = load_adapter(
                    adapter, device=device, resources=resources, tokenizer_workers=tokenizer_workers,
                    ner_pipeline_depth=ner_pipeline_depth, ner_decoder=ner_decoder,
                    sentence_order="input",
                )
            config = {"model": model.metadata, "device": device,
                      "batch_articles": batch_articles, "batch_size": batch_size,
                      "ner_decoder": ner_decoder, "sentence_order": "input"}
            register_run(database.conn, run_id, config)
            initial = completed_count(database.conn, run_id)
            remaining = None if max_articles is None else max(0, max_articles - initial)
            last_rowid = 0
            upper = database.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM articles").fetchone()[0]
            exhausted = False

            def read_batch():
                nonlocal last_rowid, remaining, exhausted
                if exhausted or remaining == 0:
                    return []
                limit = batch_articles if remaining is None else min(batch_articles, remaining)
                # Keyset iteration, no OFFSET and no up-front scan of corpus content.
                # Each preparation callback owns a short-lived read connection.
                with closing(open_connection(path, read_only=True)) as reader:
                    rows = [dict(row) for row in reader.execute(
                        "SELECT a.rowid AS source_rowid, a.article_id, a.content FROM articles a "
                        "WHERE a.rowid > ? AND a.rowid <= ? AND a.is_valid = 1 "
                        "AND NOT EXISTS (SELECT 1 FROM article_ner_state s "
                        "WHERE s.ner_run_id = ? AND s.article_id = a.article_id) "
                        "AND TRIM(a.content) <> '' ORDER BY a.rowid LIMIT ?", (last_rowid, upper, run_id, limit),
                    )]
                if rows:
                    last_rowid = rows[-1]["source_rowid"]
                    if remaining is not None:
                        remaining -= len(rows)
                if len(rows) < limit:
                    exhausted = True
                return rows

            added = mentions = 0
            with closing(iter_predictions(model, read_batch, batch_size, pipeline_depth)) as batches:
                with tqdm(initial=initial, total=max_articles, desc="Extracting NER", unit="article") as progress:
                    for batch in batches:
                        save_batch(database.conn, run_id, batch)
                        added += len(batch.rows)
                        mentions += sum(map(len, batch.prediction.entities))
                        progress.update(len(batch.rows))
            result = {"ner_run_id": run_id, "processed_articles": initial + added,
                      "added_articles": added, "added_entities": mentions,
                      "exhausted_snapshot": exhausted, "elapsed_seconds": time.perf_counter() - started}
            database.log(attempt, "INFO", json.dumps({"event": "index_finished", **result}))
            return result
        except BaseException as error:
            if logged:
                try:
                    database.log(attempt, "ERROR", json.dumps({
                        "event": "index_failed", "ner_run_id": run_id,
                        "error": f"{type(error).__name__}: {error}",
                    }))
                except Exception:
                    pass  # Preserve the original failure if logging is unavailable.
            raise
        finally:
            try:
                if model is not None and hasattr(model, "close"):
                    model.close()
            finally:
                database.close()
