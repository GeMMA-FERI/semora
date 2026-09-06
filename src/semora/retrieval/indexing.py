"""Persistent lexical and semantic index builders."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from tqdm import tqdm

from semora.corpus import DEFAULT_MODEL_ID
from semora.storage import Database
from semora.text.lemmatization import (
    ClasslaLemmatizer,
    LemmatizationProfile,
    Lemmatizer,
    LemmaToken,
)


@dataclass(frozen=True)
class LemmaIndexStats:
    surface_articles: int
    processed_articles: int
    indexed_articles: int
    complete: bool


@dataclass(frozen=True)
class _WorkerAnnotations:
    documents: list[list[LemmaToken]]
    profile: LemmatizationProfile | None
    initialization_seconds: float
    processing_seconds: float


@dataclass(frozen=True)
class _LemmaArticle:
    article_id: str
    fts_id: int
    title: str
    content: str


@dataclass(frozen=True)
class _LemmaBatch:
    articles: list[_LemmaArticle]
    fetch_seconds: float
    processed_articles: int

    @property
    def last_article_id(self) -> str:
        return self.articles[-1].article_id


@dataclass(frozen=True)
class _AnnotationJob:
    batch: _LemmaBatch
    futures: tuple[Future[_WorkerAnnotations], ...]


@dataclass(frozen=True)
class _LemmaWriteResult:
    processed_articles: int
    indexed_articles: int
    added_articles: int
    mapping_seconds: float
    write_seconds: float


@dataclass(frozen=True)
class _PendingLemmaWrite:
    future: Future[_LemmaWriteResult]
    fetch_seconds: float
    worker_profiles: list[_WorkerAnnotations]


_INDEX_WORKER_LEMMATIZER: ClasslaLemmatizer | None = None
_INDEX_WORKER_INITIALIZATION_SECONDS = 0.0
_INDEX_WORKER_REPORT_INITIALIZATION = False
_LEMMA_WRITER_DATABASE: Database | None = None


def build_bm25_index(
    database_path: str | Path = "indexes/semora.sqlite",
    *,
    max_articles: int | None = None,
    batch_size: int = 10_000,
    rebuild: bool = False,
) -> int:
    """Build or resume the contentless BM25 index up to a total article target."""
    if max_articles is not None and max_articles < 0:
        raise ValueError("max_articles must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    database = Database(database_path)
    try:
        database.initialize()
        if rebuild:
            with database.conn:
                database.conn.execute("INSERT INTO article_fts(article_fts) VALUES('delete-all')")
                database.conn.execute("DELETE FROM article_fts_map")
                database.conn.execute("DELETE FROM article_fts_state")
                database.conn.execute(
                    "INSERT INTO article_lemma_fts(article_lemma_fts) VALUES('delete-all')"
                )
                database.conn.execute("DELETE FROM article_lemma_index_state")
        state = database.conn.execute(
            "SELECT * FROM article_fts_state WHERE state_id = 1"
        ).fetchone()
        if state is None:
            with database.conn:
                database.conn.execute("INSERT INTO article_fts_state (state_id) VALUES (1)")
            indexed = 0
            last_article_id = ""
            complete = False
        else:
            indexed = int(state["indexed_articles"])
            last_article_id = str(state["last_article_id"])
            complete = bool(state["complete"])
        next_fts_id = indexed + 1
        available = int(
            database.conn.execute(
                """
                SELECT COUNT(*)
                FROM articles
                WHERE articles.is_valid = 1
                """
            ).fetchone()[0]
        )
        target = available if max_articles is None else min(max_articles, available)
        if complete or indexed >= target:
            return indexed
        with tqdm(
            total=max(indexed, target),
            initial=indexed,
            desc="Indexing surface BM25",
            unit="article",
            dynamic_ncols=True,
        ) as progress:
            while max_articles is None or indexed < max_articles:
                limit = batch_size if max_articles is None else min(batch_size, max_articles - indexed)
                rows = database.conn.execute(
                    """
                    SELECT
                        articles.article_id,
                        COALESCE(articles.title, '') AS title,
                        articles.content AS text
                    FROM articles
                    WHERE articles.is_valid = 1
                      AND articles.article_id > ?
                    ORDER BY articles.article_id
                    LIMIT ?
                    """,
                    (last_article_id, limit),
                ).fetchall()
                if not rows:
                    break
                mapping = [
                    (
                        next_fts_id + offset,
                        str(row["article_id"]),
                    )
                    for offset, row in enumerate(rows)
                ]
                documents = [
                    (next_fts_id + offset, str(row["title"]), str(row["text"]))
                    for offset, row in enumerate(rows)
                ]
                with database.conn:
                    database.conn.executemany(
                        """
                        INSERT INTO article_fts_map (fts_id, article_id)
                        VALUES (?, ?)
                        """,
                        mapping,
                    )
                    database.conn.executemany(
                        "INSERT INTO article_fts (rowid, title, text) VALUES (?, ?, ?)",
                        documents,
                    )
                    database.conn.execute(
                        """
                        UPDATE article_fts_state
                        SET last_article_id = ?, indexed_articles = ?, complete = 0,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE state_id = 1
                        """,
                        (str(rows[-1]["article_id"]), indexed + len(rows)),
                    )
                added = len(rows)
                indexed += added
                next_fts_id += added
                last_article_id = str(rows[-1]["article_id"])
                progress.update(added)
            if indexed == available:
                with database.conn:
                    database.conn.execute(
                        "UPDATE article_fts_state SET complete = 1, updated_at = CURRENT_TIMESTAMP WHERE state_id = 1"
                    )
        return indexed
    finally:
        database.close()


def build_lemma_index(
    database_path: str | Path = "indexes/semora.sqlite",
    *,
    max_articles: int | None = None,
    batch_articles: int = 50,
    rebuild: bool = False,
    pipeline_type: str = "default",
    device: str = "auto",
    resources_dir: str | Path | None = None,
    pos_batch_size: int | None = None,
    lemma_batch_size: int | None = None,
    profile: bool = False,
    workers: int = 1,
    pipeline_depth: int = 3,
    tokenizer_workers: int | None = None,
    lemmatizer: Lemmatizer | None = None,
) -> LemmaIndexStats:
    """Lemmatize and index each article present in the surface BM25 index."""
    if max_articles is not None and max_articles < 0:
        raise ValueError("max_articles must be non-negative.")
    if batch_articles <= 0:
        raise ValueError("batch_articles must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if pipeline_depth <= 0:
        raise ValueError("pipeline_depth must be positive.")
    if tokenizer_workers is not None and tokenizer_workers < 0:
        raise ValueError("tokenizer_workers cannot be negative.")
    if pipeline_depth == 1 and tokenizer_workers not in (None, 0):
        raise ValueError("tokenizer_workers requires pipeline_depth greater than 1.")
    if workers > 1 and tokenizer_workers not in (None, 0):
        raise ValueError("tokenizer_workers is only used with one CLASSLA worker.")
    if workers > 1 and lemmatizer is not None:
        raise ValueError("A custom lemmatizer can only be used with one worker.")
    database = Database(database_path)
    process_executor: ProcessPoolExecutor | None = None
    local_executor: ThreadPoolExecutor | None = None
    writer_executor: ThreadPoolExecutor | None = None
    active_lemmatizer = lemmatizer
    try:
        database.initialize()
        if str(database_path) != ":memory:":
            database.conn.execute("PRAGMA journal_mode = WAL")
        if rebuild:
            with database.conn:
                database.conn.execute(
                    "INSERT INTO article_lemma_fts(article_lemma_fts) VALUES('delete-all')"
                )
                database.conn.execute("DELETE FROM article_lemma_index_state")
        surface_articles = int(
            database.conn.execute("SELECT COUNT(*) FROM article_fts_map").fetchone()[0]
        )
        if surface_articles == 0:
            raise ValueError("Build the surface BM25 index before building the lemma index.")
        state = database.conn.execute(
            "SELECT * FROM article_lemma_index_state WHERE state_id = 1"
        ).fetchone()
        if state is None:
            with database.conn:
                database.conn.execute(
                    """
                    INSERT INTO article_lemma_index_state
                        (state_id, surface_articles, pipeline_type)
                    VALUES (1, ?, ?)
                    """,
                    (surface_articles, pipeline_type),
                )
            last_article_id = ""
            processed_articles = indexed_articles = 0
            complete = False
        else:
            if int(state["surface_articles"]) != surface_articles:
                raise ValueError("The surface BM25 sample changed; rebuild the lemma index with --rebuild.")
            if str(state["pipeline_type"]) != pipeline_type:
                raise ValueError("The CLASSLA pipeline type changed; rebuild the lemma index with --rebuild.")
            last_article_id = str(state["last_article_id"])
            processed_articles = int(state["processed_articles"])
            indexed_articles = int(state["indexed_articles"])
            complete = bool(state["complete"])
        if complete or (max_articles is not None and processed_articles >= max_articles):
            return LemmaIndexStats(surface_articles, processed_articles, indexed_articles, complete)

        worker_config: dict[str, Any] = {
            "pipeline_type": pipeline_type,
            "device": device,
            "resources_dir": str(resources_dir) if resources_dir is not None else None,
            "pos_batch_size": pos_batch_size,
            "lemma_batch_size": lemma_batch_size,
        }
        if workers > 1:
            print(f"Starting {workers} CLASSLA worker processes...", file=sys.stderr)
            process_executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_index_worker,
                initargs=(worker_config,),
            )
        concurrent_batches = workers if workers > 1 else pipeline_depth
        if lemmatizer is not None:
            concurrent_batches = 1
        scheduled_articles = processed_articles
        scheduled_article_id = last_article_id
        exhausted = False

        with tqdm(
            total=max(processed_articles, max_articles) if max_articles is not None else None,
            initial=processed_articles,
            desc="Indexing lemma BM25",
            unit="article",
            dynamic_ncols=True,
        ) as progress:
            progress.set_postfix(indexed_articles=f"{indexed_articles:,}")
            first_limit = _lemma_fetch_limit(
                batch_articles,
                concurrent_batches,
                max_articles,
                scheduled_articles,
            )
            current_batch = _fetch_lemma_batch(
                database,
                scheduled_article_id,
                first_limit,
                scheduled_articles,
            )
            if current_batch is None:
                exhausted = max_articles is None or scheduled_articles < max_articles
            else:
                scheduled_articles = current_batch.processed_articles
                scheduled_article_id = current_batch.last_article_id

            if current_batch is not None and active_lemmatizer is None and process_executor is None:
                print("Loading CLASSLA pipeline...", file=sys.stderr)
                load_started = time.perf_counter()
                active_lemmatizer = ClasslaLemmatizer(
                    pipeline_type=pipeline_type,
                    device=device,
                    resources_dir=resources_dir,
                    pos_batch_size=pos_batch_size,
                    lemma_batch_size=lemma_batch_size,
                    tokenizer_workers=(
                        (1 if pipeline_depth > 1 else 0)
                        if tokenizer_workers is None
                        else tokenizer_workers
                    ),
                )
                active_lemmatizer.start_tokenizer_workers()
                print(
                    f"CLASSLA pipeline loaded in {time.perf_counter() - load_started:.2f}s.",
                    file=sys.stderr,
                )
            if current_batch is not None and process_executor is None:
                assert active_lemmatizer is not None
                local_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="semora-classla",
                )
            if current_batch is not None:
                writer_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="semora-sqlite-writer",
                    initializer=_initialize_lemma_writer,
                    initargs=(str(database_path),),
                )

            current_job = _submit_annotation_job(
                current_batch,
                batch_articles=batch_articles,
                pipeline_depth=pipeline_depth,
                process_executor=process_executor,
                local_executor=local_executor,
                lemmatizer=active_lemmatizer,
            )
            pending_write: _PendingLemmaWrite | None = None
            while current_job is not None:
                next_batch: _LemmaBatch | None = None
                if max_articles is None or scheduled_articles < max_articles:
                    next_limit = _lemma_fetch_limit(
                        batch_articles,
                        concurrent_batches,
                        max_articles,
                        scheduled_articles,
                    )
                    next_batch = _fetch_lemma_batch(
                        database,
                        scheduled_article_id,
                        next_limit,
                        scheduled_articles,
                    )
                    if next_batch is None:
                        exhausted = True
                    else:
                        scheduled_articles = next_batch.processed_articles
                        scheduled_article_id = next_batch.last_article_id

                # Queue the next CLASSLA group before collecting the current
                # one. ProcessPool workers can then move straight into their
                # next CPU-tokenization phase instead of waiting at a group
                # barrier for the slowest worker.
                next_job = _submit_annotation_job(
                    next_batch,
                    batch_articles=batch_articles,
                    pipeline_depth=pipeline_depth,
                    process_executor=process_executor,
                    local_executor=local_executor,
                    lemmatizer=active_lemmatizer,
                )

                if pending_write is not None:
                    write_result = pending_write.future.result()
                    processed_articles = write_result.processed_articles
                    indexed_articles = write_result.indexed_articles
                    progress.update(write_result.added_articles)
                    progress.set_postfix(indexed_articles=f"{indexed_articles:,}")
                    if profile:
                        _print_pipelined_profile(pending_write, write_result)

                annotations, worker_profiles = _finish_annotation_job(current_job)
                assert writer_executor is not None
                pending_write = _PendingLemmaWrite(
                    future=writer_executor.submit(
                        _write_lemma_batch,
                        current_job.batch,
                        annotations,
                        indexed_articles,
                    ),
                    fetch_seconds=current_job.batch.fetch_seconds,
                    worker_profiles=worker_profiles,
                )
                current_job = next_job

            if pending_write is not None:
                write_result = pending_write.future.result()
                processed_articles = write_result.processed_articles
                indexed_articles = write_result.indexed_articles
                last_article_id = scheduled_article_id
                progress.update(write_result.added_articles)
                progress.set_postfix(indexed_articles=f"{indexed_articles:,}")
                if profile:
                    _print_pipelined_profile(pending_write, write_result)

            if exhausted:
                complete = True
                with database.conn:
                    database.conn.execute(
                        """
                        UPDATE article_lemma_index_state
                        SET complete = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE state_id = 1
                        """
                    )
        return LemmaIndexStats(surface_articles, processed_articles, indexed_articles, complete)
    finally:
        if process_executor is not None:
            process_executor.shutdown(wait=True, cancel_futures=True)
        if local_executor is not None:
            local_executor.shutdown(wait=True, cancel_futures=True)
        if writer_executor is not None:
            writer_executor.submit(_close_lemma_writer).result()
            writer_executor.shutdown(wait=True, cancel_futures=True)
        if isinstance(active_lemmatizer, ClasslaLemmatizer):
            active_lemmatizer.close()
        database.close()


def _lemma_fetch_limit(
    batch_articles: int,
    concurrent_batches: int,
    max_articles: int | None,
    scheduled_articles: int,
) -> int:
    limit = batch_articles * concurrent_batches
    if max_articles is not None:
        limit = min(limit, max_articles - scheduled_articles)
    return limit


def _fetch_lemma_batch(
    database: Database,
    last_article_id: str,
    limit: int,
    processed_articles: int,
) -> _LemmaBatch | None:
    if limit <= 0:
        return None
    started = time.perf_counter()
    rows = database.conn.execute(
        """
        SELECT
            articles.article_id,
            article_fts_map.fts_id,
            articles.title,
            articles.content
        FROM article_fts_map
        JOIN articles ON articles.article_id = article_fts_map.article_id
        WHERE article_fts_map.article_id > ?
          AND articles.is_valid = 1
        ORDER BY article_fts_map.article_id
        LIMIT ?
        """,
        (last_article_id, limit),
    ).fetchall()
    if not rows:
        return None
    articles = [
        _LemmaArticle(
            article_id=str(row["article_id"]),
            fts_id=int(row["fts_id"]),
            title=str(row["title"] or ""),
            content=str(row["content"]),
        )
        for row in rows
    ]
    return _LemmaBatch(
        articles=articles,
        fetch_seconds=time.perf_counter() - started,
        processed_articles=processed_articles + len(articles),
    )


def _article_payloads(articles: list[_LemmaArticle]) -> list[str]:
    return [
        f"{article.title}\n{article.content}" if article.title else article.content
        for article in articles
    ]


def _submit_annotation_job(
    batch: _LemmaBatch | None,
    *,
    batch_articles: int,
    pipeline_depth: int,
    process_executor: ProcessPoolExecutor | None,
    local_executor: ThreadPoolExecutor | None,
    lemmatizer: Lemmatizer | None,
) -> _AnnotationJob | None:
    if batch is None:
        return None
    payloads = _article_payloads(batch.articles)
    if process_executor is not None:
        futures = tuple(
            process_executor.submit(_annotate_index_worker, payloads[index : index + batch_articles])
            for index in range(0, len(payloads), batch_articles)
        )
    else:
        assert local_executor is not None and lemmatizer is not None
        futures = (
            local_executor.submit(
                _annotate_local_batch,
                lemmatizer,
                payloads,
                batch_articles,
                pipeline_depth,
            ),
        )
    return _AnnotationJob(batch=batch, futures=futures)


def _finish_annotation_job(
    job: _AnnotationJob,
) -> tuple[list[list[LemmaToken]], list[_WorkerAnnotations]]:
    worker_profiles = [future.result() for future in job.futures]
    annotations = [
        document
        for worker_result in worker_profiles
        for document in worker_result.documents
    ]
    return annotations, worker_profiles


def _annotate_local_batch(
    lemmatizer: Lemmatizer,
    payloads: list[str],
    batch_articles: int,
    pipeline_depth: int,
) -> _WorkerAnnotations:
    started = time.perf_counter()
    if isinstance(lemmatizer, ClasslaLemmatizer) and pipeline_depth > 1:
        payload_batches = [
            payloads[index : index + batch_articles]
            for index in range(0, len(payloads), batch_articles)
        ]
        annotation_batches = lemmatizer.annotate_batches(
            payload_batches,
            pipeline_depth=pipeline_depth,
        )
        annotations = [
            document
            for batch_annotations in annotation_batches
            for document in batch_annotations
        ]
    else:
        annotations = lemmatizer.annotate_many(payloads)
    return _WorkerAnnotations(
        documents=annotations,
        profile=getattr(lemmatizer, "last_profile", None),
        initialization_seconds=0.0,
        processing_seconds=time.perf_counter() - started,
    )


def _initialize_lemma_writer(database_path: str) -> None:
    global _LEMMA_WRITER_DATABASE
    _LEMMA_WRITER_DATABASE = Database(database_path)
    _LEMMA_WRITER_DATABASE.conn.execute("PRAGMA busy_timeout = 60000")
    _LEMMA_WRITER_DATABASE.conn.execute("PRAGMA wal_autocheckpoint = 10000")


def _close_lemma_writer() -> None:
    global _LEMMA_WRITER_DATABASE
    if _LEMMA_WRITER_DATABASE is not None:
        _LEMMA_WRITER_DATABASE.close()
        _LEMMA_WRITER_DATABASE = None


def _write_lemma_batch(
    batch: _LemmaBatch,
    annotations: list[list[LemmaToken]],
    indexed_articles: int,
) -> _LemmaWriteResult:
    if _LEMMA_WRITER_DATABASE is None:
        raise RuntimeError("The lemma-index writer was not initialized.")
    mapping_started = time.perf_counter()
    documents = _lemma_documents_from_annotations(batch, annotations)
    mapping_seconds = time.perf_counter() - mapping_started
    write_started = time.perf_counter()
    new_indexed_articles = indexed_articles + len(documents)
    with _LEMMA_WRITER_DATABASE.conn:
        _LEMMA_WRITER_DATABASE.conn.executemany(
            "INSERT INTO article_lemma_fts (rowid, title, text) VALUES (?, ?, ?)",
            documents,
        )
        _LEMMA_WRITER_DATABASE.conn.execute(
            """
            UPDATE article_lemma_index_state
            SET last_article_id = ?, processed_articles = ?, indexed_articles = ?,
                complete = 0, updated_at = CURRENT_TIMESTAMP
            WHERE state_id = 1
            """,
            (batch.last_article_id, batch.processed_articles, new_indexed_articles),
        )
    return _LemmaWriteResult(
        processed_articles=batch.processed_articles,
        indexed_articles=new_indexed_articles,
        added_articles=len(batch.articles),
        mapping_seconds=mapping_seconds,
        write_seconds=time.perf_counter() - write_started,
    )


def _lemma_documents_from_annotations(
    batch: _LemmaBatch,
    annotations: list[list[LemmaToken]],
) -> list[tuple[int, str, str]]:
    if len(annotations) != len(batch.articles):
        raise ValueError("The lemmatizer returned a different number of documents than it received.")
    documents: list[tuple[int, str, str]] = []
    for article, tokens in zip(batch.articles, annotations, strict=True):
        title_end = len(article.title)
        content_start = title_end + 1 if article.title else 0
        lemma_title = " ".join(
            lemma
            for token in tokens
            if token.end <= title_end
            for lemma in token.lemmas
        ) or article.title
        lemma_text = " ".join(
            lemma
            for token in tokens
            if token.start >= content_start
            for lemma in token.lemmas
        ) or article.content
        documents.append(
            (article.fts_id, lemma_title, lemma_text)
        )
    return documents


def _initialize_index_worker(config: dict[str, Any]) -> None:
    global _INDEX_WORKER_INITIALIZATION_SECONDS, _INDEX_WORKER_LEMMATIZER, _INDEX_WORKER_REPORT_INITIALIZATION
    started = time.perf_counter()
    _INDEX_WORKER_LEMMATIZER = ClasslaLemmatizer(**config)
    _INDEX_WORKER_INITIALIZATION_SECONDS = time.perf_counter() - started
    _INDEX_WORKER_REPORT_INITIALIZATION = True


def _annotate_index_worker(texts: list[str]) -> _WorkerAnnotations:
    global _INDEX_WORKER_REPORT_INITIALIZATION
    if _INDEX_WORKER_LEMMATIZER is None:
        raise RuntimeError("CLASSLA index worker was not initialized.")
    started = time.perf_counter()
    documents = _INDEX_WORKER_LEMMATIZER.annotate_many(texts)
    processing_seconds = time.perf_counter() - started
    initialization_seconds = (
        _INDEX_WORKER_INITIALIZATION_SECONDS if _INDEX_WORKER_REPORT_INITIALIZATION else 0.0
    )
    _INDEX_WORKER_REPORT_INITIALIZATION = False
    return _WorkerAnnotations(
        documents=documents,
        profile=_INDEX_WORKER_LEMMATIZER.last_profile,
        initialization_seconds=initialization_seconds,
        processing_seconds=processing_seconds,
    )


def _print_pipelined_profile(
    pending: _PendingLemmaWrite,
    write_result: _LemmaWriteResult,
) -> None:
    profiles = [result.profile for result in pending.worker_profiles if result.profile is not None]
    classla_seconds = max(
        (result.processing_seconds for result in pending.worker_profiles),
        default=0.0,
    )
    tokens = sum(profile.tokens for profile in profiles)
    details = ""
    if profiles:
        details = (
            f" tokenize={sum(profile.tokenize_seconds for profile in profiles):.3f}s"
            f" pos={sum(profile.pos_seconds for profile in profiles):.3f}s"
            f" lemma={sum(profile.lemma_seconds for profile in profiles):.3f}s"
            f" tokens/s={tokens / classla_seconds if classla_seconds else 0:,.0f}"
            f" peak_cuda_per_worker="
            f"{max(profile.peak_cuda_bytes for profile in profiles) / 1024**2:,.0f}MiB"
        )
    initialization_seconds = max(
        (result.initialization_seconds for result in pending.worker_profiles),
        default=0.0,
    )
    print(
        "Profile: "
        f"workers={len(pending.worker_profiles)} "
        f"fetch={pending.fetch_seconds:.3f}s "
        f"classla_wall={classla_seconds:.3f}s "
        f"map={write_result.mapping_seconds:.3f}s "
        f"write={write_result.write_seconds:.3f}s "
        f"initialization={initialization_seconds:.2f}s"
        f"{details}",
        file=sys.stderr,
    )


def build_semantic_index(
    database_path: str | Path = "indexes/semora.sqlite",
    output_dir: str | Path = "indexes/semantic",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 64,
    device: str | None = None,
) -> int:
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic indexing requires the 'retrieval' extra.") from exc

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_id, device=device)
    database = Database(database_path)
    index = None
    chunk_ids: list[str] = []
    try:
        chunking_rows = database.conn.execute(
            """
            SELECT DISTINCT chunking_runs.config_json
            FROM chunks
            JOIN chunking_runs ON chunking_runs.chunking_run_id = chunks.chunking_run_id
            JOIN articles ON articles.article_id = chunks.article_id
            WHERE articles.is_valid = 1
            """
        ).fetchall()
        if len(chunking_rows) != 1:
            raise ValueError("Semantic indexing requires exactly one chunking configuration.")
        chunking_config = json.loads(chunking_rows[0]["config_json"])
        rows = database.conn.execute(
            """
            SELECT chunks.chunk_id, chunks.text
            FROM chunks
            JOIN articles ON articles.article_id = chunks.article_id
            WHERE articles.is_valid = 1
            ORDER BY chunks.chunk_id
            """
        )
        total_chunks = int(
            database.conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                JOIN articles ON articles.article_id = chunks.article_id
                WHERE articles.is_valid = 1
                """
            ).fetchone()[0]
        )
        with tqdm(
            total=total_chunks,
            desc="Building semantic index",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            while batch := rows.fetchmany(batch_size):
                texts = [str(row["text"]) for row in batch]
                encode = getattr(model, "encode_document", model.encode)
                vectors = encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                matrix = np.asarray(vectors, dtype="float32")
                if index is None:
                    index = faiss.IndexFlatIP(matrix.shape[1])
                index.add(matrix)
                chunk_ids.extend(str(row["chunk_id"]) for row in batch)
                progress.update(len(batch))
        if index is None:
            raise ValueError("The database contains no valid chunks to index.")
        index_temp = target / ".index.faiss.tmp"
        ids_temp = target / ".chunk_ids.json.tmp"
        manifest_temp = target / ".manifest.json.tmp"
        faiss.write_index(index, str(index_temp))
        ids_temp.write_text(
            json.dumps(chunk_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {
            "format": "semora.semantic.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "dimensions": int(index.d),
            "normalized": True,
            "metric": "cosine_via_inner_product",
            "chunking": chunking_config,
            "chunks": len(chunk_ids),
            "database": Path(database_path).name,
        }
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_temp.replace(target / "index.faiss")
        ids_temp.replace(target / "chunk_ids.json")
        manifest_temp.replace(target / "manifest.json")
        return len(chunk_ids)
    finally:
        database.close()
