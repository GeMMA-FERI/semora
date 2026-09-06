"""Persistent lexical and semantic index builders."""

from __future__ import annotations

import json
import sys
import time
from bisect import bisect_left, bisect_right
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
    surface_chunks: int
    processed_articles: int
    indexed_chunks: int
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
    title: str
    content: str
    char_end: int


@dataclass(frozen=True)
class _LemmaChunk:
    fts_id: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _LemmaBatch:
    articles: list[_LemmaArticle]
    chunks_by_article: dict[str, list[_LemmaChunk]]
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
    indexed_chunks: int
    added_articles: int
    added_chunks: int
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
    max_chunks: int | None = None,
    batch_size: int = 10_000,
    rebuild: bool = False,
) -> int:
    """Build or resume the contentless BM25 index up to a total chunk target."""
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    database = Database(database_path)
    try:
        database.initialize()
        if rebuild:
            with database.conn:
                database.conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('delete-all')")
                database.conn.execute("DELETE FROM chunk_fts_map")
        indexed = int(database.conn.execute("SELECT COUNT(*) FROM chunk_fts_map").fetchone()[0])
        last_row = database.conn.execute(
            "SELECT fts_id, chunk_id FROM chunk_fts_map ORDER BY fts_id DESC LIMIT 1"
        ).fetchone()
        next_fts_id = int(last_row["fts_id"]) + 1 if last_row is not None else 1
        last_chunk_id = str(last_row["chunk_id"]) if last_row is not None else ""
        available = int(
            database.conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                JOIN articles ON articles.article_id = chunks.article_id
                WHERE articles.is_valid = 1
                """
            ).fetchone()[0]
        )
        target = available if max_chunks is None else min(max_chunks, available)
        with tqdm(
            total=max(indexed, target),
            initial=indexed,
            desc="Indexing surface BM25",
            unit="chunk",
            dynamic_ncols=True,
        ) as progress:
            while max_chunks is None or indexed < max_chunks:
                limit = batch_size if max_chunks is None else min(batch_size, max_chunks - indexed)
                rows = database.conn.execute(
                    """
                    SELECT chunks.chunk_id, COALESCE(articles.title, '') AS title, chunks.text
                    FROM chunks
                    JOIN articles ON articles.article_id = chunks.article_id
                    WHERE articles.is_valid = 1
                      AND chunks.chunk_id > ?
                    ORDER BY chunks.chunk_id
                    LIMIT ?
                    """,
                    (last_chunk_id, limit),
                ).fetchall()
                if not rows:
                    break
                mapping = [
                    (next_fts_id + offset, str(row["chunk_id"]))
                    for offset, row in enumerate(rows)
                ]
                documents = [
                    (next_fts_id + offset, str(row["title"]), str(row["text"]))
                    for offset, row in enumerate(rows)
                ]
                with database.conn:
                    database.conn.executemany(
                        "INSERT INTO chunk_fts_map (fts_id, chunk_id) VALUES (?, ?)",
                        mapping,
                    )
                    database.conn.executemany(
                        "INSERT INTO chunk_fts (rowid, title, text) VALUES (?, ?, ?)",
                        documents,
                    )
                added = len(rows)
                indexed += added
                next_fts_id += added
                last_chunk_id = str(rows[-1]["chunk_id"])
                progress.update(added)
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
    """Lemmatize each article once and index chunks present in the surface index."""
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
                database.conn.execute("INSERT INTO chunk_lemma_fts(chunk_lemma_fts) VALUES('delete-all')")
                database.conn.execute("DELETE FROM lemma_index_state")
        surface_chunks = int(database.conn.execute("SELECT COUNT(*) FROM chunk_fts_map").fetchone()[0])
        if surface_chunks == 0:
            raise ValueError("Build the surface BM25 index before building the lemma index.")
        state = database.conn.execute("SELECT * FROM lemma_index_state WHERE state_id = 1").fetchone()
        if state is None:
            with database.conn:
                database.conn.execute(
                    """
                    INSERT INTO lemma_index_state (state_id, surface_chunks, pipeline_type)
                    VALUES (1, ?, ?)
                    """,
                    (surface_chunks, pipeline_type),
                )
            last_article_id = ""
            processed_articles = indexed_chunks = 0
            complete = False
        else:
            if int(state["surface_chunks"]) != surface_chunks:
                raise ValueError("The surface BM25 sample changed; rebuild the lemma index with --rebuild.")
            if str(state["pipeline_type"]) != pipeline_type:
                raise ValueError("The CLASSLA pipeline type changed; rebuild the lemma index with --rebuild.")
            last_article_id = str(state["last_article_id"])
            processed_articles = int(state["processed_articles"])
            indexed_chunks = int(state["indexed_chunks"])
            complete = bool(state["complete"])
        if complete or (max_articles is not None and processed_articles >= max_articles):
            return LemmaIndexStats(surface_chunks, processed_articles, indexed_chunks, complete)

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
            progress.set_postfix(indexed_chunks=f"{indexed_chunks:,}")
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

                if pending_write is not None:
                    write_result = pending_write.future.result()
                    processed_articles = write_result.processed_articles
                    indexed_chunks = write_result.indexed_chunks
                    progress.update(write_result.added_articles)
                    progress.set_postfix(indexed_chunks=f"{indexed_chunks:,}")
                    if profile:
                        _print_pipelined_profile(pending_write, write_result)

                annotations, worker_profiles = _finish_annotation_job(current_job)
                assert writer_executor is not None
                pending_write = _PendingLemmaWrite(
                    future=writer_executor.submit(
                        _write_lemma_batch,
                        current_job.batch,
                        annotations,
                        indexed_chunks,
                    ),
                    fetch_seconds=current_job.batch.fetch_seconds,
                    worker_profiles=worker_profiles,
                )
                current_job = _submit_annotation_job(
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
                indexed_chunks = write_result.indexed_chunks
                last_article_id = scheduled_article_id
                progress.update(write_result.added_articles)
                progress.set_postfix(indexed_chunks=f"{indexed_chunks:,}")
                if profile:
                    _print_pipelined_profile(pending_write, write_result)

            if exhausted:
                complete = True
                with database.conn:
                    database.conn.execute(
                        """
                        UPDATE lemma_index_state
                        SET complete = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE state_id = 1
                        """
                    )
        return LemmaIndexStats(surface_chunks, processed_articles, indexed_chunks, complete)
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
        SELECT articles.article_id, articles.title, articles.content, articles.char_end
        FROM articles
        WHERE articles.is_valid = 1
          AND articles.char_end IS NOT NULL
          AND articles.article_id > ?
          AND EXISTS (
              SELECT 1
              FROM chunks
              JOIN chunk_fts_map ON chunk_fts_map.chunk_id = chunks.chunk_id
              WHERE chunks.article_id = articles.article_id
          )
        ORDER BY articles.article_id
        LIMIT ?
        """,
        (last_article_id, limit),
    ).fetchall()
    if not rows:
        return None
    articles = [
        _LemmaArticle(
            article_id=str(row["article_id"]),
            title=str(row["title"] or ""),
            content=str(row["content"]),
            char_end=int(row["char_end"]),
        )
        for row in rows
    ]
    article_ids = [article.article_id for article in articles]
    placeholders = ",".join("?" for _ in article_ids)
    chunk_rows = database.conn.execute(
        f"""
        SELECT
            chunks.article_id,
            chunk_fts_map.fts_id,
            chunks.text,
            chunks.char_start,
            chunks.char_end
        FROM chunks
        JOIN chunk_fts_map ON chunk_fts_map.chunk_id = chunks.chunk_id
        WHERE chunks.article_id IN ({placeholders})
        ORDER BY chunks.article_id, chunks.chunk_index
        """,
        article_ids,
    ).fetchall()
    chunks_by_article: dict[str, list[_LemmaChunk]] = {
        article_id: [] for article_id in article_ids
    }
    for row in chunk_rows:
        chunks_by_article[str(row["article_id"])].append(
            _LemmaChunk(
                fts_id=int(row["fts_id"]),
                text=str(row["text"]),
                char_start=int(row["char_start"]),
                char_end=int(row["char_end"]),
            )
        )
    return _LemmaBatch(
        articles=articles,
        chunks_by_article=chunks_by_article,
        fetch_seconds=time.perf_counter() - started,
        processed_articles=processed_articles + len(articles),
    )


def _article_payloads(articles: list[_LemmaArticle]) -> list[str]:
    payloads = []
    for article in articles:
        payloads.append(f"{article.title}\n{article.content}" if article.title else article.content)
    return payloads


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
    indexed_chunks: int,
) -> _LemmaWriteResult:
    if _LEMMA_WRITER_DATABASE is None:
        raise RuntimeError("The lemma-index writer was not initialized.")
    mapping_started = time.perf_counter()
    documents = _lemma_documents_from_annotations(batch, annotations)
    mapping_seconds = time.perf_counter() - mapping_started
    write_started = time.perf_counter()
    new_indexed_chunks = indexed_chunks + len(documents)
    with _LEMMA_WRITER_DATABASE.conn:
        _LEMMA_WRITER_DATABASE.conn.executemany(
            "INSERT INTO chunk_lemma_fts (rowid, title, text) VALUES (?, ?, ?)",
            documents,
        )
        _LEMMA_WRITER_DATABASE.conn.execute(
            """
            UPDATE lemma_index_state
            SET last_article_id = ?, processed_articles = ?, indexed_chunks = ?,
                complete = 0, updated_at = CURRENT_TIMESTAMP
            WHERE state_id = 1
            """,
            (batch.last_article_id, batch.processed_articles, new_indexed_chunks),
        )
    return _LemmaWriteResult(
        processed_articles=batch.processed_articles,
        indexed_chunks=new_indexed_chunks,
        added_articles=len(batch.articles),
        added_chunks=len(documents),
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
        documents.extend(
            _lemma_documents(article, tokens, batch.chunks_by_article[article.article_id])
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


def _lemma_documents(
    article: _LemmaArticle,
    tokens: list[LemmaToken],
    chunks: list[_LemmaChunk],
) -> list[tuple[int, str, str]]:
    title = article.title
    content = article.content
    prefix = f"{title}\n" if title else ""
    title_end = len(title)
    token_starts = [token.start for token in tokens]
    token_ends = [token.end for token in tokens]
    lemma_title = " ".join(
        lemma
        for token in tokens[:bisect_left(token_starts, title_end)]
        for lemma in token.lemmas
    ) or title
    content_char_start = article.char_end - len(content)
    documents: list[tuple[int, str, str]] = []
    for chunk in chunks:
        local_start = chunk.char_start - content_char_start + len(prefix)
        local_end = chunk.char_end - content_char_start + len(prefix)
        first_token = bisect_right(token_ends, local_start)
        last_token = bisect_left(token_starts, local_end, lo=first_token)
        lemma_text = " ".join(
            lemma
            for token in tokens[first_token:last_token]
            for lemma in token.lemmas
        ) or chunk.text
        documents.append((chunk.fts_id, lemma_title, lemma_text))
    return documents


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
