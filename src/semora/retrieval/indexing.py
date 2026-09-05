"""Persistent lexical and semantic index builders."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from semora.corpus import DEFAULT_MODEL_ID
from semora.storage import Database
from semora.text.lemmatization import ClasslaLemmatizer, Lemmatizer, LemmaToken


@dataclass(frozen=True)
class LemmaIndexStats:
    surface_chunks: int
    processed_articles: int
    indexed_chunks: int
    complete: bool


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
    lemmatizer: Lemmatizer | None = None,
) -> LemmaIndexStats:
    """Lemmatize each article once and index chunks present in the surface index."""
    if max_articles is not None and max_articles < 0:
        raise ValueError("max_articles must be non-negative.")
    if batch_articles <= 0:
        raise ValueError("batch_articles must be positive.")
    database = Database(database_path)
    try:
        database.initialize()
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

        active_lemmatizer = lemmatizer
        with tqdm(
            total=max(processed_articles, max_articles) if max_articles is not None else None,
            initial=processed_articles,
            desc="Indexing lemma BM25",
            unit="article",
            dynamic_ncols=True,
        ) as progress:
            progress.set_postfix(indexed_chunks=f"{indexed_chunks:,}")
            while max_articles is None or processed_articles < max_articles:
                limit = (
                    batch_articles
                    if max_articles is None
                    else min(batch_articles, max_articles - processed_articles)
                )
                fetch_started = time.perf_counter()
                articles = database.conn.execute(
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
                fetch_seconds = time.perf_counter() - fetch_started
                if not articles:
                    complete = True
                    with database.conn:
                        database.conn.execute(
                            """
                            UPDATE lemma_index_state
                            SET complete = 1, updated_at = CURRENT_TIMESTAMP
                            WHERE state_id = 1
                            """
                        )
                    break
                if active_lemmatizer is None:
                    print("Loading CLASSLA pipeline...", file=sys.stderr)
                    load_started = time.perf_counter()
                    active_lemmatizer = ClasslaLemmatizer(
                        pipeline_type=pipeline_type,
                        device=device,
                        resources_dir=resources_dir,
                        pos_batch_size=pos_batch_size,
                        lemma_batch_size=lemma_batch_size,
                    )
                    print(
                        f"CLASSLA pipeline loaded in {time.perf_counter() - load_started:.2f}s.",
                        file=sys.stderr,
                    )
                processing_started = time.perf_counter()
                documents = _lemma_documents_batch(database, articles, active_lemmatizer)
                processing_seconds = time.perf_counter() - processing_started
                last_article_id = str(articles[-1]["article_id"])
                processed_articles += len(articles)
                indexed_chunks += len(documents)
                write_started = time.perf_counter()
                with database.conn:
                    database.conn.executemany(
                        "INSERT INTO chunk_lemma_fts (rowid, title, text) VALUES (?, ?, ?)",
                        documents,
                    )
                    database.conn.execute(
                        """
                        UPDATE lemma_index_state
                        SET last_article_id = ?, processed_articles = ?, indexed_chunks = ?,
                            complete = 0, updated_at = CURRENT_TIMESTAMP
                        WHERE state_id = 1
                        """,
                        (last_article_id, processed_articles, indexed_chunks),
                    )
                write_seconds = time.perf_counter() - write_started
                progress.update(len(articles))
                progress.set_postfix(indexed_chunks=f"{indexed_chunks:,}")
                if profile:
                    _print_lemma_profile(
                        active_lemmatizer,
                        fetch_seconds=fetch_seconds,
                        processing_seconds=processing_seconds,
                        write_seconds=write_seconds,
                    )
        return LemmaIndexStats(surface_chunks, processed_articles, indexed_chunks, complete)
    finally:
        database.close()


def _lemma_documents_batch(
    database: Database,
    articles: list[sqlite3.Row],
    lemmatizer: Lemmatizer,
) -> list[tuple[int, str, str]]:
    payloads = []
    for article in articles:
        title = str(article["title"] or "")
        content = str(article["content"])
        payloads.append(f"{title}\n{content}" if title else content)
    annotations = lemmatizer.annotate_many(payloads)
    if len(annotations) != len(articles):
        raise ValueError("The lemmatizer returned a different number of documents than it received.")
    documents: list[tuple[int, str, str]] = []
    for article, tokens in zip(articles, annotations, strict=True):
        documents.extend(_lemma_documents(database, article, tokens))
    return documents


def _lemma_documents(
    database: Database,
    article: sqlite3.Row,
    tokens: list[LemmaToken],
) -> list[tuple[int, str, str]]:
    title = str(article["title"] or "")
    content = str(article["content"])
    prefix = f"{title}\n" if title else ""
    title_end = len(title)
    lemma_title = " ".join(
        lemma
        for token in tokens
        if token.start < title_end
        for lemma in token.lemmas
    ) or title
    content_char_start = int(article["char_end"]) - len(content)
    chunks = database.conn.execute(
        """
        SELECT chunk_fts_map.fts_id, chunks.text, chunks.char_start, chunks.char_end
        FROM chunks
        JOIN chunk_fts_map ON chunk_fts_map.chunk_id = chunks.chunk_id
        WHERE chunks.article_id = ?
        ORDER BY chunks.chunk_index
        """,
        (article["article_id"],),
    ).fetchall()
    documents: list[tuple[int, str, str]] = []
    for chunk in chunks:
        local_start = int(chunk["char_start"]) - content_char_start + len(prefix)
        local_end = int(chunk["char_end"]) - content_char_start + len(prefix)
        lemma_text = " ".join(
            lemma
            for token in tokens
            if token.start < local_end and token.end > local_start
            for lemma in token.lemmas
        ) or str(chunk["text"])
        documents.append((int(chunk["fts_id"]), lemma_title, lemma_text))
    return documents


def _print_lemma_profile(
    lemmatizer: Lemmatizer,
    *,
    fetch_seconds: float,
    processing_seconds: float,
    write_seconds: float,
) -> None:
    classla_profile = getattr(lemmatizer, "last_profile", None)
    if classla_profile is None:
        print(
            f"Profile: fetch={fetch_seconds:.3f}s process={processing_seconds:.3f}s "
            f"write={write_seconds:.3f}s",
            file=sys.stderr,
        )
        return
    print(
        "Profile: "
        f"fetch={fetch_seconds:.3f}s "
        f"tokenize={classla_profile.tokenize_seconds:.3f}s "
        f"pos={classla_profile.pos_seconds:.3f}s "
        f"lemma={classla_profile.lemma_seconds:.3f}s "
        f"map={max(0.0, processing_seconds - classla_profile.total_seconds):.3f}s "
        f"write={write_seconds:.3f}s "
        f"tokens/s={classla_profile.tokens_per_second:,.0f} "
        f"peak_cuda={classla_profile.peak_cuda_bytes / 1024**2:,.0f}MiB",
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
