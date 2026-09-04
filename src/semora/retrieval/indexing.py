"""Persistent lexical and semantic index builders."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from semora.corpus import DEFAULT_MODEL_ID
from semora.storage import Database


def build_bm25_index(database_path: str | Path = "indexes/semora.sqlite") -> int:
    database = Database(database_path)
    try:
        database.initialize()
        with database.conn:
            database.conn.execute("DELETE FROM chunk_fts")
            database.conn.execute(
                """
                INSERT INTO chunk_fts (chunk_id, title, text)
                SELECT chunks.chunk_id, COALESCE(articles.title, ''), chunks.text
                FROM chunks
                JOIN articles ON articles.article_id = chunks.article_id
                WHERE articles.is_valid = 1
                """
            )
        return int(database.conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
    finally:
        database.close()


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
            if len(chunk_ids) % (batch_size * 100) == 0:
                print(f"Embedded {len(chunk_ids):,} chunks", file=sys.stderr)
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
