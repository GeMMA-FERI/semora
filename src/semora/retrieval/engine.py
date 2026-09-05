"""Unified BM25, regular-expression, and semantic newspaper search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from semora.retrieval.models import SearchHit
from semora.storage import Database


class SearchEngine:
    def __init__(
        self,
        database_path: str | Path = "indexes/semora.sqlite",
        semantic_dir: str | Path = "indexes/semantic",
        *,
        load_semantic: bool = False,
    ) -> None:
        self.database = Database(database_path)
        self.database.initialize()
        self.semantic_dir = Path(semantic_dir).resolve()
        self._semantic_index: Any = None
        self._semantic_model: Any = None
        self._semantic_chunk_ids: list[str] = []
        self._semantic_manifest: dict[str, Any] | None = None
        if load_semantic:
            self.load_semantic()

    def close(self) -> None:
        self.database.close()

    @property
    def semantic_loaded(self) -> bool:
        return self._semantic_index is not None

    def load_semantic(self) -> None:
        if self._semantic_index is not None:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic search requires the 'retrieval' extra.") from exc
        manifest_path = self.semantic_dir / "manifest.json"
        index_path = self.semantic_dir / "index.faiss"
        chunk_ids_path = self.semantic_dir / "chunk_ids.json"
        if not manifest_path.is_file() or not index_path.is_file() or not chunk_ids_path.is_file():
            raise FileNotFoundError(f"Semantic index is incomplete: {self.semantic_dir}")
        self._semantic_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._semantic_chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
        self._semantic_index = faiss.read_index(str(index_path))
        if self._semantic_index.ntotal != len(self._semantic_chunk_ids):
            raise ValueError("FAISS index and chunk ID mapping contain different numbers of entries.")
        self._semantic_model = SentenceTransformer(self._semantic_manifest["model_id"])

    def search(
        self,
        mode: str,
        query: str,
        *,
        limit: int = 10,
        before: int = 0,
        after: int = 0,
        context_lines: int = 0,
        ignore_case: bool = False,
        newspaper: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[SearchHit]:
        if limit < 1 or before < 0 or after < 0 or context_lines < 0:
            raise ValueError("limit must be positive and context values must be non-negative.")
        if mode == "bm25":
            matches = self._search_bm25(query, limit, newspaper, date_from, date_to)
        elif mode == "regex":
            matches = self._search_regex(
                query,
                limit,
                ignore_case=ignore_case,
                newspaper=newspaper,
                date_from=date_from,
                date_to=date_to,
            )
        elif mode == "semantic":
            matches = self._search_semantic(query, limit, newspaper, date_from, date_to)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        return [
            self._make_hit(row, score, before=before, after=after, context_lines=context_lines)
            for row, score in matches
        ]

    def _search_bm25(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        rows = self.database.conn.execute(
            """
            SELECT chunks.*, articles.title AS article_title,
                   newspapers.source, newspapers.title AS newspaper_title,
                   newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                   newspapers.relative_path,
                   bm25(chunk_fts, 2.0, 1.0) AS rank
            FROM chunk_fts
            JOIN chunk_fts_map ON chunk_fts_map.fts_id = chunk_fts.rowid
            JOIN chunks ON chunks.chunk_id = chunk_fts_map.chunk_id
            JOIN articles ON articles.article_id = chunks.article_id
            JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
            WHERE chunk_fts MATCH ?
              AND (? IS NULL OR newspapers.source = ? OR newspapers.title = ?)
              AND (? IS NULL OR newspapers.date >= ?)
              AND (? IS NULL OR newspapers.date <= ?)
            ORDER BY rank
            LIMIT ?
            """,
            (
                query,
                newspaper,
                newspaper,
                newspaper,
                date_from,
                date_from,
                date_to,
                date_to,
                limit,
            ),
        ).fetchall()
        return [(row, -float(row["rank"])) for row in rows]

    def _search_regex(
        self,
        pattern: str,
        limit: int,
        *,
        ignore_case: bool,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        results: list[tuple[Any, float]] = []
        articles = self.database.conn.execute(
            """
            SELECT articles.*, newspapers.content AS newspaper_content
            FROM articles
            JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
            WHERE articles.is_valid = 1
              AND (? IS NULL OR newspapers.source = ? OR newspapers.title = ?)
              AND (? IS NULL OR newspapers.date >= ?)
              AND (? IS NULL OR newspapers.date <= ?)
            ORDER BY articles.article_id
            """,
            (newspaper, newspaper, newspaper, date_from, date_from, date_to, date_to),
        )
        for article in articles:
            match = expression.search(str(article["content"]))
            if match is None:
                continue
            content_start = int(article["char_end"]) - len(str(article["content"]))
            match_start = content_start + match.start()
            row = self.database.conn.execute(
                """
                SELECT chunks.*, articles.title AS article_title,
                       newspapers.source, newspapers.title AS newspaper_title,
                       newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                       newspapers.relative_path
                FROM chunks
                JOIN articles ON articles.article_id = chunks.article_id
                JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
                WHERE chunks.article_id = ?
                  AND chunks.char_start <= ?
                  AND chunks.char_end > ?
                ORDER BY chunks.chunk_index
                LIMIT 1
                """,
                (article["article_id"], match_start, match_start),
            ).fetchone()
            if row is not None:
                results.append((row, 1.0))
            if len(results) >= limit:
                break
        return results

    def _search_semantic(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        self.load_semantic()
        import numpy as np

        encode = getattr(self._semantic_model, "encode_query", self._semantic_model.encode)
        vector = encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        candidate_limit = min(int(self._semantic_index.ntotal), max(limit, limit * 20))
        while True:
            scores, indices = self._semantic_index.search(np.asarray(vector, dtype="float32"), candidate_limit)
            results: list[tuple[Any, float]] = []
            for score, index in zip(scores[0], indices[0], strict=True):
                if index < 0:
                    continue
                row = self._chunk_row(self._semantic_chunk_ids[int(index)])
                if row is not None and _matches_filters(row, newspaper, date_from, date_to):
                    results.append((row, float(score)))
                    if len(results) >= limit:
                        return results
            if candidate_limit >= int(self._semantic_index.ntotal):
                return results
            candidate_limit = min(int(self._semantic_index.ntotal), candidate_limit * 2)

    def _chunk_row(self, chunk_id: str) -> Any:
        return self.database.conn.execute(
            """
            SELECT chunks.*, articles.title AS article_title,
                   newspapers.source, newspapers.title AS newspaper_title,
                   newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                   newspapers.relative_path
            FROM chunks
            JOIN articles ON articles.article_id = chunks.article_id
            JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
            WHERE chunks.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()

    def _make_hit(
        self,
        row: Any,
        score: float,
        *,
        before: int,
        after: int,
        context_lines: int,
    ) -> SearchHit:
        span = self.database.conn.execute(
            """
            SELECT MIN(line_start) AS line_start, MAX(line_end) AS line_end
            FROM chunks
            WHERE article_id = ?
              AND chunk_index BETWEEN ? AND ?
            """,
            (
                row["article_id"],
                max(0, int(row["chunk_index"]) - before),
                int(row["chunk_index"]) + after,
            ),
        ).fetchone()
        line_start = max(1, int(span["line_start"] or row["line_start"]) - context_lines)
        line_end = int(span["line_end"] or row["line_end"]) + context_lines
        newspaper_content = self.database.conn.execute(
            """
            SELECT newspapers.content AS content FROM newspapers
            JOIN articles ON articles.newspaper_id = newspapers.newspaper_id
            WHERE articles.article_id = ?
            """,
            (row["article_id"],),
        ).fetchone()["content"]
        lines = str(newspaper_content).splitlines()
        line_end = min(line_end, len(lines))
        snippet = "\n".join(lines[line_start - 1 : line_end])
        return SearchHit(
            newspaper=row["source"] or row["newspaper_title"],
            date=row["date"],
            document_id=row["urn"] or row["document_id"],
            relative_path=row["relative_path"] or "",
            score=score,
            line_start=line_start,
            line_end=line_end,
            snippet=snippet,
            article_title=row["article_title"],
        )


def _matches_filters(
    row: Any,
    newspaper: str | None,
    date_from: str | None,
    date_to: str | None,
) -> bool:
    if newspaper is not None and newspaper not in {row["source"], row["newspaper_title"]}:
        return False
    date = row["date"]
    if date_from is not None and (date is None or date < date_from):
        return False
    return not (date_to is not None and (date is None or date > date_to))
