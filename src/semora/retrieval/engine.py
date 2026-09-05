"""Unified BM25, regular-expression, and semantic newspaper search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from semora.retrieval.models import SearchHit
from semora.storage import Database
from semora.text.lemmatization import ClasslaLemmatizer, Lemmatizer


class SearchEngine:
    def __init__(
        self,
        database_path: str | Path = "indexes/semora.sqlite",
        semantic_dir: str | Path = "indexes/semantic",
        *,
        load_semantic: bool = False,
        lemmatizer: Lemmatizer | None = None,
        classla_type: str = "default",
        classla_device: str = "auto",
        classla_resources_dir: str | Path | None = None,
    ) -> None:
        self.database = Database(database_path)
        self.database.initialize()
        self.semantic_dir = Path(semantic_dir).resolve()
        self._semantic_index: Any = None
        self._semantic_model: Any = None
        self._semantic_chunk_ids: list[str] = []
        self._semantic_manifest: dict[str, Any] | None = None
        self._lemmatizer = lemmatizer
        self._classla_type = classla_type
        self._classla_device = classla_device
        self._classla_resources_dir = classla_resources_dir
        if load_semantic:
            self.load_semantic()

    def close(self) -> None:
        self.database.close()

    @property
    def semantic_loaded(self) -> bool:
        return self._semantic_index is not None

    @property
    def lemma_loaded(self) -> bool:
        return self._lemmatizer is not None

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
        lemma_weight: float = 1.0,
    ) -> list[SearchHit]:
        if limit < 1 or before < 0 or after < 0 or context_lines < 0:
            raise ValueError("limit must be positive and context values must be non-negative.")
        if lemma_weight < 0:
            raise ValueError("lemma_weight must be non-negative.")
        if mode == "bm25":
            matches = self._search_bm25(query, limit, newspaper, date_from, date_to)
        elif mode == "bm25-lemma":
            matches = self._search_lemma_bm25(query, limit, newspaper, date_from, date_to)
        elif mode == "bm25-combined":
            matches = self._search_combined_bm25(
                query,
                limit,
                newspaper,
                date_from,
                date_to,
                lemma_weight,
            )
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
        return self._search_fts("chunk_fts", query, limit, newspaper, date_from, date_to)

    def _search_lemma_bm25(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        lemma_query = self._lemmatize_query(query)
        if not lemma_query:
            return []
        return self._search_fts("chunk_lemma_fts", lemma_query, limit, newspaper, date_from, date_to)

    def _search_combined_bm25(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
        lemma_weight: float,
    ) -> list[tuple[Any, float]]:
        candidate_limit = max(50, limit * 5)
        surface = self._search_bm25(query, candidate_limit, newspaper, date_from, date_to)
        lemma = self._search_lemma_bm25(query, candidate_limit, newspaper, date_from, date_to)
        combined: dict[str, tuple[Any, float]] = {}
        for row, score in surface:
            combined[str(row["chunk_id"])] = (row, score)
        for row, score in lemma:
            chunk_id = str(row["chunk_id"])
            previous = combined.get(chunk_id)
            combined[chunk_id] = (row, lemma_weight * score + (previous[1] if previous else 0.0))
        return sorted(combined.values(), key=lambda item: item[1], reverse=True)[:limit]

    def _search_fts(
        self,
        table: str,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        if table not in {"chunk_fts", "chunk_lemma_fts"}:
            raise ValueError(f"Unsupported FTS table: {table}")
        rows = self.database.conn.execute(
            f"""
            SELECT chunks.*, articles.title AS article_title,
                   newspapers.source, newspapers.title AS newspaper_title,
                   newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                   newspapers.relative_path,
                   bm25({table}, 2.0, 1.0) AS rank
            FROM {table}
            JOIN chunk_fts_map ON chunk_fts_map.fts_id = {table}.rowid
            JOIN chunks ON chunks.chunk_id = chunk_fts_map.chunk_id
            JOIN articles ON articles.article_id = chunks.article_id
            JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
            WHERE {table} MATCH ?
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

    def _lemmatize_query(self, query: str) -> str:
        state = self.database.conn.execute(
            "SELECT indexed_chunks FROM lemma_index_state WHERE state_id = 1"
        ).fetchone()
        if state is None or int(state["indexed_chunks"]) == 0:
            raise ValueError("Build the lemma index with 'semora index lemma' before lemma search.")
        if self._lemmatizer is None:
            self._lemmatizer = ClasslaLemmatizer(
                pipeline_type=self._classla_type,
                device=self._classla_device,
                resources_dir=self._classla_resources_dir,
            )
        lemmas = (
            lemma
            for token in self._lemmatizer.annotate(query)
            for lemma in token.lemmas
            if any(character.isalnum() for character in lemma)
        )
        return " ".join(f'"{lemma.replace(chr(34), chr(34) * 2)}"' for lemma in lemmas)

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
