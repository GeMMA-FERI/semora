"""Unified BM25, regular-expression, and semantic newspaper search."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from semora.retrieval.models import SearchHit, SourceExcerpt
from semora.storage import Database
from semora.text.lemmatization import ClasslaLemmatizer, Lemmatizer

DEFAULT_MAX_SNIPPET_CHARS = 600
DEFAULT_MAX_READ_BYTES = 65_536
URN_PREFIX = "URN:NBN:SI:doc-"


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
        self.database = Database(database_path, read_only=True)
        self.semantic_dir = Path(semantic_dir).resolve()
        self._semantic_index: Any = None
        self._semantic_model: Any = None
        self._semantic_chunk_ids: list[str] = []
        self._semantic_manifest: dict[str, Any] | None = None
        self._lemmatizer = lemmatizer
        self._classla_type = classla_type
        self._classla_device = classla_device
        self._classla_resources_dir = classla_resources_dir
        self.last_profile: dict[str, Any] | None = None
        self._active_profile: dict[str, Any] | None = None
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

    def index_status(self) -> dict[str, dict[str, Any]]:
        """Return inexpensive readiness information from index state records."""
        bm25 = self.database.conn.execute(
            "SELECT indexed_articles, complete, updated_at FROM article_fts_state WHERE state_id = 1"
        ).fetchone()
        lemma = self.database.conn.execute(
            """
            SELECT surface_articles, processed_articles, indexed_articles,
                   pipeline_type, complete, updated_at
            FROM article_lemma_index_state
            WHERE state_id = 1
            """
        ).fetchone()
        semantic_files = {
            "manifest": self.semantic_dir / "manifest.json",
            "index": self.semantic_dir / "index.faiss",
            "chunk_ids": self.semantic_dir / "chunk_ids.json",
        }
        semantic_available = all(path.is_file() for path in semantic_files.values())
        semantic: dict[str, Any] = {
            "available": semantic_available,
            "complete": semantic_available,
            "loaded": self.semantic_loaded,
        }
        if semantic_available:
            manifest = json.loads(semantic_files["manifest"].read_text(encoding="utf-8"))
            semantic.update(
                model_id=manifest.get("model_id"),
                chunks=manifest.get("chunks"),
                dimensions=manifest.get("dimensions"),
                created_at=manifest.get("created_at"),
            )
        return {
            "bm25": {
                "available": bm25 is not None and int(bm25["indexed_articles"]) > 0,
                "complete": bool(bm25["complete"]) if bm25 else False,
                "indexed_articles": int(bm25["indexed_articles"]) if bm25 else 0,
                "updated_at": bm25["updated_at"] if bm25 else None,
            },
            "lemma": {
                "available": lemma is not None and int(lemma["indexed_articles"]) > 0,
                "complete": bool(lemma["complete"]) if lemma else False,
                "surface_articles": int(lemma["surface_articles"]) if lemma else 0,
                "processed_articles": int(lemma["processed_articles"]) if lemma else 0,
                "indexed_articles": int(lemma["indexed_articles"]) if lemma else 0,
                "pipeline_type": lemma["pipeline_type"] if lemma else None,
                "updated_at": lemma["updated_at"] if lemma else None,
            },
            "semantic": semantic,
        }

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

    def read_source(
        self,
        document_id: str,
        line_start: int,
        line_end: int,
        *,
        max_bytes: int = DEFAULT_MAX_READ_BYTES,
    ) -> SourceExcerpt:
        """Read a bounded line range from one newspaper stored in SQLite."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if line_start < 1:
            raise ValueError("line_start must be at least 1.")
        if line_end < line_start:
            raise ValueError("line_end must be greater than or equal to line_start.")
        if not 1 <= max_bytes <= DEFAULT_MAX_READ_BYTES:
            raise ValueError(f"max_bytes must be between 1 and {DEFAULT_MAX_READ_BYTES}.")

        full_urn = document_id if document_id.startswith(URN_PREFIX) else URN_PREFIX + document_id
        rows = self.database.conn.execute(
            """
            SELECT newspaper_id, content, source, title, date, urn, relative_path
            FROM newspapers
            WHERE urn = ?
            LIMIT 2
            """,
            (full_urn,),
        ).fetchall()
        if not rows:
            rows = self.database.conn.execute(
                """
                SELECT newspaper_id, content, source, title, date, urn, relative_path
                FROM newspapers
                WHERE newspaper_id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"Unknown document_id: {document_id}")
        if len(rows) > 1:
            raise ValueError(f"document_id is not unique: {document_id}")

        row = rows[0]
        lines = str(row["content"]).splitlines(keepends=True)
        if line_start > len(lines):
            raise ValueError(f"line_start exceeds the document's {len(lines)} lines.")
        bounded_end = min(line_end, len(lines))
        text, returned_lines, truncated = _bounded_source_lines(
            lines[line_start - 1 : bounded_end],
            max_bytes,
        )
        urn = str(row["urn"]) if row["urn"] else None
        return SourceExcerpt(
            newspaper=row["source"] or row["title"],
            date=row["date"],
            document_id=_document_id(urn, str(row["newspaper_id"])),
            urn=urn,
            relative_path=str(row["relative_path"] or ""),
            line_start=line_start,
            line_end=line_start + returned_lines - 1,
            text=text,
            truncated=truncated or bounded_end < line_end,
        )

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
        max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
        profile: bool = False,
    ) -> list[SearchHit]:
        if limit < 1 or before < 0 or after < 0 or context_lines < 0:
            raise ValueError("limit must be positive and context values must be non-negative.")
        if lemma_weight < 0:
            raise ValueError("lemma_weight must be non-negative.")
        if max_snippet_chars < 1:
            raise ValueError("max_snippet_chars must be positive.")
        search_started = time.perf_counter()
        self.last_profile = None
        self._active_profile = (
            {
                "mode": mode,
                "query": query,
                "limit": limit,
                "filters": {
                    "newspaper": newspaper,
                    "date_from": date_from,
                    "date_to": date_to,
                },
                "timings_seconds": {},
                "query_plans": {},
            }
            if profile
            else None
        )
        retrieval_started = time.perf_counter()
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
        self._record_profile_time("retrieval", retrieval_started)
        hit_build_started = time.perf_counter()
        hits = [
            self._make_hit(
                row,
                score,
                before=before,
                after=after,
                context_lines=context_lines,
                max_snippet_chars=max_snippet_chars,
            )
            for row, score in matches
        ]
        self._record_profile_time("hit_building", hit_build_started)
        if self._active_profile is not None:
            self._active_profile["returned_hits"] = len(hits)
            self._active_profile["sqlite"] = self._sqlite_profile()
            self._active_profile["timings_seconds"]["total"] = (
                time.perf_counter() - search_started
            )
            self.last_profile = self._active_profile
            self._active_profile = None
        return hits

    def _search_bm25(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        return self._search_fts(
            "article_fts", query, limit, newspaper, date_from, date_to, focus_query=query
        )

    def _search_lemma_bm25(
        self,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[tuple[Any, float]]:
        started = time.perf_counter()
        lemma_query = self._lemmatize_query(query)
        self._record_profile_time("query_lemmatization", started)
        if not lemma_query:
            return []
        return self._search_fts(
            "article_lemma_fts",
            lemma_query,
            limit,
            newspaper,
            date_from,
            date_to,
            focus_query=query,
        )

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
            combined[str(row["article_id"])] = (row, score)
        for row, score in lemma:
            article_id = str(row["article_id"])
            previous = combined.get(article_id)
            combined[article_id] = (row, lemma_weight * score + (previous[1] if previous else 0.0))
        return sorted(combined.values(), key=lambda item: item[1], reverse=True)[:limit]

    def _search_fts(
        self,
        table: str,
        query: str,
        limit: int,
        newspaper: str | None,
        date_from: str | None,
        date_to: str | None,
        *,
        focus_query: str,
    ) -> list[tuple[Any, float]]:
        if table not in {"article_fts", "article_lemma_fts"}:
            raise ValueError(f"Unsupported FTS table: {table}")
        sql = f"""
            SELECT articles.*, articles.title AS article_title,
                   newspapers.source, newspapers.title AS newspaper_title,
                   newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                   newspapers.relative_path, newspapers.content AS newspaper_content,
                   bm25({table}, 2.0, 1.0) AS rank
            FROM {table}
            JOIN article_fts_map ON article_fts_map.fts_id = {table}.rowid
            JOIN articles ON articles.article_id = article_fts_map.article_id
            JOIN newspapers ON newspapers.newspaper_id = articles.newspaper_id
            WHERE {table} MATCH ?
              AND articles.is_valid = 1
              AND (? IS NULL OR newspapers.source = ? OR newspapers.title = ?)
              AND (? IS NULL OR newspapers.date >= ?)
              AND (? IS NULL OR newspapers.date <= ?)
            ORDER BY rank
            LIMIT ?
            """
        parameters = (
            query,
            newspaper,
            newspaper,
            newspaper,
            date_from,
            date_from,
            date_to,
            date_to,
            limit,
        )
        if self._active_profile is not None:
            plan_started = time.perf_counter()
            plan = self.database.conn.execute(
                f"EXPLAIN QUERY PLAN {sql}",
                parameters,
            ).fetchall()
            self._active_profile["query_plans"][table] = [str(row["detail"]) for row in plan]
            self._record_profile_time("query_planning", plan_started)
        sql_started = time.perf_counter()
        rows = self.database.conn.execute(sql, parameters).fetchall()
        self._record_profile_time(f"{table}_sql", sql_started)
        result_started = time.perf_counter()
        results: list[tuple[Any, float]] = []
        for row in rows:
            match = dict(row)
            match["_focus_char"] = _query_focus_char(match, focus_query)
            results.append((match, -float(row["rank"])))
        self._record_profile_time("fts_result_processing", result_started)
        return results

    def _record_profile_time(self, name: str, started: float) -> None:
        if self._active_profile is None:
            return
        timings = self._active_profile["timings_seconds"]
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    def _sqlite_profile(self) -> dict[str, Any]:
        profile = self._active_profile
        assert profile is not None
        page_size = int(self.database.conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(self.database.conn.execute("PRAGMA page_count").fetchone()[0])
        cache_setting = int(self.database.conn.execute("PRAGMA cache_size").fetchone()[0])
        mmap_size = int(self.database.conn.execute("PRAGMA mmap_size").fetchone()[0])
        cache_bytes = (
            abs(cache_setting) * 1024
            if cache_setting < 0
            else cache_setting * page_size
        )
        plans = [
            step
            for plan in profile["query_plans"].values()
            for step in plan
        ]
        warnings = []
        if cache_bytes < 64 * 1024**2:
            warnings.append("SQLite page cache is below 64 MiB; random joins may repeatedly read from disk.")
        if mmap_size == 0:
            warnings.append("SQLite memory-mapped I/O is disabled.")
        if any("USE TEMP B-TREE FOR ORDER BY" in step for step in plans):
            warnings.append(
                "SQLite sorts all FTS matches by BM25 rank before LIMIT; common terms can be expensive."
            )
        return {
            "database_size_bytes": page_size * page_count,
            "page_size_bytes": page_size,
            "page_count": page_count,
            "cache_size_bytes": cache_bytes,
            "mmap_size_bytes": mmap_size,
            "warnings": warnings,
        }

    def _lemmatize_query(self, query: str) -> str:
        state = self.database.conn.execute(
            "SELECT indexed_articles FROM article_lemma_index_state WHERE state_id = 1"
        ).fetchone()
        if state is None or int(state["indexed_articles"]) == 0:
            raise ValueError("Build the lemma index with 'semora index lemma' before lemma search.")
        if self._lemmatizer is None:
            self._lemmatizer = ClasslaLemmatizer(
                pipeline_type=self._classla_type,
                device=self._classla_device,
                resources_dir=self._classla_resources_dir,
            )
        lemmas = (
            lemma
            for lemma in self._lemmatizer.lemmatize(query).split()
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
            SELECT articles.*, articles.title AS article_title,
                   newspapers.source, newspapers.title AS newspaper_title,
                   newspapers.date, newspapers.urn, newspapers.newspaper_id AS document_id,
                   newspapers.relative_path, newspapers.content AS newspaper_content
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
            row = dict(article)
            content_start = int(row["char_end"]) - len(str(row["content"]))
            row["_focus_char"] = content_start + match.start()
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
            seen_articles: set[str] = set()
            for score, index in zip(scores[0], indices[0], strict=True):
                if index < 0:
                    continue
                row = self._chunk_row(self._semantic_chunk_ids[int(index)])
                if (
                    row is not None
                    and str(row["article_id"]) not in seen_articles
                    and _matches_filters(row, newspaper, date_from, date_to)
                ):
                    seen_articles.add(str(row["article_id"]))
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
              AND articles.is_valid = 1
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
        max_snippet_chars: int,
    ) -> SearchHit:
        if "chunk_index" in row.keys():
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
            base_start = int(span["line_start"] or row["line_start"])
            base_end = int(span["line_end"] or row["line_end"])
        else:
            base_start = int(row["line_start"])
            base_end = int(row["line_end"])
        focus_char = int(
            row.get("_focus_char", (int(row["char_start"]) + int(row["char_end"])) // 2)
            if isinstance(row, dict)
            else (int(row["char_start"]) + int(row["char_end"])) // 2
        )
        line_start = max(1, base_start - context_lines)
        line_end = base_end + context_lines
        if "newspaper_content" in row.keys():
            newspaper_content = row["newspaper_content"]
        else:
            newspaper_content = self.database.conn.execute(
                """
                SELECT newspapers.content AS content FROM newspapers
                JOIN articles ON articles.newspaper_id = newspapers.newspaper_id
                WHERE articles.article_id = ?
                """,
                (row["article_id"],),
            ).fetchone()["content"]
        snippet, line_start, line_end = _trim_source_span(
            str(newspaper_content),
            line_start,
            line_end,
            focus_char,
            max_snippet_chars,
        )
        return SearchHit(
            newspaper=row["source"] or row["newspaper_title"],
            date=row["date"],
            document_id=_document_id(row["urn"], str(row["document_id"])),
            relative_path=row["relative_path"] or "",
            score=score,
            line_start=line_start,
            line_end=line_end,
            snippet=snippet,
            article_title=row["article_title"],
            urn=row["urn"],
        )


def _document_id(urn: str | None, fallback: str) -> str:
    if urn and urn.startswith(URN_PREFIX):
        return urn[len(URN_PREFIX) :]
    return urn or fallback


def _bounded_source_lines(lines: list[str], max_bytes: int) -> tuple[str, int, bool]:
    parts: list[str] = []
    used_bytes = 0
    for line in lines:
        encoded = line.encode("utf-8")
        remaining = max_bytes - used_bytes
        if remaining == 0:
            return "".join(parts), len(parts), True
        if len(encoded) <= remaining:
            parts.append(line)
            used_bytes += len(encoded)
            continue
        parts.append(encoded[:remaining].decode("utf-8", errors="ignore"))
        return "".join(parts), len(parts), True
    return "".join(parts), len(parts), False


def _query_focus_char(row: dict[str, Any], query: str) -> int:
    title = str(row.get("article_title") or "")
    content = str(row.get("content") or "")
    terms = [
        term
        for term in re.findall(r"[^\W_]+", query, re.UNICODE)
        if term.casefold() not in {"and", "or", "not", "near"}
    ]
    for text, start in (
        (title, int(row["char_start"])),
        (content, int(row["char_end"]) - len(content)),
    ):
        folded = text.casefold()
        positions = [folded.find(term.casefold()) for term in terms]
        positions = [position for position in positions if position >= 0]
        if positions:
            return start + min(positions)
    return int(row["char_start"])


def _trim_source_span(
    source: str,
    line_start: int,
    line_end: int,
    focus_char: int,
    max_chars: int,
) -> tuple[str, int, int]:
    line_starts = [0, *(match.end() for match in re.finditer("\n", source))]
    line_count = len(line_starts)
    line_start = min(max(1, line_start), line_count)
    line_end = min(max(line_start, line_end), line_count)
    span_start = line_starts[line_start - 1]
    span_end = line_starts[line_end] if line_end < line_count else len(source)
    if span_end - span_start <= max_chars:
        return source[span_start:span_end].rstrip("\r\n"), line_start, line_end

    focus_char = min(max(focus_char, span_start), max(span_start, span_end - 1))
    excerpt_start = min(
        max(span_start, focus_char - max_chars // 3),
        span_end - max_chars,
    )
    excerpt_end = excerpt_start + max_chars
    excerpt = source[excerpt_start:excerpt_end]
    if excerpt_start > span_start:
        excerpt = "…" + excerpt[1:]
    if excerpt_end < span_end:
        excerpt = excerpt[:-1] + "…"
    actual_line_start = source.count("\n", 0, excerpt_start) + 1
    actual_line_end = source.count("\n", 0, max(excerpt_start, excerpt_end - 1)) + 1
    return excerpt, actual_line_start, actual_line_end


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
