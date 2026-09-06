from __future__ import annotations

import io
import json
import re
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from semora.cli.main import _configure_console_encoding, _parser
from semora.corpus import indexer
from semora.retrieval import SearchEngine
from semora.retrieval.indexing import build_bm25_index, build_lemma_index, build_semantic_index
from semora.retrieval.stdio import run_stdio
from semora.storage import Database


class WordTokenizer:
    is_fast = True

    def __call__(self, text: str, **_kwargs) -> dict:
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class FakeSloveneLemmatizer:
    def __init__(self) -> None:
        self.annotated_articles = 0

    @staticmethod
    def lemmatize(text: str) -> str:
        normalized = {"appears": "appear", "appeared": "appear"}
        return " ".join(
            normalized.get(match.group().casefold(), match.group().casefold())
            for match in re.finditer(r"[^\W\d_]+", text, re.UNICODE)
        )

    def lemmatize_many(self, texts: Sequence[str]) -> list[str]:
        self.annotated_articles += len(texts) // 2
        return [self.lemmatize(text) for text in texts]


class ReconfigurableStream:
    def __init__(self) -> None:
        self.encoding: str | None = None

    def reconfigure(self, *, encoding: str) -> None:
        self.encoding = encoding


def test_cli_configures_all_protocol_streams_as_utf8(monkeypatch) -> None:
    streams = [ReconfigurableStream() for _ in range(3)]
    monkeypatch.setattr(sys, "stdin", streams[0])
    monkeypatch.setattr(sys, "stdout", streams[1])
    monkeypatch.setattr(sys, "stderr", streams[2])

    _configure_console_encoding()

    assert [stream.encoding for stream in streams] == ["utf-8", "utf-8", "utf-8"]


def _index_log_events(database_path: Path, run_type: str) -> list[tuple[str, str, dict]]:
    database = Database(database_path)
    try:
        rows = database.conn.execute(
            """
            SELECT logs.run_id, logs.level, logs.message
            FROM logs
            JOIN runs ON runs.run_id = logs.run_id
            WHERE runs.run_type = ?
            ORDER BY logs.log_id
            """,
            (run_type,),
        ).fetchall()
        return [
            (str(row["run_id"]), str(row["level"]), json.loads(row["message"]))
            for row in rows
        ]
    finally:
        database.close()


def _build_corpus(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    root = tmp_path / "workspace"
    issue_dir = root / "corpus" / "Jutro_Ljubljana"
    issue_dir.mkdir(parents=True)
    markdown = (
        "# First article\n"
        "Alpha beta gamma delta.\n"
        "Second line here.\n"
        "\n"
        "# Other article\n"
        "Needle appears here.\n"
    )
    stem = "URN_NBN_SI_doc-0L8XYEOC"
    (issue_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    metadata = {
        "Record": {
            "date": "1934 10 10",
            "source": "Jutro",
            "identifier": {"@identifier_type": "URN", "#text": "URN:NBN:SI:doc-0L8XYEOC"},
        }
    }
    (issue_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda _model_id: WordTokenizer())
    stats = indexer.ingest_corpus(
        root / "corpus",
        root / "indexes" / "semora.sqlite",
        token_count=3,
        token_overlap=1,
    )
    assert (stats.newspapers, stats.articles, stats.chunks, stats.skipped) == (1, 2, 4, 0)
    return root, markdown


def test_ingestion_retains_original_source_spans(tmp_path: Path, monkeypatch) -> None:
    root, markdown = _build_corpus(tmp_path, monkeypatch)
    database = Database(root / "indexes" / "semora.sqlite")
    try:
        newspaper = database.conn.execute("SELECT * FROM newspapers").fetchone()
        assert newspaper["relative_path"] == "Jutro_Ljubljana/URN_NBN_SI_doc-0L8XYEOC.md"
        articles = database.conn.execute("SELECT * FROM articles ORDER BY line_start").fetchall()
        assert [(row["line_start"], row["line_end"]) for row in articles] == [(1, 3), (5, 6)]
        for row in articles:
            assert markdown[row["char_start"] : row["char_end"]].startswith("#")
        chunks = database.conn.execute("SELECT * FROM chunks ORDER BY line_start, chunk_index").fetchall()
        for row in chunks:
            assert markdown[row["char_start"] : row["char_end"]] == row["text"]
    finally:
        database.close()


def test_ingestion_infers_newspaper_name_when_metadata_is_missing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspace"
    issue_dir = root / "corpus" / "Slovenski_Narod"
    issue_dir.mkdir(parents=True)
    (issue_dir / "issue.md").write_text("# News\nArticle text.\n", encoding="utf-8")
    monkeypatch.setattr(indexer, "_load_tokenizer", lambda _model_id: WordTokenizer())

    stats = indexer.ingest_corpus(root / "corpus", root / "indexes" / "semora.sqlite")

    assert (stats.newspapers, stats.articles, stats.chunks, stats.skipped) == (1, 1, 1, 0)
    database = Database(root / "indexes" / "semora.sqlite")
    try:
        newspaper = database.conn.execute("SELECT * FROM newspapers").fetchone()
        assert newspaper["source"] == "Slovenski Narod"
        assert newspaper["date"] is None
        assert newspaper["urn"] is None
        assert json.loads(newspaper["metadata_json"]) == {"Record": {"source": "Slovenski Narod"}}
    finally:
        database.close()


def test_ingestion_stages_can_run_separately(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspace"
    issue_dir = root / "corpus" / "Daily_Paper"
    issue_dir.mkdir(parents=True)
    (issue_dir / "issue.md").write_text("# One\nAlpha beta gamma.\n\n# Two\nDelta epsilon.\n", encoding="utf-8")
    database_path = root / "indexes" / "semora.sqlite"
    monkeypatch.setattr(
        indexer,
        "_load_tokenizer",
        lambda _model_id: (_ for _ in ()).throw(AssertionError("tokenizer loaded before chunk stage")),
    )

    newspaper_stats = indexer.ingest_corpus(
        root / "corpus",
        database_path,
        stages=("newspapers",),
    )
    assert (newspaper_stats.newspapers, newspaper_stats.articles, newspaper_stats.chunks) == (1, 0, 0)
    article_stats = indexer.ingest_corpus(
        root / "does-not-need-to-exist",
        database_path,
        stages=("articles",),
    )
    assert (article_stats.newspapers, article_stats.articles, article_stats.chunks) == (0, 2, 0)

    monkeypatch.setattr(indexer, "_load_tokenizer", lambda _model_id: WordTokenizer())
    chunk_stats = indexer.ingest_corpus(
        root / "does-not-need-to-exist",
        database_path,
        stages=("chunks",),
        token_count=3,
        token_overlap=1,
    )
    assert (chunk_stats.newspapers, chunk_stats.articles, chunk_stats.chunks) == (0, 0, 2)

    database = Database(database_path)
    try:
        assert database.conn.execute("SELECT COUNT(*) FROM newspapers").fetchone()[0] == 1
        assert database.conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 2
        assert database.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    finally:
        database.close()


def test_ingest_cli_accepts_combinable_stage_flags() -> None:
    args = _parser().parse_args(["ingest", "--newspapers", "--articles"])
    assert args.newspapers is True
    assert args.articles is True
    assert args.chunks is False

    index_args = _parser().parse_args(["index", "bm25", "--max-articles", "100000"])
    assert index_args.max_articles == 100_000

    lemma_args = _parser().parse_args(
        [
            "index",
            "lemma",
            "--max-articles",
            "1000",
            "--classla-device",
            "cpu",
            "--classla-pos-batch-size",
            "10000",
            "--classla-lemma-batch-size",
            "200",
            "--workers",
            "4",
            "--profile",
        ]
    )
    assert lemma_args.max_articles == 1_000
    assert lemma_args.classla_device == "cpu"
    assert lemma_args.classla_pos_batch_size == 10_000
    assert lemma_args.classla_lemma_batch_size == 200
    assert lemma_args.workers == 4
    assert lemma_args.pipeline_depth == 3
    assert lemma_args.tokenizer_workers is None
    assert lemma_args.profile is True

    search_args = _parser().parse_args(
        ["search", "bm25-combined", "gledališča", "--lemma-weight", "0.5", "--profile"]
    )
    assert search_args.lemma_weight == 0.5
    assert search_args.max_snippet_chars == 600
    assert search_args.profile is True

    model_args = _parser().parse_args(["models", "download-classla"])
    assert model_args.classla_type == "default"

    benchmark_args = _parser().parse_args(
        ["benchmark", "classla", "--articles", "200", "--workers", "1", "2"]
    )
    assert benchmark_args.articles == 200
    assert benchmark_args.workers == [1, 2]
    assert benchmark_args.pipeline_depth == 1
    assert benchmark_args.tokenizer_workers == 0

    profile_args = _parser().parse_args(
        ["profile", "classla", "--articles", "2", "--output", "trace.json"]
    )
    assert profile_args.articles == 2
    assert profile_args.output == "trace.json"


def test_bm25_regex_and_stdio_share_json_contract(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"
    database = Database(database_path)
    try:
        with database.conn:
            database.conn.execute("DELETE FROM chunks")
    finally:
        database.close()
    assert build_bm25_index(database_path) == 2
    engine = SearchEngine(database_path, root / "indexes" / "semantic")
    try:
        bm25 = engine.search("bm25", "Needle", limit=1, profile=True)
        assert bm25[0].newspaper == "Jutro"
        assert bm25[0].date == "1934-10-10"
        assert bm25[0].document_id == "0L8XYEOC"
        assert bm25[0].urn == "URN:NBN:SI:doc-0L8XYEOC"
        assert bm25[0].line_start == 5
        assert "Needle appears here." in bm25[0].snippet
        assert engine.last_profile is not None
        assert engine.last_profile["returned_hits"] == 1
        assert engine.last_profile["timings_seconds"]["article_fts_sql"] >= 0
        assert engine.last_profile["timings_seconds"]["hit_building"] >= 0
        assert engine.last_profile["query_plans"]["article_fts"]
        assert engine.last_profile["sqlite"]["database_size_bytes"] > 0

        excerpt = engine.read_source("0L8XYEOC", 2, 4)
        assert excerpt.document_id == "0L8XYEOC"
        assert excerpt.urn == "URN:NBN:SI:doc-0L8XYEOC"
        assert excerpt.line_start == 2
        assert excerpt.line_end == 4
        assert excerpt.text == "Alpha beta gamma delta.\nSecond line here.\n\n"
        assert excerpt.truncated is False

        truncated = engine.read_source("URN:NBN:SI:doc-0L8XYEOC", 2, 4, max_bytes=5)
        assert truncated.text == "Alpha"
        assert truncated.line_end == 2
        assert truncated.truncated is True
        with pytest.raises(KeyError, match="Unknown document_id"):
            engine.read_source("unknown", 1, 1)
        with pytest.raises(ValueError, match="line_start"):
            engine.read_source("0L8XYEOC", 0, 1)

        short_bm25 = engine.search("bm25", "Needle", limit=1, max_snippet_chars=20)
        assert len(short_bm25[0].snippet) == 20
        assert "Needle" in short_bm25[0].snippet

        regex = engine.search("regex", r"beta\s+gamma", limit=1, context_lines=1)
        assert regex[0].line_start == 1
        assert regex[0].line_end == 4
        assert regex[0].article_title == "First article"
        assert "Second line here." in regex[0].snippet

        requests = io.StringIO(
            json.dumps({"id": "one", "op": "search", "mode": "bm25", "query": "Needle", "limit": 1})
            + "\n"
            + json.dumps(
                {
                    "id": "read",
                    "op": "read",
                    "document_id": "0L8XYEOC",
                    "line_start": 5,
                    "line_end": 6,
                }
            )
            + "\n"
            + json.dumps({"id": "health", "op": "health"})
            + "\n"
            + json.dumps({"id": "stop", "op": "shutdown"})
            + "\n"
        )
        responses = io.StringIO()
        run_stdio(engine, requests, responses)
        values = [json.loads(line) for line in responses.getvalue().splitlines()]
        assert values[0]["id"] == "one"
        assert values[0]["hits"][0]["relative_path"].startswith("Jutro_Ljubljana/")
        assert values[1]["id"] == "read"
        assert values[1]["source"]["document_id"] == "0L8XYEOC"
        assert "Needle appears here." in values[1]["source"]["text"]
        assert values[2]["id"] == "health"
        assert values[2]["indexes"]["bm25"]["complete"] is True
        assert values[2]["indexes"]["semantic"]["available"] is False
        assert values[3] == {"id": "stop", "ok": True}
    finally:
        engine.close()


def test_contentless_bm25_index_resumes_to_total_target(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"

    assert build_bm25_index(database_path, max_articles=1, batch_size=1) == 1
    events = _index_log_events(database_path, "index_bm25")
    assert [event[2]["event"] for event in events] == ["index_started", "index_completed"]
    assert events[0][0] == events[1][0]
    assert events[1][2]["indexed_articles"] == 1
    assert events[1][2]["added_articles"] == 1
    database = Database(database_path)
    try:
        first_mapping = database.conn.execute(
            "SELECT fts_id, article_id FROM article_fts_map ORDER BY fts_id"
        ).fetchall()
        schema = database.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'article_fts'"
        ).fetchone()["sql"]
        stored_columns = database.conn.execute("SELECT title, text FROM article_fts LIMIT 1").fetchone()
        assert "content = ''" in schema
        assert tuple(stored_columns) == (None, None)
    finally:
        database.close()

    assert build_bm25_index(database_path, max_articles=2, batch_size=1) == 2
    assert build_bm25_index(database_path, max_articles=2, batch_size=1) == 2
    database = Database(database_path)
    try:
        resumed_mapping = database.conn.execute(
            "SELECT fts_id, article_id FROM article_fts_map ORDER BY fts_id"
        ).fetchall()
        assert tuple(resumed_mapping[0]) == tuple(first_mapping[0])
    finally:
        database.close()

    assert build_bm25_index(database_path, batch_size=1) == 2
    assert build_bm25_index(database_path, max_articles=1, rebuild=True) == 1


def test_contentless_lemma_index_resumes_and_supports_combined_search(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"
    assert build_bm25_index(database_path) == 2
    lemmatizer = FakeSloveneLemmatizer()

    partial = build_lemma_index(
        database_path,
        max_articles=1,
        batch_articles=1,
        lemmatizer=lemmatizer,
    )
    assert partial.processed_articles == 1
    assert partial.complete is False
    finished = build_lemma_index(database_path, batch_articles=1, lemmatizer=lemmatizer)
    assert finished.processed_articles == 2
    assert finished.indexed_articles == 2
    assert finished.complete is True
    assert lemmatizer.annotated_articles == 2

    database = Database(database_path)
    try:
        schema = database.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'article_lemma_fts'"
        ).fetchone()["sql"]
        stored_columns = database.conn.execute(
            "SELECT title, text FROM article_lemma_fts LIMIT 1"
        ).fetchone()
        materialized = database.conn.execute(
            "SELECT title, content, pipeline_type FROM article_lemmas ORDER BY article_id"
        ).fetchall()
        assert "content = ''" in schema
        assert tuple(stored_columns) == (None, None)
        assert len(materialized) == 2
        assert materialized[0]["pipeline_type"] == "default"
        assert any("needle appear here" in row["content"] for row in materialized)
    finally:
        database.close()

    engine = SearchEngine(
        database_path,
        root / "indexes" / "semantic",
        lemmatizer=lemmatizer,
    )
    try:
        assert engine.search("bm25", "appeared") == []
        lemma_hits = engine.search("bm25-lemma", "appeared", limit=1)
        assert "Needle appears here." in lemma_hits[0].snippet
        combined_hits = engine.search("bm25-combined", "Needle appeared", limit=1)
        assert "Needle appears here." in combined_hits[0].snippet
    finally:
        engine.close()


def test_pipelined_lemma_index_checkpoints_completed_writes(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"
    assert build_bm25_index(database_path) == 2

    class FailingLemmatizer(FakeSloveneLemmatizer):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def lemmatize_many(self, texts: Sequence[str]) -> list[str]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated CLASSLA failure")
            return super().lemmatize_many(texts)

    with pytest.raises(RuntimeError, match="simulated CLASSLA failure"):
        build_lemma_index(
            database_path,
            batch_articles=1,
            lemmatizer=FailingLemmatizer(),
        )

    failed_events = _index_log_events(database_path, "index_lemma")
    assert failed_events[-1][1] == "ERROR"
    assert failed_events[-1][2]["event"] == "index_failed"
    assert failed_events[-1][2]["error_type"] == "RuntimeError"

    database = Database(database_path)
    try:
        state = database.conn.execute("SELECT * FROM article_lemma_index_state").fetchone()
        assert state["processed_articles"] == 1
        indexed_rows = database.conn.execute("SELECT COUNT(*) FROM article_lemma_fts").fetchone()[0]
        materialized_rows = database.conn.execute("SELECT COUNT(*) FROM article_lemmas").fetchone()[0]
        assert indexed_rows == state["indexed_articles"]
        assert materialized_rows == indexed_rows
        assert 0 < indexed_rows < 2
    finally:
        database.close()

    finished = build_lemma_index(
        database_path,
        batch_articles=1,
        lemmatizer=FakeSloveneLemmatizer(),
    )
    assert finished.processed_articles == 2
    assert finished.indexed_articles == 2
    assert finished.complete is True


def test_semantic_index_is_persistent_and_uses_manifest_model(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    stored_indexes = {}

    class FakeIndex:
        def __init__(self, dimensions: int) -> None:
            self.d = dimensions
            self.vectors = np.empty((0, dimensions), dtype="float32")

        @property
        def ntotal(self) -> int:
            return len(self.vectors)

        def add(self, vectors) -> None:
            self.vectors = np.vstack((self.vectors, vectors))

        def search(self, queries, limit: int):
            similarities = queries @ self.vectors.T
            indices = np.argsort(-similarities, axis=1)[:, :limit]
            return np.take_along_axis(similarities, indices, axis=1), indices

    def write_index(index, path: str) -> None:
        Path(path).write_bytes(b"fake-faiss")
        stored_indexes[str(Path(path).with_name("index.faiss"))] = index

    def read_index(path: str):
        return stored_indexes[path]

    class FakeModel:
        def __init__(self, model_id: str, device=None) -> None:
            assert model_id == "google/embeddinggemma-300m"

        def encode(self, texts, **_kwargs):
            return self._vectors(texts)

        def encode_document(self, texts, **_kwargs):
            return self._vectors(texts)

        def encode_query(self, texts, **_kwargs):
            return self._vectors(texts)

        @staticmethod
        def _vectors(texts):
            vectors = np.asarray(
                [[1.0, 0.0] if "needle" in text.casefold() else [0.0, 1.0] for text in texts],
                dtype="float32",
            )
            return vectors

    monkeypatch.setitem(
        sys.modules,
        "faiss",
        types.SimpleNamespace(IndexFlatIP=FakeIndex, write_index=write_index, read_index=read_index),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeModel),
    )
    semantic_dir = root / "indexes" / "semantic"
    count = build_semantic_index(root / "indexes" / "semora.sqlite", semantic_dir, batch_size=2)
    assert count == 4
    semantic_events = _index_log_events(root / "indexes" / "semora.sqlite", "index_semantic")
    assert [event[2]["event"] for event in semantic_events] == [
        "index_started",
        "index_completed",
    ]
    assert semantic_events[-1][2]["indexed_chunks"] == 4
    assert semantic_events[-1][2]["dimensions"] == 2
    manifest = json.loads((semantic_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "google/embeddinggemma-300m"
    assert manifest["chunking"]["token_count"] == 3
    assert manifest["chunking"]["token_overlap"] == 1

    engine = SearchEngine(root / "indexes" / "semora.sqlite", semantic_dir, load_semantic=True)
    try:
        hits = engine.search("semantic", "needle", limit=1)
        assert "Needle appears here." in hits[0].snippet
        article_hits = engine.search("semantic", "needle", limit=2)
        assert len({hit.article_title for hit in article_hits}) == 2
        assert engine.search("semantic", "needle", limit=1, newspaper="Other") == []
    finally:
        engine.close()
