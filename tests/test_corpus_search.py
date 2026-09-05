from __future__ import annotations

import io
import json
import re
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from semora.cli.main import _parser
from semora.corpus import indexer
from semora.retrieval import SearchEngine
from semora.retrieval.indexing import build_bm25_index, build_lemma_index, build_semantic_index
from semora.retrieval.stdio import run_stdio
from semora.storage import Database
from semora.text import LemmaToken


class WordTokenizer:
    is_fast = True

    def __call__(self, text: str, **_kwargs) -> dict:
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class FakeSloveneLemmatizer:
    def __init__(self) -> None:
        self.annotated_articles = 0

    def annotate(self, text: str) -> list[LemmaToken]:
        if "\n" in text:
            self.annotated_articles += 1
        normalized = {"appears": "appear", "appeared": "appear"}
        return [
            LemmaToken(
                match.start(),
                match.end(),
                (normalized.get(match.group().casefold(), match.group().casefold()),),
            )
            for match in re.finditer(r"[^\W\d_]+", text, re.UNICODE)
        ]

    def annotate_many(self, texts: Sequence[str]) -> list[list[LemmaToken]]:
        return [self.annotate(text) for text in texts]

    def lemmatize(self, text: str) -> str:
        return " ".join(lemma for token in self.annotate(text) for lemma in token.lemmas)


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

    index_args = _parser().parse_args(["index", "bm25", "--max-chunks", "100000"])
    assert index_args.max_chunks == 100_000

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
            "--profile",
        ]
    )
    assert lemma_args.max_articles == 1_000
    assert lemma_args.classla_device == "cpu"
    assert lemma_args.classla_pos_batch_size == 10_000
    assert lemma_args.classla_lemma_batch_size == 200
    assert lemma_args.profile is True

    search_args = _parser().parse_args(["search", "bm25-combined", "gledališča", "--lemma-weight", "0.5"])
    assert search_args.lemma_weight == 0.5

    model_args = _parser().parse_args(["models", "download-classla"])
    assert model_args.classla_type == "default"


def test_bm25_regex_and_stdio_share_json_contract(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"
    assert build_bm25_index(database_path) == 4
    engine = SearchEngine(database_path, root / "indexes" / "semantic")
    try:
        bm25 = engine.search("bm25", "Needle", limit=1)
        assert bm25[0].newspaper == "Jutro"
        assert bm25[0].date == "1934-10-10"
        assert bm25[0].document_id == "URN:NBN:SI:doc-0L8XYEOC"
        assert bm25[0].line_start == 6
        assert "Needle appears here." in bm25[0].snippet

        regex = engine.search("regex", r"beta\s+gamma", limit=1, context_lines=1)
        assert regex[0].line_start == 1
        assert regex[0].article_title == "First article"

        requests = io.StringIO(
            json.dumps({"id": "one", "op": "search", "mode": "bm25", "query": "Needle", "limit": 1})
            + "\n"
            + json.dumps({"id": "stop", "op": "shutdown"})
            + "\n"
        )
        responses = io.StringIO()
        run_stdio(engine, requests, responses)
        values = [json.loads(line) for line in responses.getvalue().splitlines()]
        assert values[0]["id"] == "one"
        assert values[0]["hits"][0]["relative_path"].startswith("Jutro_Ljubljana/")
        assert values[1] == {"id": "stop", "ok": True}
    finally:
        engine.close()


def test_contentless_bm25_index_resumes_to_total_target(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"

    assert build_bm25_index(database_path, max_chunks=2, batch_size=1) == 2
    database = Database(database_path)
    try:
        first_mapping = database.conn.execute(
            "SELECT fts_id, chunk_id FROM chunk_fts_map ORDER BY fts_id"
        ).fetchall()
        schema = database.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunk_fts'"
        ).fetchone()["sql"]
        stored_columns = database.conn.execute("SELECT title, text FROM chunk_fts LIMIT 1").fetchone()
        assert "content = ''" in schema
        assert tuple(stored_columns) == (None, None)
    finally:
        database.close()

    assert build_bm25_index(database_path, max_chunks=3, batch_size=1) == 3
    assert build_bm25_index(database_path, max_chunks=3, batch_size=1) == 3
    database = Database(database_path)
    try:
        resumed_mapping = database.conn.execute(
            "SELECT fts_id, chunk_id FROM chunk_fts_map ORDER BY fts_id"
        ).fetchall()
        assert [tuple(row) for row in resumed_mapping[:2]] == [tuple(row) for row in first_mapping]
    finally:
        database.close()

    assert build_bm25_index(database_path, batch_size=1) == 4
    assert build_bm25_index(database_path, max_chunks=1, rebuild=True) == 1


def test_contentless_lemma_index_resumes_and_supports_combined_search(tmp_path: Path, monkeypatch) -> None:
    root, _ = _build_corpus(tmp_path, monkeypatch)
    database_path = root / "indexes" / "semora.sqlite"
    assert build_bm25_index(database_path) == 4
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
    assert finished.indexed_chunks == 4
    assert finished.complete is True
    assert lemmatizer.annotated_articles == 2

    database = Database(database_path)
    try:
        schema = database.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunk_lemma_fts'"
        ).fetchone()["sql"]
        stored_columns = database.conn.execute(
            "SELECT title, text FROM chunk_lemma_fts LIMIT 1"
        ).fetchone()
        assert "content = ''" in schema
        assert tuple(stored_columns) == (None, None)
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
    manifest = json.loads((semantic_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "google/embeddinggemma-300m"
    assert manifest["chunking"]["token_count"] == 3
    assert manifest["chunking"]["token_overlap"] == 1

    engine = SearchEngine(root / "indexes" / "semora.sqlite", semantic_dir, load_semantic=True)
    try:
        hits = engine.search("semantic", "needle", limit=1)
        assert "Needle appears here." in hits[0].snippet
        assert engine.search("semantic", "needle", limit=1, newspaper="Other") == []
    finally:
        engine.close()
