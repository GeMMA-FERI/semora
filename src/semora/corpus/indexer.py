"""Build Semora's canonical SQLite corpus from Markdown newspaper issues."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from semora.corpus.spans import LineMap, SourceArticle, split_articles
from semora.storage import Article, Chunk, ChunkingRun, Database, Newspaper, Run

DEFAULT_MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_TOKEN_COUNT = 256
DEFAULT_TOKEN_OVERLAP = 64
INGESTION_STAGES = ("newspapers", "articles", "chunks")
WRITE_BATCH_SIZE = 1_000
NEWSPAPER_BATCH_SIZE = 50


@dataclass(frozen=True)
class CorpusStats:
    newspapers: int
    articles: int
    chunks: int
    skipped: int


def ingest_corpus(
    corpus_dir: str | Path = "corpus",
    database_path: str | Path = "indexes/semora.sqlite",
    *,
    model_id: str = DEFAULT_MODEL_ID,
    token_count: int = DEFAULT_TOKEN_COUNT,
    token_overlap: int = DEFAULT_TOKEN_OVERLAP,
    replace: bool = False,
    stages: tuple[str, ...] | None = None,
) -> CorpusStats:
    """Run selected corpus-ingestion stages in dependency order."""
    corpus_path = Path(corpus_dir).resolve()
    target_path = Path(database_path).resolve()
    selected_stages = _validate_stages(stages)
    if "newspapers" in selected_stages and not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_path}")
    if token_count <= 0 or token_overlap < 0 or token_overlap >= token_count:
        raise ValueError("Token count must be positive and overlap must be in [0, token_count).")
    if replace and "newspapers" not in selected_stages:
        raise ValueError("--replace can only be used when the newspapers stage is selected.")
    if "newspapers" in selected_stages:
        if target_path.exists() and not replace:
            raise FileExistsError(f"Index already exists: {target_path}. Pass replace=True to rebuild it.")
        if replace:
            _remove_database_files(target_path)
    elif not target_path.is_file():
        raise FileNotFoundError(
            f"Corpus database does not exist: {target_path}. Run the newspapers stage first."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    database = Database(target_path)
    newspapers = articles = chunks = 0
    try:
        database.initialize()
        if "newspapers" in selected_stages:
            newspapers = _store_newspapers(database, corpus_path)
        if "articles" in selected_stages:
            articles = _store_articles(database)
        if "chunks" in selected_stages:
            chunks = _store_chunks(
                database,
                model_id=model_id,
                token_count=token_count,
                token_overlap=token_overlap,
            )
    finally:
        database.close()
    return CorpusStats(newspapers=newspapers, articles=articles, chunks=chunks, skipped=0)


def _store_newspapers(database: Database, corpus_path: Path) -> int:
    _require_empty(database, "newspapers", "newspapers")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"corpus_{timestamp}"
    database.insert_run(Run(run_id=run_id, run_type="corpus"))
    markdown_paths = sorted(
        path for path in corpus_path.rglob("*.md") if not path.name.endswith("_metadata.md")
    )
    batch: list[Newspaper] = []
    stored = 0
    for markdown_path in tqdm(markdown_paths, desc="Storing newspapers", unit="newspaper"):
        relative_path = markdown_path.relative_to(corpus_path).as_posix()
        metadata_path = markdown_path.with_name(f"{markdown_path.stem}_metadata.json")
        metadata = (
            _read_metadata(metadata_path)
            if metadata_path.is_file()
            else {"Record": {"source": _infer_newspaper_name(markdown_path, corpus_path)}}
        )
        batch.append(
            Newspaper(
                newspaper_id=_stable_id("newspaper", relative_path),
                run_id=run_id,
                content=markdown_path.read_text(encoding="utf-8"),
                metadata=metadata,
                relative_path=relative_path,
            )
        )
        if len(batch) >= NEWSPAPER_BATCH_SIZE:
            database.insert_newspapers(batch)
            stored += len(batch)
            batch.clear()
    database.insert_newspapers(batch)
    stored += len(batch)
    database.log(run_id, "INFO", f"Stored {stored} newspapers.")
    return stored


def _store_articles(database: Database) -> int:
    _require_empty(database, "articles", "articles")
    total = _table_count(database, "newspapers")
    if total == 0:
        raise ValueError("Cannot extract articles because the newspapers table is empty.")
    rows = database.conn.execute(
        "SELECT newspaper_id, run_id, content, relative_path FROM newspapers ORDER BY newspaper_id"
    )
    batch: list[Article] = []
    stored = 0
    run_ids: set[str] = set()
    for row in tqdm(rows, total=total, desc="Extracting articles", unit="newspaper"):
        run_id = str(row["run_id"])
        run_ids.add(run_id)
        relative_path = str(row["relative_path"] or row["newspaper_id"])
        batch.extend(
            _article_record(
                source,
                newspaper_id=str(row["newspaper_id"]),
                run_id=run_id,
                relative_path=relative_path,
            )
            for source in split_articles(str(row["content"]))
        )
        if len(batch) >= WRITE_BATCH_SIZE:
            database.insert_articles(batch)
            stored += len(batch)
            batch.clear()
    database.insert_articles(batch)
    stored += len(batch)
    for run_id in run_ids:
        database.log(run_id, "INFO", f"Stored {stored} articles.")
    return stored


def _store_chunks(
    database: Database,
    *,
    model_id: str,
    token_count: int,
    token_overlap: int,
) -> int:
    _require_empty(database, "chunks", "chunks")
    total_articles = _table_count(database, "articles")
    if total_articles == 0:
        raise ValueError("Cannot create chunks because the articles table is empty.")
    run_rows = database.conn.execute("SELECT DISTINCT run_id FROM articles").fetchall()
    if len(run_rows) != 1:
        raise ValueError("Chunk ingestion requires articles from exactly one corpus run.")
    run_id = str(run_rows[0]["run_id"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    chunking_run_id = f"token_{token_count}_{token_overlap}_{timestamp}"
    database.insert_chunking_run(
        ChunkingRun(
            chunking_run_id=chunking_run_id,
            run_id=run_id,
            method="token",
            config={
                "model_id": model_id,
                "token_count": token_count,
                "token_overlap": token_overlap,
                "source_spans": True,
            },
        )
    )
    tokenizer = _load_tokenizer(model_id)
    newspaper_rows = database.conn.execute(
        "SELECT newspaper_id, content FROM newspapers ORDER BY newspaper_id"
    )
    batch: list[Chunk] = []
    stored = 0
    with tqdm(total=total_articles, desc="Creating chunks", unit="article") as progress:
        for newspaper in newspaper_rows:
            markdown = str(newspaper["content"])
            line_map = LineMap(markdown)
            article_rows = database.conn.execute(
                "SELECT * FROM articles WHERE newspaper_id = ? ORDER BY line_start, article_id",
                (newspaper["newspaper_id"],),
            ).fetchall()
            for article in article_rows:
                content = str(article["content"])
                source = SourceArticle(
                    title=article["title"],
                    content=content,
                    article_index=0,
                    char_start=int(article["char_start"]),
                    char_end=int(article["char_end"]),
                    content_char_start=int(article["char_end"]) - len(content),
                    line_start=int(article["line_start"]),
                    line_end=int(article["line_end"]),
                )
                batch.extend(
                    _token_chunks(
                        source,
                        article_id=str(article["article_id"]),
                        run_id=str(article["run_id"]),
                        chunking_run_id=chunking_run_id,
                        tokenizer=tokenizer,
                        token_count=token_count,
                        token_overlap=token_overlap,
                        line_map=line_map,
                    )
                )
                progress.update()
                if len(batch) >= WRITE_BATCH_SIZE:
                    database.insert_chunks(batch)
                    stored += len(batch)
                    batch.clear()
    database.insert_chunks(batch)
    stored += len(batch)
    database.log(run_id, "INFO", f"Stored {stored} chunks.")
    return stored


def _validate_stages(stages: tuple[str, ...] | None) -> tuple[str, ...]:
    if stages is None:
        return INGESTION_STAGES
    if not stages:
        raise ValueError("At least one ingestion stage must be selected.")
    unknown = set(stages).difference(INGESTION_STAGES)
    if unknown:
        raise ValueError(f"Unknown ingestion stage(s): {', '.join(sorted(unknown))}")
    selected = set(stages)
    return tuple(stage for stage in INGESTION_STAGES if stage in selected)


def _require_empty(database: Database, table: str, stage: str) -> None:
    if _table_count(database, table):
        raise ValueError(f"The {stage} stage has already been populated; rebuild with --newspapers --replace.")


def _table_count(database: Database, table: str) -> int:
    if table not in {"newspapers", "articles", "chunks"}:
        raise ValueError(f"Unsupported table: {table}")
    return int(database.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _remove_database_files(database_path: Path) -> None:
    related_paths = (
        database_path,
        Path(f"{database_path}-journal"),
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    )
    for path in related_paths:
        path.unlink(missing_ok=True)


def _article_record(
    source: SourceArticle,
    *,
    newspaper_id: str,
    run_id: str,
    relative_path: str,
) -> Article:
    article_id = _stable_id("article", relative_path, str(source.char_start), str(source.char_end))
    return Article(
        article_id=article_id,
        run_id=run_id,
        newspaper_id=newspaper_id,
        title=source.title,
        content=source.content,
        metadata={"article_index": source.article_index},
        char_start=source.char_start,
        char_end=source.char_end,
        line_start=source.line_start,
        line_end=source.line_end,
        is_valid=True,
    )


def _token_chunks(
    source: SourceArticle,
    *,
    article_id: str,
    run_id: str,
    chunking_run_id: str,
    tokenizer: Any,
    token_count: int,
    token_overlap: int,
    line_map: LineMap,
) -> list[Chunk]:
    encoded = tokenizer(
        source.content,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    step = token_count - token_overlap
    result: list[Chunk] = []
    for token_start in range(0, len(offsets), step):
        window = [pair for pair in offsets[token_start : token_start + token_count] if pair[1] > pair[0]]
        if not window:
            continue
        local_start, local_end = window[0][0], window[-1][1]
        text = source.content[local_start:local_end]
        left_trim = len(text) - len(text.lstrip())
        right_trim = len(text.rstrip())
        local_start += left_trim
        local_end = local_start + max(0, right_trim - left_trim)
        text = source.content[local_start:local_end]
        if not text:
            continue
        char_start = source.content_char_start + local_start
        char_end = source.content_char_start + local_end
        line_start, line_end = line_map.span(char_start, char_end)
        chunk_index = len(result)
        result.append(
            Chunk(
                chunk_id=_stable_id("chunk", article_id, str(chunk_index), str(char_start), str(char_end)),
                run_id=run_id,
                article_id=article_id,
                chunking_run_id=chunking_run_id,
                chunk_index=chunk_index,
                method="token",
                text=text,
                char_start=char_start,
                char_end=char_end,
                line_start=line_start,
                line_end=line_end,
            )
        )
        if token_start + token_count >= len(offsets):
            break
    return result


def _load_tokenizer(model_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Corpus ingestion requires the 'chunking' extra.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"Tokenizer must support offset mappings: {model_id}")
    return tokenizer


def _read_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return value


def _infer_newspaper_name(markdown_path: Path, corpus_path: Path) -> str:
    relative_path = markdown_path.relative_to(corpus_path)
    folder_name = relative_path.parts[-2] if len(relative_path.parts) > 1 else markdown_path.stem
    return folder_name.replace("_", " ").strip()


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"
