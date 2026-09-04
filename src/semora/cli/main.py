"""Unified command-line interface for corpus indexing and retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semora.corpus import DEFAULT_MODEL_ID, DEFAULT_TOKEN_COUNT, DEFAULT_TOKEN_OVERLAP, ingest_corpus
from semora.retrieval.engine import SearchEngine
from semora.retrieval.indexing import build_bm25_index, build_semantic_index
from semora.retrieval.stdio import run_stdio


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.root).resolve()
    corpus_dir = root / "corpus"
    indexes_dir = root / "indexes"
    database_path = indexes_dir / "semora.sqlite"
    semantic_dir = indexes_dir / "semantic"

    if args.command == "ingest":
        requested_stages = tuple(
            stage
            for stage in ("newspapers", "articles", "chunks")
            if getattr(args, stage)
        )
        stats = ingest_corpus(
            corpus_dir,
            database_path,
            model_id=args.model_id,
            token_count=args.token_count,
            token_overlap=args.token_overlap,
            replace=args.replace,
            stages=requested_stages or None,
        )
        _print_json(
            {
                "newspapers": stats.newspapers,
                "articles": stats.articles,
                "chunks": stats.chunks,
                "skipped": stats.skipped,
            }
        )
        return
    if args.command == "index" and args.index_type == "bm25":
        _print_json({"indexed_chunks": build_bm25_index(database_path), "index": "bm25"})
        return
    if args.command == "index" and args.index_type == "semantic":
        count = build_semantic_index(
            database_path,
            semantic_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            device=args.device,
        )
        _print_json({"indexed_chunks": count, "index": "semantic", "model_id": args.model_id})
        return
    if args.command == "search":
        engine = SearchEngine(database_path, semantic_dir)
        try:
            hits = engine.search(
                args.mode,
                args.query,
                limit=args.limit,
                before=args.before,
                after=args.after,
                context_lines=args.context_lines,
                ignore_case=args.ignore_case,
                newspaper=args.newspaper,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            _print_json({"hits": [hit.as_dict() for hit in hits]})
        finally:
            engine.close()
        return
    if args.command == "stdio":
        engine = SearchEngine(database_path, semantic_dir, load_semantic=not args.no_semantic)
        try:
            run_stdio(engine)
        finally:
            engine.close()
        return
    raise AssertionError("Unhandled command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semora", description="Index and search a newspaper corpus.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing ./corpus and ./indexes (default: current directory).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Build indexes/semora.sqlite from corpus Markdown and metadata.")
    ingest.add_argument(
        "--replace",
        action="store_true",
        help="Remove the existing database before storing newspapers.",
    )
    ingest.add_argument("--newspapers", action="store_true", help="Store source issues and metadata.")
    ingest.add_argument("--articles", action="store_true", help="Extract articles from stored newspapers.")
    ingest.add_argument("--chunks", action="store_true", help="Create token chunks from stored articles.")
    ingest.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Tokenizer model used for token chunks.")
    ingest.add_argument("--token-count", type=int, default=DEFAULT_TOKEN_COUNT)
    ingest.add_argument("--token-overlap", type=int, default=DEFAULT_TOKEN_OVERLAP)

    index = commands.add_parser("index", help="Build a retrieval index from indexes/semora.sqlite.")
    index_types = index.add_subparsers(dest="index_type", required=True)
    index_types.add_parser("bm25", help="Build the SQLite FTS5 BM25 index.")
    semantic = index_types.add_parser("semantic", help="Build the persistent FAISS semantic index.")
    semantic.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    semantic.add_argument("--batch-size", type=int, default=64)
    semantic.add_argument("--device", default=None, help="Sentence Transformers device, such as cpu or cuda.")

    search = commands.add_parser("search", help="Run one search and write JSON to stdout.")
    search.add_argument("mode", choices=("bm25", "regex", "semantic"))
    search.add_argument("query")
    _add_search_options(search)

    stdio = commands.add_parser("stdio", help="Serve newline-delimited JSON requests on stdin/stdout.")
    stdio.add_argument(
        "--no-semantic",
        action="store_true",
        help="Do not preload FAISS and EmbeddingGemma; semantic requests will load them lazily.",
    )
    return parser


def _add_search_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--before", type=int, default=0, help="Include this many preceding chunks.")
    parser.add_argument("--after", type=int, default=0, help="Include this many following chunks.")
    parser.add_argument("--context-lines", type=int, default=0, help="Add source lines around the chunk span.")
    parser.add_argument("--ignore-case", action="store_true", help="Use case-insensitive regex matching.")
    parser.add_argument("--newspaper", help="Restrict matches to a normalized source or newspaper title.")
    parser.add_argument("--date-from", help="Restrict matches to this ISO date or later.")
    parser.add_argument("--date-to", help="Restrict matches to this ISO date or earlier.")


def _print_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
