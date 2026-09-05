"""Unified command-line interface for corpus indexing and retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semora.corpus import DEFAULT_MODEL_ID, DEFAULT_TOKEN_COUNT, DEFAULT_TOKEN_OVERLAP, ingest_corpus
from semora.retrieval.engine import SearchEngine
from semora.retrieval.indexing import build_bm25_index, build_lemma_index, build_semantic_index
from semora.retrieval.stdio import run_stdio
from semora.text import download_classla_models


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
        indexed = build_bm25_index(
            database_path,
            max_chunks=args.max_chunks,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
        _print_json({"indexed_chunks": indexed, "index": "bm25"})
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
    if args.command == "index" and args.index_type == "lemma":
        lemma_stats = build_lemma_index(
            database_path,
            max_articles=args.max_articles,
            batch_articles=args.batch_articles,
            rebuild=args.rebuild,
            pipeline_type=args.classla_type,
            device=args.classla_device,
            resources_dir=args.classla_resources_dir,
            pos_batch_size=args.classla_pos_batch_size,
            lemma_batch_size=args.classla_lemma_batch_size,
            profile=args.profile,
        )
        _print_json(
            {
                "index": "lemma",
                "surface_chunks": lemma_stats.surface_chunks,
                "processed_articles": lemma_stats.processed_articles,
                "indexed_chunks": lemma_stats.indexed_chunks,
                "complete": lemma_stats.complete,
            }
        )
        return
    if args.command == "models" and args.model_command == "download-classla":
        download_classla_models(
            pipeline_type=args.classla_type,
            resources_dir=args.classla_resources_dir,
        )
        _print_json({"model": "classla", "language": "sl", "type": args.classla_type})
        return
    if args.command == "search":
        engine = SearchEngine(
            database_path,
            semantic_dir,
            classla_type=args.classla_type,
            classla_device=args.classla_device,
            classla_resources_dir=args.classla_resources_dir,
        )
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
                lemma_weight=args.lemma_weight,
            )
            _print_json({"hits": [hit.as_dict() for hit in hits]})
        finally:
            engine.close()
        return
    if args.command == "stdio":
        engine = SearchEngine(
            database_path,
            semantic_dir,
            load_semantic=not args.no_semantic,
            classla_type=args.classla_type,
            classla_device=args.classla_device,
            classla_resources_dir=args.classla_resources_dir,
        )
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
    bm25 = index_types.add_parser("bm25", help="Build or resume the SQLite FTS5 BM25 index.")
    bm25.add_argument("--max-chunks", type=int, help="Stop when the index reaches this total size.")
    bm25.add_argument("--batch-size", type=int, default=10_000)
    bm25.add_argument("--rebuild", action="store_true", help="Clear the BM25 index before adding chunks.")
    semantic = index_types.add_parser("semantic", help="Build the persistent FAISS semantic index.")
    semantic.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    semantic.add_argument("--batch-size", type=int, default=64)
    semantic.add_argument("--device", default=None, help="Sentence Transformers device, such as cpu or cuda.")
    lemma = index_types.add_parser("lemma", help="Build or resume the CLASSLA lemma BM25 index.")
    lemma.add_argument("--max-articles", type=int, help="Stop when this total number of articles is processed.")
    lemma.add_argument("--batch-articles", type=int, default=50)
    lemma.add_argument("--rebuild", action="store_true", help="Clear the lemma index before processing.")
    lemma.add_argument(
        "--classla-pos-batch-size",
        type=int,
        help="Override CLASSLA's POS inference batch size (model default: 5000).",
    )
    lemma.add_argument(
        "--classla-lemma-batch-size",
        type=int,
        help="Override CLASSLA's lemma inference batch size (model default: 50).",
    )
    lemma.add_argument(
        "--profile",
        action="store_true",
        help="Report database, CLASSLA processor, throughput, and CUDA-memory timings per batch.",
    )
    _add_classla_options(lemma)

    models = commands.add_parser("models", help="Manage optional external language models.")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    classla = model_commands.add_parser("download-classla", help="Download Slovene CLASSLA models.")
    classla.add_argument("--classla-type", default="default", help="CLASSLA pipeline type (default: default).")
    classla.add_argument("--classla-resources-dir", help="Custom CLASSLA resources directory.")

    search = commands.add_parser("search", help="Run one search and write JSON to stdout.")
    search.add_argument("mode", choices=("bm25", "bm25-lemma", "bm25-combined", "regex", "semantic"))
    search.add_argument("query")
    _add_search_options(search)
    _add_classla_options(search)

    stdio = commands.add_parser("stdio", help="Serve newline-delimited JSON requests on stdin/stdout.")
    stdio.add_argument(
        "--no-semantic",
        action="store_true",
        help="Do not preload FAISS and EmbeddingGemma; semantic requests will load them lazily.",
    )
    _add_classla_options(stdio)
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
    parser.add_argument(
        "--lemma-weight",
        type=float,
        default=1.0,
        help="Weight of lemma BM25 scores in bm25-combined mode (default: 1.0).",
    )


def _add_classla_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--classla-type", default="default", help="CLASSLA pipeline type (default: default).")
    parser.add_argument(
        "--classla-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="CLASSLA execution device (default: auto).",
    )
    parser.add_argument("--classla-resources-dir", help="Custom CLASSLA resources directory.")


def _print_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
