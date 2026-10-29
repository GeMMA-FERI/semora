"""Unified command-line interface for corpus indexing and retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from semora.corpus import DEFAULT_MODEL_ID, DEFAULT_TOKEN_COUNT, DEFAULT_TOKEN_OVERLAP, ingest_corpus
from semora.diagnostics.classla import (
    benchmark_classla,
    load_classla_workload,
    profile_classla,
    save_diagnostics,
)
from semora.retrieval.engine import (
    DEFAULT_MAX_READ_BYTES,
    DEFAULT_MAX_SNIPPET_CHARS,
    DEFAULT_SOURCE_WRAP_CHARS,
    SearchEngine,
)
from semora.retrieval.indexing import build_bm25_index, build_lemma_index, build_semantic_index
from semora.retrieval.semantic_faiss import FaissBuildConfig
from semora.retrieval.semantic_vectors import DEFAULT_DIMENSIONS, DEFAULT_SHARD_SIZE
from semora.retrieval.stdio import run_stdio
from semora.text import download_classla_models


def main() -> None:
    _configure_console_encoding()
    args = _parser().parse_args()
    root = Path(args.root).resolve()
    corpus_dir = root / "corpus"
    indexes_dir = root / "indexes"
    database_path = indexes_dir / "semora.sqlite"
    semantic_dir = indexes_dir / "semantic"

    if args.command == "pipeline":
        from semora.pipeline.batches import batch_status, plan_batch, run_batch
        from semora.pipeline.manifest import load_manifest

        target = Path(args.database).resolve() if args.database else database_path
        if args.pipeline_command == "status":
            _print_json(batch_status(target, args.batch))
            return
        manifest = load_manifest(args.manifest)
        input_root = Path(args.input_root).resolve()
        def progress(event: dict) -> None:
            print(json.dumps(event), file=sys.stderr, flush=True)

        if args.pipeline_command == "plan":
            _print_json(plan_batch(manifest, input_root, target, progress=progress))
            return
        if not args.maintenance:
            raise ValueError(
                "Stop retrieval readers and index writers, then pass --maintenance. "
                "This milestone ingests corpus data only; retrieval indexes are not additive yet."
            )
        stages = ("newspapers", "articles", "chunks")
        _print_json(
            run_batch(
                manifest,
                input_root,
                target,
                stages=stages[: stages.index(args.through) + 1],
                model_id=args.model_id,
                token_count=args.token_count,
                token_overlap=args.token_overlap,
                tokenizer_revision=args.tokenizer_revision,
                progress=progress,
            )
        )
        return

    if args.command == "extensions":
        from semora.ner.spellfix import install_spellfix

        _print_json(install_spellfix(root, args.source, args.sha256))
        return
    if args.command == "index" and args.index_type == "entities":
        from semora.ner.names import build_entity_names

        _print_json(build_entity_names(root, args.run_id, args.batch_size))
        return

    if args.command == "index" and args.index_type == "ner":
        from semora.ner.indexing import build_ner_index

        _print_json(build_ner_index(
            database_path, run_id=args.run_id, device=args.device,
            batch_articles=args.batch_articles, batch_size=args.batch_size,
            pipeline_depth=args.pipeline_depth, max_articles=args.max_articles,
            tokenizer_workers=args.tokenizer_workers, ner_pipeline_depth=args.ner_pipeline_depth,
            ner_decoder=args.ner_decoder, resources=args.classla_resources_dir,
        ))
        return

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
            max_articles=args.max_articles,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
        _print_json({"indexed_articles": indexed, "index": "bm25"})
        return
    if args.command == "index" and args.index_type == "semantic":
        count = build_semantic_index(
            database_path,
            semantic_dir,
            model_id=args.model_id,
            model_revision=args.model_revision,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            max_chunks=args.max_chunks,
            device=args.device,
            stage=args.stage,
            index_config=FaissBuildConfig(
                index_type=args.faiss_type,
                nlist=args.nlist,
                pq_m=args.pq_m,
                pq_bits=args.pq_bits,
                train_samples=args.train_samples,
                nprobe=args.nprobe,
            ),
            allow_partial_index=args.allow_partial_index,
            checkpoint_shards=args.checkpoint_shards,
            rebuild_faiss=args.rebuild_faiss,
        )
        _print_json(
            {
                "indexed_chunks": count,
                "index": "semantic",
                "stage": args.stage,
                "model_id": args.model_id,
                "dimensions": args.dimensions,
            }
        )
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
            workers=args.workers,
            pipeline_depth=args.pipeline_depth,
            tokenizer_workers=args.tokenizer_workers,
        )
        _print_json(
            {
                "index": "lemma",
                "surface_articles": lemma_stats.surface_articles,
                "processed_articles": lemma_stats.processed_articles,
                "indexed_articles": lemma_stats.indexed_articles,
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
    if args.command == "benchmark" and args.benchmark_type == "classla":
        texts = load_classla_workload(database_path, args.articles)
        results = benchmark_classla(
            texts,
            worker_counts=args.workers,
            batch_articles=args.batch_articles,
            pipeline_type=args.classla_type,
            device=args.classla_device,
            resources_dir=args.classla_resources_dir,
            pos_batch_size=args.classla_pos_batch_size,
            lemma_batch_size=args.classla_lemma_batch_size,
            pipeline_depth=args.pipeline_depth,
            tokenizer_workers=args.tokenizer_workers,
        )
        output_path = _resolve_output(root, args.output, "indexes/classla_benchmark.json")
        results["output"] = str(save_diagnostics(results, output_path))
        _print_json(results)
        return
    if args.command == "profile" and args.profile_type == "classla":
        texts = load_classla_workload(database_path, args.articles)
        output_path = _resolve_output(root, args.output, "indexes/profiles/classla_trace.json")
        results = profile_classla(
            texts,
            output_path,
            pipeline_type=args.classla_type,
            device=args.classla_device,
            resources_dir=args.classla_resources_dir,
            pos_batch_size=args.classla_pos_batch_size,
            lemma_batch_size=args.classla_lemma_batch_size,
        )
        _print_json(results)
        return
    if args.command == "search":
        command_started = time.perf_counter()
        engine_started = time.perf_counter()
        engine = SearchEngine(
            database_path,
            semantic_dir,
            semantic_device=args.semantic_device,
        )
        engine_initialization_seconds = time.perf_counter() - engine_started
        try:
            hits = engine.search(
                args.mode,
                args.query,
                limit=args.limit,
                offset=args.offset,
                before=args.before,
                after=args.after,
                context_lines=args.context_lines,
                ignore_case=args.ignore_case,
                newspaper=args.newspaper,
                date_from=args.date_from,
                date_to=args.date_to,
                lemma_weight=args.lemma_weight,
                max_snippet_chars=args.max_snippet_chars,
                profile=args.profile,
            )
            if args.profile and engine.last_profile is not None:
                engine.last_profile["timings_seconds"]["engine_initialization"] = (
                    engine_initialization_seconds
                )
                engine.last_profile["timings_seconds"]["command_total"] = (
                    time.perf_counter() - command_started
                )
                print(
                    json.dumps({"profile": engine.last_profile}, ensure_ascii=False, indent=2),
                    file=sys.stderr,
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
            semantic_device=args.semantic_device,
        )
        try:
            run_stdio(engine)
        finally:
            engine.close()
        return
    if args.command in {"read", "status"}:
        engine = SearchEngine(database_path, semantic_dir)
        try:
            if args.command == "read":
                source = engine.read_source(
                    args.document_id,
                    args.line_start,
                    args.line_end,
                    position=args.position,
                    before=args.before,
                    after=args.after,
                    wrap_chars=args.wrap_chars,
                    max_bytes=args.max_bytes,
                )
                _print_json({"source": source.as_dict()})
            else:
                _print_json({"indexes": engine.index_status()})
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

    pipeline = commands.add_parser("pipeline", help="Plan and ingest additive batches (not retrieval-ready yet).")
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    for name in ("plan", "run", "status"):
        command = pipeline_commands.add_parser(name)
        command.add_argument("--database", help="Explicit SQLite path, independent of --root.")
        if name == "status":
            command.add_argument("--batch", required=True)
            continue
        command.add_argument("--manifest", required=True, help="Portable semora.batch.v1 JSON manifest.")
        command.add_argument("--input-root", required=True, help="Root resolving manifest input file paths.")
        if name == "run":
            command.add_argument(
                "--maintenance", action="store_true", help="Confirm readers and old writers are stopped."
            )
            command.add_argument("--through", choices=("newspapers", "articles", "chunks"), default="chunks")
            command.add_argument("--model-id", default=DEFAULT_MODEL_ID)
            command.add_argument("--tokenizer-revision", help="Pin the tokenizer revision for reproducible chunks.")
            command.add_argument("--token-count", type=int, default=DEFAULT_TOKEN_COUNT)
            command.add_argument("--token-overlap", type=int, default=DEFAULT_TOKEN_OVERLAP)

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
    bm25.add_argument("--max-articles", type=int, help="Stop when the index reaches this total size.")
    bm25.add_argument("--batch-size", type=int, default=10_000)
    bm25.add_argument("--rebuild", action="store_true", help="Clear both lexical indexes before adding articles.")
    semantic = index_types.add_parser("semantic", help="Build the persistent FAISS semantic index.")
    semantic.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    semantic.add_argument("--model-revision")
    semantic.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    semantic.add_argument("--batch-size", type=int, default=64)
    semantic.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    semantic.add_argument("--max-chunks", type=int, help="Stop when vector storage reaches this total size.")
    semantic.add_argument("--device", default=None, help="Sentence Transformers device, such as cpu or cuda.")
    semantic.add_argument("--stage", choices=("vectors", "faiss", "all"), default="all")
    semantic.add_argument("--faiss-type", choices=("flat", "ivfpq"), default="ivfpq")
    semantic.add_argument("--nlist", type=int, default=16_384)
    semantic.add_argument("--pq-m", type=int, default=32)
    semantic.add_argument("--pq-bits", type=int, default=8)
    semantic.add_argument("--train-samples", type=int, default=1_000_000)
    semantic.add_argument("--nprobe", type=int, default=32)
    semantic.add_argument("--checkpoint-shards", type=int, default=1)
    semantic.add_argument(
        "--rebuild-faiss",
        action="store_true",
        help="Discard only FAISS files and rebuild them from retained vector shards.",
    )
    semantic.add_argument(
        "--allow-partial-index",
        action="store_true",
        help="Publish a pilot FAISS index before all vector shards exist.",
    )
    extensions = commands.add_parser("extensions", help="Install trusted native SQLite extensions.")
    extension_commands = extensions.add_subparsers(dest="extension_command", required=True)
    spellfix = extension_commands.add_parser("install-spellfix")
    spellfix.add_argument("--source", required=True, help="Trusted local spellfix DLL/SO file to install.")
    spellfix.add_argument("--sha256", help="Expected SHA-256 checksum, verified before loading.")
    entities = index_types.add_parser("entities", help="Build a spellfix entity-name snapshot after NER.")
    entities.add_argument("--run-id", default="classla-sl")
    entities.add_argument("--batch-size", type=int, default=10000)

    ner = index_types.add_parser("ner", help="Extract and resume article entities in the corpus database.")
    ner.add_argument("--run-id", default="classla-sl", help="Stable extraction run ID; reuse to resume.")
    ner.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ner.add_argument(
        "--max-articles", type=int, help="Total completed article target for this run, including previous work."
    )
    ner.add_argument("--batch-articles", type=int, default=32)
    ner.add_argument("--batch-size", type=int, default=16, help="Sentences per NER model batch.")
    ner.add_argument("--pipeline-depth", type=int, default=3)
    ner.add_argument("--tokenizer-workers", type=int, default=2)
    ner.add_argument("--ner-pipeline-depth", type=int, default=1)
    ner.add_argument("--ner-decoder", choices=("numpy", "numba"), default="numpy")
    ner.add_argument("--classla-resources-dir")

    lemma = index_types.add_parser("lemma", help="Build or resume the CLASSLA lemma BM25 index.")
    lemma.add_argument("--max-articles", type=int, help="Stop when this total number of articles is processed.")
    lemma.add_argument("--batch-articles", type=int, default=50)
    lemma.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CLASSLA worker processes; SQLite writes remain in the parent (default: 1).",
    )
    lemma.add_argument(
        "--pipeline-depth",
        type=int,
        default=3,
        help="Batches kept in the single-process tokenizer/POS/lemma pipeline (default: 3).",
    )
    lemma.add_argument(
        "--tokenizer-workers",
        type=int,
        default=None,
        help="Obeliks subprocesses for single-process staged indexing (default: 1; use 0 for threads).",
    )
    lemma.add_argument("--rebuild", action="store_true", help="Clear the lemma index before processing.")
    _add_classla_batch_options(lemma, lemma_default=200)
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

    benchmark = commands.add_parser("benchmark", help="Run read-only performance benchmarks.")
    benchmark_types = benchmark.add_subparsers(dest="benchmark_type", required=True)
    classla_benchmark = benchmark_types.add_parser("classla", help="Compare CLASSLA process counts.")
    classla_benchmark.add_argument("--articles", type=int, default=500)
    classla_benchmark.add_argument("--batch-articles", type=int, default=50)
    classla_benchmark.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4])
    classla_benchmark.add_argument(
        "--pipeline-depth",
        type=int,
        default=1,
        help="Use bounded staged execution; values above 1 require --workers 1.",
    )
    classla_benchmark.add_argument(
        "--tokenizer-workers",
        type=int,
        default=0,
        help="Persistent Obeliks subprocesses used by staged execution.",
    )
    classla_benchmark.add_argument("--output", help="JSON output path relative to the repository root.")
    _add_classla_options(classla_benchmark)
    _add_classla_batch_options(classla_benchmark)

    profile = commands.add_parser("profile", help="Capture detailed performance profiles.")
    profile_types = profile.add_subparsers(dest="profile_type", required=True)
    classla_profile = profile_types.add_parser("classla", help="Capture a PyTorch CLASSLA trace.")
    classla_profile.add_argument(
        "--articles",
        type=int,
        default=1,
        help="Number of articles to trace; limited to 5 because CLASSLA produces very large traces.",
    )
    classla_profile.add_argument("--output", help="Chrome trace path relative to the repository root.")
    _add_classla_options(classla_profile)
    _add_classla_batch_options(classla_profile)

    search = commands.add_parser("search", help="Run one search and write JSON to stdout.")
    search.add_argument("mode", choices=("bm25", "bm25-lemma", "bm25-combined", "regex", "semantic"))
    search.add_argument("query")
    search.add_argument("--semantic-device", default="cpu", help="Query encoder device (cpu or cuda).")
    search.add_argument("--offset", type=int, default=0, help="Skip ranked BM25 or semantic articles.")
    search.add_argument(
        "--profile",
        action="store_true",
        help="Report search phase timings, SQLite settings, and the query plan to stderr.",
    )
    _add_search_options(search)

    stdio = commands.add_parser("stdio", help="Serve newline-delimited JSON requests on stdin/stdout.")
    stdio.add_argument("--semantic-device", default="cpu", help="Query encoder device (cpu or cuda).")
    stdio.add_argument(
        "--no-semantic",
        action="store_true",
        help="Do not preload FAISS and EmbeddingGemma; semantic requests will load them lazily.",
    )

    read = commands.add_parser(
        "read", help="Read physical lines or a wrapped position from a newspaper source."
    )
    read.add_argument("document_id", help="Short URN component returned by search.")
    read.add_argument("--line-start", type=int)
    read.add_argument("--line-end", type=int)
    read.add_argument("--position", help="Wrapped source position such as 334.7.")
    read.add_argument("--before", type=int, default=2, help="Wrapped lines before position.")
    read.add_argument("--after", type=int, default=8, help="Wrapped lines after position.")
    read.add_argument("--wrap-chars", type=int, default=DEFAULT_SOURCE_WRAP_CHARS)
    read.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_READ_BYTES)

    commands.add_parser("status", help="Report retrieval-index readiness as JSON.")
    return parser


def _add_search_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--before",
        type=int,
        default=0,
        help="Include preceding chunks in semantic-search snippets.",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=0,
        help="Include following chunks in semantic-search snippets.",
    )
    parser.add_argument("--context-lines", type=int, default=0, help="Add source lines around the result span.")
    parser.add_argument(
        "--max-snippet-chars",
        type=int,
        default=DEFAULT_MAX_SNIPPET_CHARS,
        help=f"Maximum characters returned in each snippet (default: {DEFAULT_MAX_SNIPPET_CHARS}).",
    )
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


def _add_classla_batch_options(
    parser: argparse.ArgumentParser,
    *,
    lemma_default: int | None = None,
) -> None:
    parser.add_argument(
        "--classla-pos-batch-size",
        type=int,
        help="Override CLASSLA's POS inference batch size (model default: 5000).",
    )
    parser.add_argument(
        "--classla-lemma-batch-size",
        type=int,
        default=lemma_default,
        help="Override CLASSLA's lemma inference batch size (model default: 50).",
    )


def _resolve_output(root: Path, value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else root / path


def _print_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _configure_console_encoding() -> None:
    """Keep JSON output Unicode-safe on Windows consoles and redirected pipes."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
