"""Portable batch commands: explicit inputs, database destination and reports."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from semora.corpus import DEFAULT_MODEL_ID, DEFAULT_TOKEN_COUNT, DEFAULT_TOKEN_OVERLAP
from semora.pipeline.reports import report_path, write_report


def add_pipeline_parser(commands: argparse._SubParsersAction) -> None:
    pipeline = commands.add_parser("pipeline", help="Plan, adopt and extend a newspaper corpus.")
    operations = pipeline.add_subparsers(dest="pipeline_command", required=True)
    for name in ("plan", "run", "status"):
        command = operations.add_parser(name)
        _add_io(command)
        if name == "status":
            command.add_argument("--batch", required=True)
            continue
        command.add_argument(
            "--manifest", required=True, help="Batch JSON file; path relative to the working directory."
        )
        command.add_argument(
            "--input-root", required=True, help="Directory resolving source paths inside the manifest."
        )
        if name == "run":
            _add_maintenance(command)
            command.add_argument("--through", choices=("newspapers", "articles", "chunks"), default="chunks")
            command.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Chunk tokenizer model identity.")
            command.add_argument("--tokenizer-revision", help="Pin an immutable tokenizer revision.")
            command.add_argument("--token-count", type=int, default=DEFAULT_TOKEN_COUNT)
            command.add_argument("--token-overlap", type=int, default=DEFAULT_TOKEN_OVERLAP)

    baseline = operations.add_parser("baseline", help="Audit/adopt historical stored outputs without rerunning NLP.")
    baseline_operations = baseline.add_subparsers(dest="baseline_command", required=True)
    for name in ("audit", "adopt"):
        command = baseline_operations.add_parser(name)
        _add_io(command)
        if name == "adopt":
            _add_maintenance(command)

    lexical = operations.add_parser("index", help="Add pending lexical entries from an ingested batch.")
    _add_io(lexical)
    _add_maintenance(lexical)
    lexical.add_argument("--batch", required=True)
    lexical.add_argument("--stage", required=True, choices=("surface", "lemma"))
    lexical.add_argument("--batch-size", type=int, default=1000, help="Articles per committed checkpoint.")
    lexical.add_argument("--classla-type", default="default")
    lexical.add_argument(
        "--classla-device", choices=("auto", "cpu", "cuda"), default="cpu",
        help="CLASSLA device (default: cpu); use cuda explicitly.",
    )
    lexical.add_argument(
        "--classla-resources-dir", help="Installed CLASSLA resources, relative to the working directory."
    )
    lexical.add_argument("--classla-pos-batch-size", type=int)
    lexical.add_argument("--classla-lemma-batch-size", type=int)
    lexical.add_argument(
        "--lemma-model-fingerprint",
        help="Operator-supplied model identity (not computed here); omitted means unknown.",
    )


def _add_io(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--database",
        help="SQLite path relative to the working directory; default: <root>/indexes/semora.sqlite.",
    )
    command.add_argument(
        "--output",
        help="New JSON report file, relative to the working directory; never overwrites a file.",
    )


def _add_maintenance(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--maintenance",
        action="store_true",
        help="Confirm retrieval readers and legacy index writers are stopped.",
    )


def _progress(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)


def run_pipeline_command(args: argparse.Namespace, default_database: Path) -> None:
    """Execute one operation and keep machine-readable stdout separate from progress."""
    database = Path(args.database).resolve() if args.database else default_database
    protected = [database]
    if hasattr(args, "manifest"):
        protected.append(Path(args.manifest))
    output = report_path(args.output, protected=protected) if args.output else None
    operation = args.pipeline_command
    mutates = operation in ("run", "index") or (operation == "baseline" and args.baseline_command == "adopt")
    if mutates and not args.maintenance:
        raise ValueError("Stop retrieval readers and legacy index writers, then pass --maintenance.")

    with redirect_stdout(sys.stderr):
        report, success = _execute_operation(args, database)

    if output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        target = write_report(output, report)
        print(json.dumps({"report_path": str(target)}, ensure_ascii=False))
    if not success:
        raise SystemExit(1)


def _execute_operation(args: argparse.Namespace, database: Path) -> tuple[dict, bool]:
    from semora.pipeline.batches import batch_status, plan_batch, run_batch
    from semora.pipeline.manifest import load_manifest

    operation = args.pipeline_command
    success = True
    if operation == "status":
        report = batch_status(database, args.batch)
    elif operation in ("plan", "run"):
        manifest = load_manifest(args.manifest)
        input_root = Path(args.input_root).resolve()
        if operation == "plan":
            report = plan_batch(manifest, input_root, database, progress=_progress)
            success = report["counts"]["conflict"] == 0
        else:
            stages = ("newspapers", "articles", "chunks")
            report = run_batch(
                manifest,
                input_root,
                database,
                stages=stages[: stages.index(args.through) + 1],
                model_id=args.model_id,
                token_count=args.token_count,
                token_overlap=args.token_overlap,
                tokenizer_revision=args.tokenizer_revision,
                progress=_progress,
            )
    elif operation == "baseline":
        from semora.pipeline.baseline import adopt_baseline, audit_baseline

        if args.baseline_command == "audit":
            report = audit_baseline(database, progress=_progress)
            success = report["eligible"]
        else:
            report = adopt_baseline(database, progress=_progress)
    else:
        from semora.pipeline.lexical import index_batch_lexical

        report = index_batch_lexical(
            database,
            args.batch,
            stage=args.stage,
            batch_size=args.batch_size,
            pipeline_type=args.classla_type,
            device=args.classla_device,
            resources_dir=args.classla_resources_dir,
            pos_batch_size=args.classla_pos_batch_size,
            lemma_batch_size=args.classla_lemma_batch_size,
            model_fingerprint=args.lemma_model_fingerprint,
            progress=_progress,
        )

    return report, success
