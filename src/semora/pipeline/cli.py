"""Portable batch commands: explicit inputs, database destination and reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
            command.add_argument(
                "--include-tables", action="store_true",
                help="Retain HTML tables in derived chunks for a new legacy-policy batch.",
            )
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
            _add_audit_report(command)

    lexical = operations.add_parser("index", help="Add pending lexical entries from an ingested batch.")
    _add_io(lexical)
    _add_maintenance(lexical)
    lexical.add_argument("--batch", required=True)
    lexical.add_argument("--stage", required=True, choices=("surface", "lemma"))
    lexical.add_argument(
        "--progress-file", help="Replace this JSON snapshot after each committed lexical checkpoint."
    )
    lexical.add_argument("--batch-size", type=int, default=1000, help="Articles per committed checkpoint.")
    lexical.add_argument("--classla-type", default="default")
    lexical.add_argument(
        "--classla-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="CLASSLA device (default: cpu); use cuda explicitly.",
    )
    lexical.add_argument(
        "--classla-resources-dir", help="Installed CLASSLA resources, relative to the working directory."
    )
    lexical.add_argument("--classla-pos-batch-size", type=int)
    lexical.add_argument("--classla-lemma-batch-size", type=int)
    lexical.add_argument(
        "--classla-workers", type=int, default=1, help="Parallel CLASSLA processes for lemma indexing."
    )
    lexical.add_argument("--classla-worker-batch-articles", type=int, default=50)
    lexical.add_argument(
        "--profile", action="store_true", help="Emit per-checkpoint phase timings to stderr."
    )
    lexical.add_argument(
        "--trust-completed-state", action="store_true",
        help="Skip full content revalidation of committed lexical rows during maintenance retry.",
    )
    lexical.add_argument(
        "--lemma-model-fingerprint",
        help="Operator-supplied model identity (not computed here); omitted means unknown.",
    )

    tables = operations.add_parser("tables", help="Plan or repair table-bearing articles in one ingested batch.")
    table_operations = tables.add_subparsers(dest="tables_command", required=True)
    for name in ("plan", "repair"):
        command = table_operations.add_parser(name)
        _add_io(command)
        command.add_argument("--batch", required=True)
        command.add_argument("--semantic-dir", help="Semantic mapping directory; defaults beside the database.")
        if name == "repair":
            _add_maintenance(command)

    semantic = operations.add_parser("semantic", help="Audit/adopt vectors and build additive semantic outputs.")
    semantic_operations = semantic.add_subparsers(dest="semantic_command", required=True)
    for name in ("audit", "adopt", "vectors", "faiss"):
        command = semantic_operations.add_parser(name)
        _add_io(command)
        command.add_argument("--semantic-dir", help="Semantic artifact directory; default: <database parent>/semantic.")
        if name != "audit":
            _add_maintenance(command)
        if name == "adopt":
            _add_audit_report(command)
        if name == "vectors":
            command.add_argument("--batch", required=True)
            command.add_argument(
                "--progress-file", help="Replace this JSON snapshot after each committed vector shard."
            )
            command.add_argument("--model-id", default=DEFAULT_MODEL_ID)
            command.add_argument("--model-revision", help="Optional immutable model revision; omitted means unknown.")
            command.add_argument("--dimensions", type=int, default=256)
            command.add_argument("--batch-size", type=int, default=64)
            command.add_argument("--shard-size", type=int, default=10_000)
            command.add_argument("--device", help="SentenceTransformer device, such as cpu or cuda.")
        elif name == "faiss":
            command.add_argument(
                "--progress-file", help="Replace this JSON snapshot after each committed FAISS checkpoint."
            )
            command.add_argument("--index-type", choices=("flat", "ivfpq"), default="ivfpq")
            command.add_argument("--nlist", type=int, default=16_384)
            command.add_argument("--pq-m", type=int, default=32)
            command.add_argument("--pq-bits", type=int, default=8)
            command.add_argument("--train-samples", type=int, default=1_000_000)
            command.add_argument("--nprobe", type=int, default=32)
            command.add_argument("--checkpoint-shards", type=int, default=10)

    ner = operations.add_parser("ner", help="Audit/adopt historical NER and extend a selected batch.")
    ner_operations = ner.add_subparsers(dest="ner_command", required=True)
    for name in ("audit", "adopt", "index"):
        command = ner_operations.add_parser(name)
        _add_io(command)
        command.add_argument("--run-id", default="classla-sl")
        if name != "audit":
            _add_maintenance(command)
        if name == "adopt":
            _add_audit_report(command)
        if name == "index":
            command.add_argument("--batch", required=True)
            command.add_argument("--adapter", default="classla")
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            command.add_argument("--batch-articles", type=int, default=32)
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--pipeline-depth", type=int, default=1)
            command.add_argument("--tokenizer-workers", type=int, default=0)
            command.add_argument("--ner-pipeline-depth", type=int, default=1)
            command.add_argument("--ner-decoder", choices=("numpy", "numba"), default="numpy")
            command.add_argument("--resources", help="Installed adapter resources, relative to the working directory.")

    entities = operations.add_parser("entities", help="Audit/adopt and extend the entity-name dictionary.")
    entity_operations = entities.add_subparsers(dest="entities_command", required=True)
    for name in ("audit", "adopt", "index"):
        command = entity_operations.add_parser(name)
        _add_io(command)
        command.add_argument("--entity-dir", help="Dictionary root; default: <database parent>/entities.")
        command.add_argument("--spellfix-path", help="Trusted local spellfix DLL/SO; default: <root>/extensions.")
        command.add_argument("--run-id", default="classla-sl")
        if name != "audit":
            _add_maintenance(command)
        if name == "adopt":
            _add_audit_report(command)
        if name == "index":
            command.add_argument("--batch", required=True)
            command.add_argument("--batch-articles", type=int, default=1000)

    publication = operations.add_parser("publication", help="Audit a complete private serving revision.")
    _add_io(publication)
    publication.add_argument("--batch", required=True, help="Registered ingested batch ID to certify.")
    publication.add_argument("--semantic-dir", help="Semantic artifact root; default: <database parent>/semantic.")
    publication.add_argument(
        "--semantic-generation", required=True, help="Exact 64-character private FAISS generation ID."
    )
    publication.add_argument("--entity-dir", help="Entity dictionary root; default: <database parent>/entities.")
    publication.add_argument("--run-id", default="classla-sl", help="NER run associated with the dictionary.")
    publication.add_argument("--spellfix-path", help="Trusted local spellfix DLL/SO; default: <root>/extensions.")
    publication.add_argument(
        "--publish", action="store_true", help="Repeat the audit and atomically select this revision."
    )
    _add_maintenance(publication)


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


def _add_audit_report(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--audit-report",
        help="Trust this earlier eligible audit report instead of repeating its full scan."
    )


def _read_audit_report(args: argparse.Namespace) -> dict | None:
    if not args.audit_report:
        return None
    with Path(args.audit_report).open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("Audit report must be a JSON object.")
    return report


def _progress(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)


def _write_progress_file(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _snapshot_progress(path: Path | None, snapshot: dict, events: tuple[str, ...], count_key: str):
    def progress(event: dict) -> None:
        _progress(event)
        if path is not None and event.get("event") in events:
            snapshot["completed"] = event[count_key]
            snapshot["total"] = event["total"]
            _write_progress_file(path, snapshot)

    return progress


def run_pipeline_command(args: argparse.Namespace, default_database: Path) -> None:
    """Execute one operation and keep machine-readable stdout separate from progress."""
    database = Path(args.database).resolve() if args.database else default_database
    protected = [database]
    if hasattr(args, "manifest"):
        protected.append(Path(args.manifest))
    if getattr(args, "audit_report", None):
        protected.append(Path(args.audit_report))
    if args.pipeline_command == "semantic":
        semantic_dir = Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic"
        protected.extend(
            (
                semantic_dir,
                semantic_dir / "mapping.sqlite",
                semantic_dir / "manifest.json",
                semantic_dir / "index.faiss",
            )
        )
    if args.pipeline_command == "tables":
        semantic_dir = Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic"
        protected.append(semantic_dir / "mapping.sqlite")
    if args.pipeline_command == "entities":
        entity_dir = Path(args.entity_dir).resolve() if args.entity_dir else database.parent / "entities"
        protected.extend((entity_dir, entity_dir / "manifest.json"))
        if args.spellfix_path:
            protected.append(Path(args.spellfix_path))
    if args.pipeline_command == "publication":
        semantic_dir = Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic"
        entity_dir = Path(args.entity_dir).resolve() if args.entity_dir else database.parent / "entities"
        protected.extend((semantic_dir, entity_dir, database.parent / "publication.json"))
        if args.spellfix_path:
            protected.append(Path(args.spellfix_path))
    output = report_path(args.output, protected=protected) if args.output else None
    if getattr(args, "progress_file", None):
        progress_file = Path(args.progress_file).resolve()
        if progress_file in {Path(item).resolve() for item in protected} or progress_file == output:
            raise ValueError("Progress file conflicts with an input, database or report.")
    operation = args.pipeline_command
    mutates = (
        operation in ("run", "index")
        or (operation == "tables" and args.tables_command == "repair")
        or (operation == "baseline" and args.baseline_command == "adopt")
        or (operation == "semantic" and args.semantic_command != "audit")
        or (operation == "ner" and args.ner_command != "audit")
        or (operation == "entities" and args.entities_command != "audit")
        or (operation == "publication" and args.publish)
    )
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
    elif operation == "tables":
        from semora.pipeline.tables import repair_batch_tables, table_repair_plan

        semantic_dir = Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic"
        report = (
            table_repair_plan(database, args.batch, semantic_dir=semantic_dir)
            if args.tables_command == "plan"
            else repair_batch_tables(database, args.batch, semantic_dir=semantic_dir, progress=_progress)
        )
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
                exclude_tables=False if args.include_tables else None,
                progress=_progress,
            )
    elif operation == "baseline":
        from semora.pipeline.baseline import adopt_baseline, audit_baseline

        if args.baseline_command == "audit":
            report = audit_baseline(database, progress=_progress)
            success = report["eligible"]
        else:
            report = adopt_baseline(database, progress=_progress, audit_report=_read_audit_report(args))
    elif operation == "semantic":
        semantic_dir = Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic"
        if args.semantic_command == "audit":
            from semora.pipeline.semantic_baseline import audit_semantic_baseline

            report = audit_semantic_baseline(database, semantic_dir, progress=_progress)
            success = report["eligible"]
        elif args.semantic_command == "adopt":
            from semora.pipeline.semantic_adoption import adopt_semantic_baseline

            report = adopt_semantic_baseline(
                database, semantic_dir, progress=_progress, audit_report=_read_audit_report(args)
            )
        elif args.semantic_command == "vectors":
            from semora.pipeline.semantic import index_batch_vectors

            progress_file = Path(args.progress_file).resolve() if args.progress_file else None
            snapshot = {
                "format": "pebble.progress.v1", "title": "Newspaper semantic vectors",
                "state": "running", "completed": 0, "total": 0,
                "unit": "chunks", "message": "",
            }
            report = index_batch_vectors(
                database,
                args.batch,
                semantic_dir,
                model_id=args.model_id,
                model_revision=args.model_revision,
                dimensions=args.dimensions,
                batch_size=args.batch_size,
                shard_size=args.shard_size,
                device=args.device,
                progress=_snapshot_progress(
                    progress_file, snapshot, ("semantic_start", "semantic_checkpoint"), "completed_chunks"
                ),
            )
            if progress_file is not None:
                snapshot["completed"] = report["completed_chunks"]
                snapshot["total"] = report["eligible_chunks"]
                snapshot["state"] = "completed" if report["complete"] else "paused"
                _write_progress_file(progress_file, snapshot)
        else:
            from semora.pipeline.semantic_faiss import build_semantic_generation
            from semora.retrieval.semantic_faiss import FaissBuildConfig

            progress_file = Path(args.progress_file).resolve() if args.progress_file else None
            snapshot = {
                "format": "pebble.progress.v1", "title": "Newspaper FAISS indexing",
                "state": "running", "completed": 0, "total": 0,
                "unit": "vectors", "message": "",
            }
            config = FaissBuildConfig(
                index_type=args.index_type,
                nlist=args.nlist,
                pq_m=args.pq_m,
                pq_bits=args.pq_bits,
                train_samples=args.train_samples,
                nprobe=args.nprobe,
            )
            report = build_semantic_generation(
                database,
                semantic_dir,
                config=config,
                checkpoint_shards=args.checkpoint_shards,
                progress=_snapshot_progress(
                    progress_file, snapshot, ("faiss_generation_start", "faiss_generation_checkpoint"),
                    "indexed_vectors",
                ),
            )
            if progress_file is not None:
                snapshot["completed"] = report["chunks"]
                snapshot["total"] = report["chunks"]
                snapshot["state"] = "completed" if report["index_complete"] else "paused"
                _write_progress_file(progress_file, snapshot)
    elif operation == "ner":
        if args.ner_command == "audit":
            from semora.pipeline.ner_baseline import audit_ner_baseline

            report = audit_ner_baseline(database, run_id=args.run_id, progress=_progress)
            success = report["eligible"]
        elif args.ner_command == "adopt":
            from semora.pipeline.ner_adoption import adopt_ner_baseline

            report = adopt_ner_baseline(
                database, run_id=args.run_id, progress=_progress, audit_report=_read_audit_report(args)
            )
        else:
            from semora.pipeline.ner import index_batch_ner

            report = index_batch_ner(
                database,
                args.batch,
                run_id=args.run_id,
                adapter=args.adapter,
                device=args.device,
                batch_articles=args.batch_articles,
                batch_size=args.batch_size,
                pipeline_depth=args.pipeline_depth,
                tokenizer_workers=args.tokenizer_workers,
                ner_pipeline_depth=args.ner_pipeline_depth,
                ner_decoder=args.ner_decoder,
                resources=args.resources,
                progress=_progress,
            )
    elif operation == "entities":
        entity_dir = Path(args.entity_dir).resolve() if args.entity_dir else database.parent / "entities"
        spellfix = Path(args.spellfix_path).resolve() if args.spellfix_path else None
        if args.entities_command == "audit":
            from semora.pipeline.entity_baseline import audit_entity_baseline

            report = audit_entity_baseline(
                database,
                entity_dir,
                run_id=args.run_id,
                spellfix_path=spellfix,
                progress=_progress,
            )
            success = report["eligible"]
        elif args.entities_command == "adopt":
            from semora.pipeline.entity_adoption import adopt_entity_baseline

            report = adopt_entity_baseline(
                database,
                entity_dir,
                run_id=args.run_id,
                spellfix_path=spellfix,
                progress=_progress,
                audit_report=_read_audit_report(args),
            )
        else:
            from semora.pipeline.entities import index_batch_entities

            report = index_batch_entities(
                database,
                args.batch,
                entity_dir,
                run_id=args.run_id,
                batch_articles=args.batch_articles,
                spellfix_path=spellfix,
                progress=_progress,
            )
    elif operation == "publication":
        if args.publish:
            from semora.pipeline.publication_publish import publish_batch
            action = publish_batch
        else:
            from semora.pipeline.publication_audit import audit_publication
            action = audit_publication

        report = action(
            database,
            args.batch,
            Path(args.semantic_dir).resolve() if args.semantic_dir else database.parent / "semantic",
            args.semantic_generation,
            Path(args.entity_dir).resolve() if args.entity_dir else database.parent / "entities",
            run_id=args.run_id,
            spellfix_path=Path(args.spellfix_path).resolve() if args.spellfix_path else None,
            progress=_progress,
        )
        success = report["eligible"] and (not args.publish or report["published"])
    else:
        from semora.pipeline.lexical import index_batch_lexical

        progress_file = Path(args.progress_file).resolve() if args.progress_file else None
        snapshot = {
            "format": "pebble.progress.v1",
            "title": f"Newspaper {args.stage} indexing",
            "state": "running",
            "completed": 0,
            "total": 0,
            "unit": "articles",
            "message": "",
        }

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
            classla_workers=args.classla_workers,
            worker_batch_articles=args.classla_worker_batch_articles,
            profile=args.profile,
            trust_completed_state=args.trust_completed_state,
            model_fingerprint=args.lemma_model_fingerprint,
            progress=_snapshot_progress(
                progress_file, snapshot, ("lexical_start", "lexical_checkpoint"), "completed_articles"
            ),
        )
        if progress_file is not None:
            snapshot["completed"] = report["completed_articles"]
            snapshot["total"] = report["eligible_articles"]
            snapshot["state"] = "completed" if report["complete"] else "paused"
            _write_progress_file(progress_file, snapshot)

    return report, success
