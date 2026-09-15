"""python -m semora.diagnostics.ner {sample,run,sweep,summarize,report}."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .report import report
from .runner import run
from .sample import create_sample
from .summary import summarize


def stop_worker(process: subprocess.Popen) -> None:
    """Reap the complete worker tree, including Windows virtualenv launchers."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Create a reusable random article sample (source read-only)")
    sample.add_argument("--database", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--articles", type=int, default=10_000)
    sample.add_argument("--seed", type=int, default=42)
    for name in ("run", "sweep"):
        command = commands.add_parser(name)
        command.add_argument("--sample", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--adapter", required=True, help="classla, transformers, or module:factory")
        command.add_argument("--model")
        command.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        command.add_argument("--batch-articles", type=int, default=32)
        command.add_argument(
            "--sentence-order",
            choices=["input", "length"],
            default="input",
            help="CLASSLA sentence grouping; length can change near-tied predictions",
        )
        command.add_argument(
            "--pipeline-depth", type=int, default=1, help="Bounded overlapping CLASSLA batches; 1 is sequential"
        )
        command.add_argument(
            "--tokenizer-workers", type=int, default=0, help="CPU-only CLASSLA tokenizer processes; 0 tokenizes locally"
        )
        command.add_argument("--limit", type=int)
        command.add_argument("--corpus-articles", type=int)
        command.add_argument("--resources", help="CLASSLA resources directory")
        command.add_argument("--download", action="store_true", help="Download CLASSLA NER resources explicitly")
        command.add_argument("--revision", default="main")
        command.add_argument("--max-length", type=int, default=256)
        command.add_argument("--stride", type=int, default=64)
        command.add_argument("--pretokenized", action="store_true", help="Lossless pretokenization for HF models")
        command.add_argument(
            "--aggregation",
            choices=["simple", "first", "average", "max"],
            help="HF subword aggregation; Slavic checkpoint defaults to first, other models to simple",
        )
        command.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
        if name == "run":
            command.add_argument("--batch-size", type=int, default=32)
        else:
            command.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--timeout", type=int, default=7200, help="Per-process timeout in seconds")
    summary = commands.add_parser("summarize", help="Aggregate repeats and record failed run statuses")
    summary.add_argument("--runs", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("report")
    comparison.add_argument("--sample", type=Path, required=True)
    comparison.add_argument("--runs", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--names", nargs="+")
    comparison.add_argument("--limit", type=int, default=40)
    comparison.add_argument("--seed", type=int, default=42)
    comparison.add_argument("--labels", nargs="+", default=["PER", "LOC", "ORG"])
    comparison.add_argument("--queries", nargs="+", help="Optional literal names for cached prediction spot checks")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "sample":
        args["count"] = args.pop("articles")
        create_sample(**args)
    elif command == "run":
        run(**args)
    elif command == "report":
        report(**args)
    elif command == "summarize":
        summarize(**args)
    else:
        sizes, repeats, timeout = args.pop("batch_sizes"), args.pop("repeats"), args.pop("timeout")
        output = args.pop("output")
        if repeats <= 0 or timeout <= 0 or any(size <= 0 for size in sizes):
            parser.error("Sizes, repeats, and timeout must be positive")
        output.mkdir(parents=True, exist_ok=False)
        failed = False
        for size in sizes:
            for repeat in range(1, repeats + 1):
                target = output / f"batch-{size}-repeat-{repeat}"
                argv = [
                    sys.executable,
                    "-m",
                    "semora.diagnostics.ner",
                    "run",
                    "--output",
                    str(target),
                    "--batch-size",
                    str(size),
                ]
                for key, value in args.items():
                    if value is True:
                        argv.append("--" + key.replace("_", "-"))
                    elif value is not None and value is not False:
                        argv.extend(["--" + key.replace("_", "-"), str(value)])
                with (output / f"{target.name}.log").open("w", encoding="utf-8") as log:
                    print(f"Running {target.name}; log: {log.name}", flush=True)
                    process = subprocess.Popen(
                        argv,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                        start_new_session=os.name != "nt",
                    )
                    try:
                        returncode = process.wait(timeout=timeout)
                        failed = failed or returncode != 0
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                        stop_worker(process)
                        message = "Benchmark timed out" if isinstance(exc, subprocess.TimeoutExpired) else "Interrupted"
                        log.write(f"\n{message}; incomplete run must not be compared.\n")
                        target.mkdir(exist_ok=True)
                        metrics_path = target / "metrics.json"
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
                        metrics.update(status="failed", error=message)
                        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                        failed = True
                        if isinstance(exc, KeyboardInterrupt):
                            raise
        if failed:
            raise SystemExit("Some runs failed or timed out; inspect sweep logs and metrics.json")


if __name__ == "__main__":
    main()
