"""python -m semora.diagnostics.ner {sample,run,sweep,report}."""

import argparse
import subprocess
import sys
from pathlib import Path

from .report import report
from .runner import run
from .sample import create_sample


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
        command.add_argument("--limit", type=int)
        command.add_argument("--corpus-articles", type=int)
        command.add_argument("--resources", help="CLASSLA resources directory")
        command.add_argument("--download", action="store_true", help="Download CLASSLA NER resources explicitly")
        command.add_argument("--revision", default="main")
        command.add_argument("--max-length", type=int, default=256)
        command.add_argument("--stride", type=int, default=64)
        command.add_argument("--pretokenized", action="store_true", help="Lossless pretokenization for HF models")
        command.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
        if name == "run":
            command.add_argument("--batch-size", type=int, default=32)
        else:
            command.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--timeout", type=int, default=7200, help="Per-process timeout in seconds")
    comparison = commands.add_parser("report")
    comparison.add_argument("--sample", type=Path, required=True)
    comparison.add_argument("--runs", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--names", nargs="+")
    comparison.add_argument("--limit", type=int, default=40)
    comparison.add_argument("--seed", type=int, default=42)
    comparison.add_argument("--labels", nargs="+", default=["PER", "LOC", "ORG"])
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "sample":
        args["count"] = args.pop("articles")
        create_sample(**args)
    elif command == "run":
        run(**args)
    elif command == "report":
        report(**args)
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
                    try:
                        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                        failed = failed or result.returncode != 0
                    except subprocess.TimeoutExpired:
                        log.write("\nBenchmark timed out; incomplete run must not be compared.\n")
                        failed = True
        if failed:
            raise SystemExit("Some runs failed or timed out; inspect sweep logs and metrics.json")


if __name__ == "__main__":
    main()
