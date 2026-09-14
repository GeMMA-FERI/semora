"""Aggregate independent repeats; never count failed runs as fast runs."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(runs: list[Path], output: Path) -> None:
    paths = sorted({path.resolve() for root in runs for path in root.rglob("metrics.json")})
    if not paths:
        raise ValueError("No metrics.json files found")
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    statuses = []
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        statuses.append({"run": str(path.parent), "status": item["status"], "error": item.get("error", "")})
        if item["status"] != "complete":
            continue
        # Config, sample prefix, package versions and hardware must all match.
        configuration = {
            key: item[key]
            for key in (
                "adapter",
                "model",
                "batch_size",
                "batch_articles",
                "device",
                "hardware",
                "versions",
                "completed_articles",
            )
        }
        configuration["sample_sha256"] = item["sample"]["sample_sha256"]
        configuration["estimated_corpus_articles"] = item.get("estimated_corpus_articles")
        groups[json.dumps(configuration, sort_keys=True)].append(item)
    values = [
        "articles_per_second",
        "characters_per_second",
        "elapsed_seconds",
        "load_seconds",
        "peak_process_rss_bytes",
        "peak_cuda_allocated_bytes",
        "estimated_corpus_hours",
    ]
    summaries = []
    for configuration_json, items in groups.items():
        config = json.loads(configuration_json)
        row = {
            "adapter": config["adapter"],
            "model": config["model"].get("model", config["model"].get("package")),
            "articles": config["completed_articles"],
            "batch_articles": config["batch_articles"],
            "device": config["device"],
            "precision": config["model"].get("precision", "native"),
            "aggregation": config["model"].get("aggregation", "native"),
            "sample_sha256": config["sample_sha256"],
            "batch_size": config["batch_size"],
            "repeats": len(items),
            "configuration": config,
        }
        for key in values:
            numbers = [item[key] for item in items if item.get(key) is not None]
            row[key + "_mean"] = statistics.mean(numbers) if numbers else None
            row[key + "_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        summaries.append(row)
    summaries.sort(key=lambda row: (row["adapter"], -(row["articles_per_second_mean"] or 0)))
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    if summaries:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key in summaries[0] if key != "configuration"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
    with (output / "statuses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "status", "error"])
        writer.writeheader()
        writer.writerows(statuses)
    print(f"Summarized {len(paths)} runs into {len(summaries)} completed configuration groups: {output}")
