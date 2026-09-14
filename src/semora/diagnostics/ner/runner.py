"""One model/configuration per process; durable outputs and explicit failure status."""

import importlib.metadata
import json
import platform
import sqlite3
import threading
import time
import traceback
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import load_adapter
from .base import validate
from .sample import readonly


class MemorySampler:
    def __init__(self):
        import psutil

        self.process = psutil.Process()
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop_event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def run(
    sample: Path,
    output: Path,
    adapter: str,
    *,
    batch_size: int = 32,
    batch_articles: int = 32,
    device: str = "cuda",
    limit: int | None = None,
    corpus_articles: int | None = None,
    **options: Any,
) -> None:
    import torch
    from tqdm import tqdm

    if batch_size <= 0 or batch_articles <= 0 or (limit is not None and limit <= 0):
        raise ValueError("Batch sizes and limit must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, Any] = {
        "status": "running",
        "adapter": adapter,
        "options": options,
        "batch_size": batch_size,
        "batch_articles": batch_articles,
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "versions": {},
        "completed_articles": 0,
        "characters": 0,
        "entities": 0,
        "read_seconds": 0.0,
        "predict_seconds": 0.0,
        "write_seconds": 0.0,
        "timing_scope": "sample SQLite reads + extraction + indexed SQLite writes/commits; warm cache possible",
    }
    for package in ("torch", "transformers", "classla", "semora"):
        try:
            metrics["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metrics_path = output / "metrics.json"

    def save():
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(metrics_path)

    save()
    try:
        with closing(readonly(sample)) as source, closing(sqlite3.connect(output / "predictions.sqlite")) as target:
            metadata = json.loads(source.execute("SELECT value FROM metadata").fetchone()[0])
            metrics["sample"] = metadata
            total = min(limit or metadata["articles"], metadata["articles"])
            metrics["expected_articles"] = total
            target.executescript("""
                CREATE TABLE predictions (
                    article_id TEXT PRIMARY KEY, entities_json TEXT NOT NULL
                );
                CREATE TABLE mentions (
                    article_id TEXT, start INTEGER, end INTEGER, label TEXT, text TEXT, score REAL
                );
                CREATE INDEX mentions_name ON mentions(text, label);
                CREATE INDEX mentions_article ON mentions(article_id);
            """)
            with MemorySampler() as memory:
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                model = load_adapter(adapter, device=device, **options)
                if device == "cuda":
                    torch.cuda.synchronize()
                metrics["load_seconds"] = time.perf_counter() - started
                metrics["model"] = model.metadata
                warmup = source.execute("SELECT content FROM articles ORDER BY position LIMIT 2").fetchall()
                started = time.perf_counter()
                warm = model.predict([row["content"] for row in warmup], batch_size)
                if len(warm.entities) != len(warmup):
                    raise ValueError("Adapter returned the wrong number of warm-up documents")
                for row, entities in zip(warmup, warm.entities, strict=True):
                    validate(row["content"], entities)
                metrics["warmup_seconds"] = time.perf_counter() - started
                save()
                started = time.perf_counter()
                with tqdm(total=total, unit="article", desc=adapter) as progress:
                    for offset in range(0, total, batch_articles):
                        tick = time.perf_counter()
                        rows = source.execute(
                            "SELECT article_id, content FROM articles WHERE position >= ? "
                            "AND position < ? ORDER BY position",
                            (offset, min(offset + batch_articles, total)),
                        ).fetchall()
                        metrics["read_seconds"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                        prediction = model.predict([r["content"] for r in rows], batch_size)
                        if len(prediction.entities) != len(rows):
                            raise ValueError("Adapter returned the wrong number of documents")
                        for row, entities in zip(rows, prediction.entities, strict=True):
                            validate(row["content"], entities)
                        metrics["predict_seconds"] += time.perf_counter() - tick
                        for key, value in prediction.timings.items():
                            metrics[key] = metrics.get(key, 0) + value
                        tick = time.perf_counter()
                        with target:
                            for row, entities in zip(rows, prediction.entities, strict=True):
                                target.execute(
                                    "INSERT INTO predictions VALUES (?, ?)",
                                    (
                                        row["article_id"],
                                        json.dumps([asdict(e) for e in entities], ensure_ascii=False),
                                    ),
                                )
                                target.executemany(
                                    "INSERT INTO mentions VALUES (?, ?, ?, ?, ?, ?)",
                                    [(row["article_id"], e.start, e.end, e.label, e.text, e.score) for e in entities],
                                )
                        metrics["write_seconds"] += time.perf_counter() - tick
                        metrics["completed_articles"] += len(rows)
                        metrics["characters"] += sum(len(r["content"]) for r in rows)
                        metrics["entities"] += sum(len(e) for e in prediction.entities)
                        metrics["elapsed_seconds"] = time.perf_counter() - started
                        metrics["peak_process_rss_bytes"] = memory.peak
                        save()
                        progress.update(len(rows))
                elapsed = time.perf_counter() - started
                metrics.update(
                    status="complete",
                    elapsed_seconds=elapsed,
                    articles_per_second=total / elapsed,
                    characters_per_second=metrics["characters"] / elapsed,
                    peak_process_rss_bytes=memory.peak,
                )
                if device == "cuda":
                    metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                if corpus_articles:
                    metrics["estimated_corpus_hours"] = corpus_articles / (total / elapsed) / 3600
                    metrics["estimated_corpus_articles"] = corpus_articles
                    metrics["estimate_note"] = "Uniform article sample; not a measured production indexing runtime"
            metrics["prediction_database_bytes"] = (output / "predictions.sqlite").stat().st_size
    except BaseException as exc:
        metrics.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({key: value for key, value in metrics.items() if key != "sample"}, indent=2, default=str))
