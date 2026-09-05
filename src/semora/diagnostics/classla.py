"""Read-only CLASSLA profiling and multi-process throughput benchmarks."""

from __future__ import annotations

import ctypes
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from semora.text import ClasslaLemmatizer


@dataclass(frozen=True)
class WorkerResult:
    pid: int
    documents: int
    tokens: int
    started_at: float
    finished_at: float
    initialization_seconds: float
    peak_cuda_bytes: int
    process_rss_bytes: int | None


_WORKER_LEMMATIZER: ClasslaLemmatizer | None = None
_WORKER_INITIALIZATION_SECONDS = 0.0


def load_classla_workload(database_path: str | Path, articles: int) -> list[str]:
    """Load a deterministic representative workload without modifying SQLite."""
    if articles <= 0:
        raise ValueError("articles must be positive.")
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT articles.title, articles.content
            FROM articles
            WHERE articles.is_valid = 1
              AND articles.char_end IS NOT NULL
              AND EXISTS (
                  SELECT 1
                  FROM chunks
                  JOIN chunk_fts_map ON chunk_fts_map.chunk_id = chunks.chunk_id
                  WHERE chunks.article_id = articles.article_id
              )
            ORDER BY articles.article_id
            LIMIT ?
            """,
            (articles,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("No surface-indexed articles are available for the CLASSLA workload.")
    return [f"{title}\n{content}" if title else str(content) for title, content in rows]


def benchmark_classla(
    texts: list[str],
    *,
    worker_counts: list[int],
    batch_articles: int,
    pipeline_type: str = "default",
    device: str = "cuda",
    resources_dir: str | Path | None = None,
    pos_batch_size: int | None = None,
    lemma_batch_size: int | None = None,
) -> dict[str, Any]:
    """Compare isolated CLASSLA process counts over the same in-memory workload."""
    if not worker_counts or any(workers <= 0 for workers in worker_counts):
        raise ValueError("worker counts must be positive.")
    if batch_articles <= 0:
        raise ValueError("batch_articles must be positive.")
    batches = [texts[index : index + batch_articles] for index in range(0, len(texts), batch_articles)]
    config = {
        "pipeline_type": pipeline_type,
        "device": device,
        "resources_dir": str(resources_dir) if resources_dir is not None else None,
        "pos_batch_size": pos_batch_size,
        "lemma_batch_size": lemma_batch_size,
    }
    runs = []
    for workers in worker_counts:
        print(f"Benchmarking CLASSLA with {workers} worker(s)...", file=sys.stderr, flush=True)
        sampler = _NvidiaSampler()
        sampler.start()
        wall_started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config,),
        ) as executor:
            futures = [executor.submit(_process_worker_batch, batch) for batch in batches]
            results = [future.result() for future in as_completed(futures)]
        wall_seconds = time.perf_counter() - wall_started
        samples = sampler.stop()
        steady_started = min(result.started_at for result in results)
        steady_finished = max(result.finished_at for result in results)
        steady_seconds = steady_finished - steady_started
        steady_samples = [sample for sample in samples if steady_started <= sample[0] <= steady_finished]
        process_ids = sorted({result.pid for result in results})
        initialization = {
            str(pid): max(result.initialization_seconds for result in results if result.pid == pid)
            for pid in process_ids
        }
        runs.append(
            {
                "workers": workers,
                "processes_used": len(process_ids),
                "documents": sum(result.documents for result in results),
                "tokens": sum(result.tokens for result in results),
                "wall_seconds_including_initialization": wall_seconds,
                "steady_seconds": steady_seconds,
                "documents_per_second": len(texts) / steady_seconds,
                "tokens_per_second": sum(result.tokens for result in results) / steady_seconds,
                "worker_initialization_seconds": initialization,
                "worker_peak_cuda_mib": max(result.peak_cuda_bytes for result in results) / 1024**2,
                "worker_peak_rss_mib": _optional_max(result.process_rss_bytes for result in results),
                **_summarize_gpu_samples(steady_samples),
            }
        )
    return {
        "format": "semora.classla-benchmark.v1",
        "articles": len(texts),
        "batch_articles": batch_articles,
        "characters": sum(len(text) for text in texts),
        "pipeline_type": pipeline_type,
        "device": device,
        "pos_batch_size": pos_batch_size,
        "lemma_batch_size": lemma_batch_size,
        "runs": runs,
    }


def profile_classla(
    texts: list[str],
    output_path: str | Path,
    *,
    pipeline_type: str = "default",
    device: str = "cuda",
    resources_dir: str | Path | None = None,
    pos_batch_size: int | None = None,
    lemma_batch_size: int | None = None,
) -> dict[str, Any]:
    """Write a PyTorch CPU/CUDA Chrome trace for one warmed CLASSLA batch."""
    if len(texts) > 5:
        raise ValueError(
            "CLASSLA operator profiling is limited to five articles because its recurrent POS model "
            "generates very large traces. Use 'semora index lemma --profile' for larger workloads."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CLASSLA profiling requires PyTorch.") from exc
    lemmatizer = ClasslaLemmatizer(
        pipeline_type=pipeline_type,
        device=device,
        resources_dir=resources_dir,
        pos_batch_size=pos_batch_size,
        lemma_batch_size=lemma_batch_size,
    )
    lemmatizer.annotate("Kratek preizkus za ogrevanje.")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device != "cpu" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        annotated = lemmatizer.annotate_many(texts)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(target))
    table_path = target.with_suffix(".txt")
    sort_key = "self_cuda_time_total" if len(activities) > 1 else "self_cpu_time_total"
    table_path.write_text(
        profiler.key_averages().table(sort_by=sort_key, row_limit=50),
        encoding="utf-8",
    )
    profile = lemmatizer.last_profile
    return {
        "format": "semora.classla-profile.v1",
        "trace": str(target),
        "operator_table": str(table_path),
        "documents": len(annotated),
        "tokens": sum(len(document) for document in annotated),
        "classla_stages": asdict(profile) if profile is not None else None,
    }


def save_diagnostics(value: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _initialize_worker(config: dict[str, Any]) -> None:
    global _WORKER_INITIALIZATION_SECONDS, _WORKER_LEMMATIZER
    started = time.perf_counter()
    _WORKER_LEMMATIZER = ClasslaLemmatizer(**config)
    _WORKER_INITIALIZATION_SECONDS = time.perf_counter() - started


def _process_worker_batch(texts: list[str]) -> WorkerResult:
    if _WORKER_LEMMATIZER is None:
        raise RuntimeError("CLASSLA benchmark worker was not initialized.")
    started = time.perf_counter()
    annotated = _WORKER_LEMMATIZER.annotate_many(texts)
    finished = time.perf_counter()
    profile = _WORKER_LEMMATIZER.last_profile
    return WorkerResult(
        pid=os.getpid(),
        documents=len(annotated),
        tokens=sum(len(document) for document in annotated),
        started_at=started,
        finished_at=finished,
        initialization_seconds=_WORKER_INITIALIZATION_SECONDS,
        peak_cuda_bytes=profile.peak_cuda_bytes if profile is not None else 0,
        process_rss_bytes=_process_rss_bytes(),
    )


class _NvidiaSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float, float, float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 3)
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                utilization, memory, power = (
                    float(value.strip())
                    for value in result.stdout.splitlines()[0].split(",")
                )
                self.samples.append((time.perf_counter(), utilization, memory, power))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                return
            self._stop.wait(self.interval_seconds)


def _summarize_gpu_samples(samples: list[tuple[float, float, float, float]]) -> dict[str, float | None]:
    if not samples:
        return {
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_peak_percent": None,
            "gpu_memory_peak_mib": None,
            "gpu_power_mean_watts": None,
        }
    return {
        "gpu_utilization_mean_percent": sum(sample[1] for sample in samples) / len(samples),
        "gpu_utilization_peak_percent": max(sample[1] for sample in samples),
        "gpu_memory_peak_mib": max(sample[2] for sample in samples),
        "gpu_power_mean_watts": sum(sample[3] for sample in samples) / len(samples),
    }


def _optional_max(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) / 1024**2 if present else None


def _process_rss_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    get_process_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_process_memory.restype = ctypes.c_bool
    handle = get_current_process()
    if not get_process_memory(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)
