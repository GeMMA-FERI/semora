"""Durable diagnostics for semantic vector generation."""

from __future__ import annotations

import faulthandler
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any


class SemanticVectorJournal:
    def __init__(self, directory: str | Path) -> None:
        target = Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=True)
        self.path = target / "vector-build.jsonl"
        self.fault_path = target / "vector-build-fault.log"
        self._output = self.path.open("a", encoding="utf-8", newline="\n")
        self._fault_output = self.fault_path.open("a", encoding="utf-8", newline="\n")
        self._enabled_fault_handler = not faulthandler.is_enabled()
        if self._enabled_fault_handler:
            faulthandler.enable(file=self._fault_output, all_threads=True)

    def write(self, event: str, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "event": event,
            **details,
            **_cuda_memory(),
        }
        json.dump(record, self._output, ensure_ascii=False, sort_keys=True)
        self._output.write("\n")
        self._output.flush()
        os.fsync(self._output.fileno())

    def close(self) -> None:
        if self._enabled_fault_handler:
            faulthandler.disable()
        self._fault_output.close()
        self._output.close()

    def __enter__(self) -> SemanticVectorJournal:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            if error is not None:
                self.write(
                    "build_failed",
                    error_type=error_type.__name__ if error_type else type(error).__name__,
                    error=str(error)[:2_000],
                )
        except Exception:
            pass
        finally:
            self.close()


def _cuda_memory() -> dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {}
