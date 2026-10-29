"""Explicit JSON report files, published atomically without overwriting."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def report_path(path: str | Path, *, protected: Iterable[str | Path] = ()) -> Path:
    """Resolve a CWD-relative output and reject existing/protected targets."""
    requested = Path(path)
    if requested.is_symlink():
        raise FileExistsError(f"Report output already exists: {requested}")
    target = requested.resolve()
    if target in {Path(item).resolve() for item in protected}:
        raise ValueError(f"Report output conflicts with an input or database: {target}")
    if target.exists():
        raise FileExistsError(f"Report output already exists: {target}")
    return target


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Publish complete UTF-8 JSON; a concurrent creator is never replaced."""
    target = report_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
