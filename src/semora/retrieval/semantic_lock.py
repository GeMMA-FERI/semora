"""Cross-process advisory locks for long-running semantic build stages."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import BinaryIO


class SemanticBuildLock:
    def __init__(self, path: str | Path, stage: str) -> None:
        self.path = Path(path)
        self.stage = stage
        self._file: BinaryIO | None = None

    def __enter__(self) -> SemanticBuildLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a+b")
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            _lock_file(lock_file)
        except OSError as exc:
            owner = ""
            try:
                lock_file.seek(0)
                owner = lock_file.read().decode("ascii", errors="replace").strip("\0\r\n ")
            except OSError:
                pass
            lock_file.close()
            detail = f" (reported owner PID: {owner})" if owner else ""
            raise RuntimeError(
                f"Another Semora {self.stage} build is already using {self.path.parent}{detail}."
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n".encode("ascii"))
        lock_file.flush()
        os.fsync(lock_file.fileno())
        lock_file.seek(0)
        self._file = lock_file
        return self

    def __exit__(self, *_args: object) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            self._file.truncate()
            self._file.write(b"\0")
            self._file.flush()
            _unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None


def _lock_file(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl = import_module("fcntl")

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl = import_module("fcntl")

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
