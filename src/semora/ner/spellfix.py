"""Explicit installation and per-connection loading of a trusted SQLite extension."""

import hashlib
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


def extension_path(root):
    override = os.environ.get("SEMORA_SPELLFIX_PATH")
    return Path(override).resolve() if override else Path(root) / "extensions" / (
        "spellfix1.dll" if os.name == "nt" else "spellfix1.so"
    )


def load_spellfix(connection, path):
    path = Path(path).resolve(strict=True)
    connection.enable_load_extension(True)
    try:
        connection.load_extension(str(path))
    finally:
        connection.enable_load_extension(False)


def install_spellfix(root, source, sha256=None):
    """Validate a user-selected native library using an in-memory database, then copy it."""
    source = Path(source).resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha256 and digest != sha256.lower():
        raise ValueError("Spellfix library checksum mismatch")
    destination = extension_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never download/compile executable code implicitly during a search.
    with sqlite3.connect(":memory:") as connection:
        load_spellfix(connection, source)
        connection.execute("CREATE VIRTUAL TABLE probe USING spellfix1")
    connection.close()
    if source != destination.resolve():
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=destination.suffix)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest}
