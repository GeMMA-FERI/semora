"""Evidence for the semantics of a loaded local CLASSLA NER pipeline."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import ntpath
from pathlib import Path
from typing import Any

_TUNING = {
    "device", "use_gpu", "batch_size", "batch_articles", "pipeline_depth", "ner_pipeline_depth",
    "tokenizer_workers", "num_threads", "decoder", "ner_decoder", "sort_during_eval", "sentence_order",
}
_EXCLUDED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _digest_file(path: Path, *, normalize_python: bool = False) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required identity file is missing or a symlink: {path}")
    if normalize_python and path.suffix == ".py":
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Required package directory is missing or a symlink: {root}")
    files: dict[str, str] = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            if path.name in _EXCLUDED_DIRS or path.suffix in _EXCLUDED_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink in package identity: {path}")
            if path.is_dir():
                visit(path)
            elif path.is_file():
                files[path.relative_to(root).as_posix()] = _digest_file(path, normalize_python=True)
            else:
                raise ValueError(f"Unsupported package entry: {path}")

    visit(root)
    if not files or not any(name.endswith(".py") for name in files):
        raise ValueError(f"Package source tree is empty: {root}")
    return files


def _package_root(name: str) -> Path:
    module = importlib.import_module(name)
    filename = getattr(module, "__file__", None)
    if not filename or Path(filename).name != "__init__.py":
        raise ValueError(f"{name} must have an inspectable package source directory")
    return Path(filename).parent


def _config(value: Any, location: str, resources: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Non-string processor config key at {location}")
            name = f"{location}.{key}"
            if key in _TUNING:
                continue
            if key in {"path", "file"} or key.endswith(("_path", "_file")):
                # Disabled optional components (e.g. character language models)
                # have explicit null paths; null does not stand for an unhashed file.
                if item is None:
                    result[key] = None
                    continue
                if not isinstance(item, (str, Path)) or not str(item):
                    raise ValueError(f"Invalid resource path at {name}")
                resources[name] = _digest_file(Path(item))
                continue
            result[key] = _config(item, name, resources)
        return result
    if isinstance(value, (list, tuple)):
        return [_config(item, f"{location}[{index}]", resources) for index, item in enumerate(value)]
    if isinstance(value, str):
        if ntpath.isabs(value) or Path(value).is_absolute() or "/" in value or "\\" in value:
            raise ValueError(f"Unidentified path at {location}")
        return value
    if value is None or type(value) in (bool, int, float):
        return value
    raise ValueError(f"Non-JSON processor config at {location}")


def build_classla_ner_identity(
    adapter: Any, *, classla_root: Path | None = None, obeliks_root: Path | None = None,
    semora_ner_root: Path | None = None,
) -> dict[str, Any]:
    """Describe loaded pipeline semantics without retaining machine-specific paths.

    Optional roots permit testing and explicit source installations; absent roots
    are resolved from the imported packages and this installation of Semora.
    """
    metadata = adapter.metadata
    if not isinstance(metadata, dict) or metadata.get("adapter") != "classla":
        raise ValueError("A loaded CLASSLA adapter is required")
    processors = adapter.pipeline.processors
    if set(processors) != {"tokenize", "ner"}:
        raise ValueError("Expected exactly tokenize and ner processors")
    model_path = processors["ner"].config.get("model_path")
    if not isinstance(model_path, (str, Path)) or not str(model_path):
        raise ValueError("NER processor has no model_path")
    resources: dict[str, str] = {}
    configs = {name: _config(processors[name].config, name, resources) for name in sorted(processors)}
    if configs["tokenize"].get("library") != "obeliks":
        raise ValueError("Only the Obeliks tokenizer is supported for CLASSLA NER identity")
    obeliks_version = None
    if obeliks_root is None:
        try:
            obeliks_version = importlib.metadata.version("obeliks")
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(semora_ner_root) if semora_ner_root is not None else Path(__file__).parent.parent / "ner"
    identity = {
        "format": "semora.classla-ner.identity.v1",
        "classla_version": metadata.get("classla_version"),
        "obeliks_version": obeliks_version,
        "package": metadata.get("package"),
        "processor_config": configs,
        "resources": resources,
        "classla_files": _tree(Path(classla_root) if classla_root is not None else _package_root("classla")),
        "obeliks_files": _tree(Path(obeliks_root) if obeliks_root is not None else _package_root("obeliks")),
        "semora_files": {name: _digest_file(root / name, normalize_python=True) for name in ("adapters.py", "base.py")},
    }
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("CLASSLA identity must contain finite JSON values") from exc
    return identity
