from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from semora.execution.contracts import semantic_fingerprint
from semora.execution.identity import build_classla_ner_identity


def fixture(root: Path, *, newline: bytes = b"\n"):
    classla = root / "classla"
    obeliks = root / "obeliks"
    semora = root / "semora"
    for directory in (classla, obeliks, semora):
        directory.mkdir(parents=True)
    (classla / "__init__.py").write_bytes(b"version = 1" + newline)
    (obeliks / "__init__.py").write_bytes(b"version = 1" + newline)
    (obeliks / "rules.xml").write_bytes(b"<rules/>")
    (semora / "adapters.py").write_bytes(b"adapter = 1" + newline)
    (semora / "base.py").write_bytes(b"base = 1" + newline)
    model = root / "model.pt"
    model.write_bytes(b"model")
    pretrain = root / "pretrain.pt"
    pretrain.write_bytes(b"pretrain")
    processors = {
        "tokenize": SimpleNamespace(config={"library": "obeliks", "lang": "sl"}),
        "ner": SimpleNamespace(config={"model_path": str(model), "nested": {"pretrain_file": str(pretrain)}}),
    }
    adapter = SimpleNamespace(
        metadata={"adapter": "classla", "classla_version": "1", "package": "default"},
        pipeline=SimpleNamespace(processors=processors),
    )
    return adapter, {"classla_root": classla, "obeliks_root": obeliks, "semora_ner_root": semora}


def test_relocation_and_python_line_endings_do_not_change_identity(tmp_path: Path) -> None:
    first, first_roots = fixture(tmp_path / "one")
    second, second_roots = fixture(tmp_path / "two", newline=b"\r\n")
    identity = build_classla_ner_identity(first, **first_roots)
    assert identity == build_classla_ner_identity(second, **second_roots)
    assert str(tmp_path) not in str(identity)
    assert semantic_fingerprint(identity) == semantic_fingerprint(build_classla_ner_identity(second, **second_roots))


@pytest.mark.parametrize("target", [
    "model.pt", "pretrain.pt", "classla/__init__.py", "obeliks/rules.xml", "semora/base.py",
])
def test_changed_resource_or_code_changes_identity(tmp_path: Path, target: str) -> None:
    adapter, roots = fixture(tmp_path / "source")
    previous = semantic_fingerprint(build_classla_ner_identity(adapter, **roots))
    (tmp_path / "source" / target).write_bytes(b"changed")
    assert semantic_fingerprint(build_classla_ner_identity(adapter, **roots)) != previous


def test_missing_or_unidentified_resources_fail(tmp_path: Path) -> None:
    adapter, roots = fixture(tmp_path / "source")
    (tmp_path / "source" / "model.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        build_classla_ner_identity(adapter, **roots)
    (tmp_path / "source" / "model.pt").write_bytes(b"model")
    adapter.pipeline.processors["ner"].config["unexpected"] = str(tmp_path / "unknown")
    with pytest.raises(ValueError, match="Unidentified path"):
        build_classla_ner_identity(adapter, **roots)


def test_missing_processor_and_unrecognized_tokenizer_fail(tmp_path: Path) -> None:
    adapter, roots = fixture(tmp_path / "source")
    adapter.pipeline.processors["tokenize"].config["library"] = "other"
    with pytest.raises(ValueError, match="Obeliks"):
        build_classla_ner_identity(adapter, **roots)
    del adapter.pipeline.processors["ner"]
    with pytest.raises(ValueError, match="processors"):
        build_classla_ner_identity(adapter, **roots)


def test_optional_null_resources_are_explicit_but_model_is_required(tmp_path: Path) -> None:
    adapter, roots = fixture(tmp_path / "source")
    adapter.pipeline.processors["ner"].config["forward_charlm_path"] = None
    identity = build_classla_ner_identity(adapter, **roots)
    assert identity["processor_config"]["ner"]["forward_charlm_path"] is None
    adapter.pipeline.processors["ner"].config["model_path"] = None
    with pytest.raises(ValueError, match="model_path"):
        build_classla_ner_identity(adapter, **roots)
