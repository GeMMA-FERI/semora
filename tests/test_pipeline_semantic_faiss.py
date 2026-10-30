"""Native FAISS coverage, checkpoint recovery and private-generation boundaries."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from semora.pipeline import semantic_faiss
from semora.pipeline.batches import run_batch
from semora.pipeline.manifest import BatchIssue, BatchManifest
from semora.pipeline.semantic import index_batch_vectors
from semora.pipeline.semantic_faiss import build_semantic_generation, validate_faiss_ids
from semora.retrieval.semantic_faiss import FaissBuildConfig

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class Model:
    def __init__(self):
        self.calls = 0

    def encode_document(self, texts, **_kwargs):
        self.calls += 1
        return np.array([[len(text), 1.0, 2.0] for text in texts], dtype="float32")


@pytest.fixture(autouse=True)
def one_faiss_thread():
    original = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    yield
    faiss.omp_set_num_threads(original)


def ingest(root: Path, batch_id: str, *names: str) -> Path:
    inputs = root / "incoming"
    entries = []
    for name in names:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Title {Path(name).stem}\nA train travels to Ljubljana each morning.\n", encoding="utf-8")
        metadata = path.with_name(path.stem + "_metadata.json")
        metadata.write_text(json.dumps({"Record": {"source": "Straža", "date": "1925 02 18"}}), encoding="utf-8")
        entries.append(
            BatchIssue(
                relative_path=name,
                markdown_path=name,
                metadata_path=metadata.relative_to(inputs).as_posix(),
                markdown_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            )
        )
    database = root / "indexes" / "semora.sqlite"
    run_batch(
        BatchManifest(batch_id, tuple(entries)), inputs, database, tokenizer=Tokenizer(), token_count=3, token_overlap=1
    )
    return database


def vectors(root: Path, batch: str = "batch-a", names=("Straza/a.md",), model=None) -> tuple[Path, Path]:
    database = ingest(root, batch, *names)
    directory = root / "indexes" / "semantic"
    index_batch_vectors(database, batch, directory, dimensions=2, batch_size=2, shard_size=2, model=model or Model())
    return database, directory


def build(database: Path, directory: Path, **kwargs):
    return build_semantic_generation(
        database,
        directory,
        config=FaissBuildConfig(index_type="flat"),
        checkpoint_shards=1,
        faiss_module=faiss,
        **kwargs,
    )


def read_index(result: dict):
    return faiss.read_index(str(Path(result["generation_dir"]) / "index.faiss"))


def ids(index) -> list[int]:
    return sorted(faiss.vector_to_array(index.id_map).tolist())


def test_two_batches_reuse_private_generation_without_reembedding_or_retraining(tmp_path: Path, monkeypatch) -> None:
    model = Model()
    database, directory = vectors(tmp_path, model=model)
    (directory / "manifest.json").write_bytes(b"serving manifest sentinel")
    (directory / "index.faiss").write_bytes(b"serving index sentinel")
    first = build(database, directory)
    original_index = Path(first["generation_dir"]) / "index.faiss"
    original_bytes = original_index.read_bytes()
    before_calls = model.calls
    before_mapping = []
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        before_mapping = connection.execute(
            "SELECT vector_id,chunk_id FROM semantic_chunks ORDER BY vector_id"
        ).fetchall()
    vectors(tmp_path, "batch-b", ("Straza/b.md", "Straza/c.md"), model=model)
    with closing(sqlite3.connect(directory / "mapping.sqlite")) as connection:
        after_mapping = connection.execute(
            "SELECT vector_id,chunk_id FROM semantic_chunks ORDER BY vector_id"
        ).fetchall()
    new_mapping = after_mapping[len(before_mapping) :]
    assert min(row[1] for row in new_mapping) < max(row[1] for row in before_mapping)
    calls_after_vectors = model.calls
    assert calls_after_vectors > before_calls

    def reject_new_index(*_args, **_kwargs):
        pytest.fail("An unchanged trained/flat predecessor must be extended, not replaced")

    monkeypatch.setattr(semantic_faiss, "_new_index", reject_new_index)
    second = build(database, directory)
    retry = build(database, directory)

    assert second["generation"] != first["generation"]
    assert second["added_vectors"] == len(new_mapping) and retry["added_vectors"] == 0
    assert ids(read_index(second)) == list(range(len(after_mapping)))
    assert before_mapping == after_mapping[: len(before_mapping)]
    assert model.calls == calls_after_vectors
    assert original_index.read_bytes() == original_bytes
    assert (directory / "manifest.json").read_bytes() == b"serving manifest sentinel"
    assert (directory / "index.faiss").read_bytes() == b"serving index sentinel"
    assert not first["retrieval_ready"] and not second["retrieval_ready"]


@pytest.mark.parametrize("after_pointer", [False, True])
def test_checkpoint_pointer_fault_resumes_exact_ids_once(tmp_path: Path, monkeypatch, after_pointer: bool) -> None:
    database, directory = vectors(tmp_path, names=("Straza/a.md", "Straza/b.md", "Straza/c.md"))
    write = semantic_faiss._write_json
    faulted = False

    def interrupted(path, value, **kwargs):
        nonlocal faulted
        if path.name == "checkpoint.json" and value["vector_count"] >= 4 and not faulted:
            faulted = True
            if after_pointer:
                write(path, value, **kwargs)
            raise RuntimeError("checkpoint pointer interrupted")
        return write(path, value, **kwargs)

    monkeypatch.setattr(semantic_faiss, "_write_json", interrupted)
    with pytest.raises(RuntimeError, match="pointer interrupted"):
        build(database, directory)
    pointer = next((directory / "generations").glob(".build-*/checkpoint.json"))
    checkpoint = json.loads(pointer.read_text(encoding="utf-8"))
    saved = faiss.read_index(str(pointer.parent / checkpoint["index_file"]))
    assert ids(saved) == list(range(checkpoint["vector_count"]))
    monkeypatch.setattr(semantic_faiss, "_write_json", write)
    monkeypatch.setattr(semantic_faiss, "_new_index", lambda *_a, **_k: pytest.fail("Checkpoint must resume"))

    result = build(database, directory)

    assert result["added_vectors"] == result["chunks"] - checkpoint["vector_count"]
    assert ids(read_index(result)) == list(range(result["chunks"]))
    assert len(list((directory / "generations").glob("[0-9a-f]*/manifest.json"))) == 1


def test_completed_private_generation_recovers_after_final_rename_failure(tmp_path: Path, monkeypatch) -> None:
    database, directory = vectors(tmp_path)
    rename = semantic_faiss.os.rename
    faulted = False

    def interrupted(source, target):
        nonlocal faulted
        if Path(source).name.startswith(".build-") and not faulted:
            faulted = True
            raise OSError("generation rename interrupted")
        return rename(source, target)

    monkeypatch.setattr(semantic_faiss.os, "rename", interrupted)
    with pytest.raises(OSError, match="generation rename interrupted"):
        build(database, directory)
    private = next((directory / "generations").glob(".build-*"))
    assert (private / "manifest.json").is_file() and (private / "index.faiss").is_file()

    monkeypatch.setattr(semantic_faiss.os, "rename", rename)
    monkeypatch.setattr(
        semantic_faiss, "_new_index", lambda *_a, **_k: pytest.fail("Must reuse complete private index")
    )
    result = build(database, directory)

    assert result["added_vectors"] == 0
    assert ids(read_index(result)) == list(range(result["chunks"]))
    assert not private.exists()


def test_vector_shard_checksum_tampering_stops_before_generation(tmp_path: Path) -> None:
    database, directory = vectors(tmp_path)
    shard = next((directory / "vectors").glob("*.f16"))
    shard.write_bytes(bytes(shard.stat().st_size))

    with pytest.raises(ValueError, match="checksum mismatch"):
        build(database, directory)

    assert not (directory / "generations").exists()


def test_completed_generation_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    database, directory = vectors(tmp_path)
    result = build(database, directory)
    target = Path(result["generation_dir"]) / "index.faiss"
    target.write_bytes(b"corrupt generation bytes")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build(database, directory)


@pytest.mark.parametrize("vector_ids", [[0, 0, 2], [0, 1, 7], [-1, 1, 2]])
def test_equal_count_forged_vector_ids_are_rejected(vector_ids) -> None:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
    index.add_with_ids(np.ones((3, 2), dtype="float32"), np.asarray(vector_ids, dtype="int64"))

    with pytest.raises(ValueError, match="duplicate or unknown"):
        validate_faiss_ids(faiss, index, 3, 2)


@pytest.mark.parametrize("metric,dimensions", [("l2", 2), ("ip", 3)])
def test_wrong_metric_or_dimension_is_rejected(metric: str, dimensions: int) -> None:
    flat = faiss.IndexFlatL2(dimensions) if metric == "l2" else faiss.IndexFlatIP(dimensions)
    index = faiss.IndexIDMap2(flat)
    index.add_with_ids(np.ones((2, dimensions), dtype="float32"), np.arange(2, dtype="int64"))

    with pytest.raises(ValueError, match="inner product|shape/training"):
        validate_faiss_ids(faiss, index, 2, 2)


def test_forged_completed_generation_ids_fail_even_with_updated_checksum(tmp_path: Path) -> None:
    database, directory = vectors(tmp_path)
    result = build(database, directory)
    generation = Path(result["generation_dir"])
    target = generation / "index.faiss"
    index = read_index(result)
    forged = np.arange(result["chunks"], dtype="int64")
    forged[-1] = 999
    faiss.copy_array_to_vector(forged, index.id_map)
    faiss.write_index(index, str(target))
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    manifest["index_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate or unknown"):
        build(database, directory)


def test_source_corpus_identity_mismatch_does_not_create_generation(tmp_path: Path) -> None:
    first, directory = vectors(tmp_path / "first")
    other = ingest(tmp_path / "unrelated", "batch-a", "Straza/a.md")

    with pytest.raises(ValueError, match="different logical corpus"):
        build(other, directory)

    assert first != other and not (directory / "generations").exists()


def test_checkpoint_cannot_reference_a_file_outside_generation(tmp_path: Path, monkeypatch) -> None:
    database, directory = vectors(tmp_path)
    checkpoint = semantic_faiss._checkpoint

    def interrupted(*args, **kwargs):
        checkpoint(*args, **kwargs)
        raise RuntimeError("stop after checkpoint")

    monkeypatch.setattr(semantic_faiss, "_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        build(database, directory)
    pointer = next((directory / "generations").glob(".build-*/checkpoint.json"))
    value = json.loads(pointer.read_text(encoding="utf-8"))
    value["index_file"] = "../outside.faiss"
    pointer.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(semantic_faiss, "_checkpoint", checkpoint)

    with pytest.raises(ValueError, match="artifact filename|escapes"):
        build(database, directory)

    assert not (directory / "generations" / "outside.faiss").exists()


def test_old_prefix_at_new_generation_path_cannot_claim_current_coverage(tmp_path: Path, monkeypatch) -> None:
    database, directory = vectors(tmp_path)
    first = build(database, directory)
    vectors(tmp_path, "batch-b", ("Straza/b.md",))
    checkpoint = semantic_faiss._checkpoint

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("capture new fingerprint")

    monkeypatch.setattr(semantic_faiss, "_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="capture new fingerprint"):
        build(database, directory)
    pending = next((directory / "generations").glob(".build-*"))
    final = pending.with_name(pending.name.removeprefix(".build-"))
    shutil.copytree(Path(first["generation_dir"]), final)
    monkeypatch.setattr(semantic_faiss, "_checkpoint", checkpoint)

    with pytest.raises(ValueError, match="generation|coverage|fingerprint|metadata"):
        build(database, directory)
