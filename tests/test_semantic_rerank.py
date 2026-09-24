import sqlite3
from unittest.mock import Mock

import numpy as np
import pytest

from semora.retrieval.engine import SearchEngine
from semora.retrieval.semantic_rerank import rerank_candidates


@pytest.fixture
def vectors(tmp_path):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE semantic_shards(shard_index,file_name,vector_count,dimensions,dtype,byte_size)")
    db.execute("CREATE TABLE semantic_chunks(vector_id,chunk_id,shard_index,shard_offset)")
    (tmp_path / "vectors").mkdir()
    for shard, values in enumerate(([[1, 0], [0, 1]], [[1, 1], [0, 0]])):
        data = np.array(values, dtype=np.float16)
        data.tofile(tmp_path / "vectors" / f"{shard}.bin")
        db.execute(
            "INSERT INTO semantic_shards VALUES (?,?,?,?,?,?)", (shard, f"{shard}.bin", 2, 2, "float16", data.nbytes)
        )
        for offset in range(2):
            index = shard * 2 + offset
            db.execute("INSERT INTO semantic_chunks VALUES (?,?,?,?)", (index, str(index), shard, offset))
    yield db, tmp_path
    db.close()


def test_selected_vectors_only_and_deterministic_cosine(vectors):
    db, root = vectors
    results = rerank_candidates(db, root, np.array([2, 0]), [2, 1, 0, 0])
    assert [index for index, _ in results] == [0, 2, 1]
    assert [score for _, score in results] == pytest.approx([1, 1 / np.sqrt(2), 0])
    # The unselected zero vector must never be validated/read for scoring.
    assert rerank_candidates(db, root, np.array([1, 0]), []) == []


@pytest.mark.parametrize("fault", ["missing", "truncated", "offset", "dimensions", "mapping", "zero", "path"])
def test_bad_shards_fail_instead_of_silent_approximate_fallback(vectors, fault):
    db, root = vectors
    ids = [0]
    if fault == "missing":
        (root / "vectors/0.bin").unlink()
    elif fault == "truncated":
        (root / "vectors/0.bin").write_bytes(b"")
    elif fault == "offset":
        db.execute("UPDATE semantic_chunks SET shard_offset=999 WHERE vector_id=0")
    elif fault == "dimensions":
        db.execute("UPDATE semantic_shards SET dimensions=3")
    elif fault == "mapping":
        ids = [99]
    elif fault == "zero":
        ids = [3]
    else:
        db.execute("UPDATE semantic_shards SET file_name='../outside'")
    with pytest.raises((ValueError, OSError)):
        rerank_candidates(db, root, np.array([1, 0]), ids)


def test_filter_then_rerank_then_article_deduplication(vectors):
    db, root = vectors
    engine = object.__new__(SearchEngine)
    engine.semantic_rerank = True
    engine.semantic_candidates = 500
    engine.semantic_dir = root
    engine._semantic_mapping = db
    engine._semantic_index = Mock(ntotal=3)
    engine._semantic_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[1, 2, 0]]))
    engine._semantic_chunk_id_map = lambda ids: {i: str(i) for i in ids}
    engine._chunk_rows = lambda ids: {
        str(i): {"article_id": "A" if i in (0, 1) else "B", "date": None} for i in range(3)
    }
    results = engine._rank_semantic_candidates(np.array([[1, 0]]), 20, 0, None, None, None)
    assert [row["article_id"] for row, _ in results] == ["A", "B"]
    assert results[0][1] == pytest.approx(1)
    assert engine._semantic_index.search.call_args.args[1] == 3
    engine.semantic_rerank = False
    results = engine._rank_semantic_candidates(np.array([[1, 0]]), 20, 0, None, None, None)
    assert results[0][1] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "options",
    [{"semantic_candidates": 19}, {"semantic_candidates": 10001}, {"semantic_nprobe": 0}, {"semantic_nprobe": 65537}],
)
def test_invalid_settings_rejected_before_opening_database(options):
    with pytest.raises(ValueError):
        SearchEngine("does-not-exist", **options)
