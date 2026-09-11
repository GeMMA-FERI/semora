from __future__ import annotations

from pathlib import Path

import pytest

from semora.retrieval.semantic_lock import SemanticBuildLock


def test_semantic_build_lock_rejects_a_second_writer_and_can_be_reused(tmp_path: Path) -> None:
    path = tmp_path / ".vectors.lock"
    with SemanticBuildLock(path, "vector"):
        with pytest.raises(RuntimeError, match="Another Semora vector build"):
            with SemanticBuildLock(path, "vector"):
                pass

    with SemanticBuildLock(path, "vector"):
        assert path.is_file()
