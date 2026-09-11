from __future__ import annotations

import json
from pathlib import Path

from semora.retrieval.semantic_diagnostics import SemanticVectorJournal


def test_semantic_vector_journal_flushes_events_and_failures(tmp_path: Path) -> None:
    try:
        with SemanticVectorJournal(tmp_path) as journal:
            journal.write("model_loaded", indexed_chunks=10)
            raise RuntimeError("failure detail")
    except RuntimeError:
        pass

    records = [json.loads(line) for line in (tmp_path / "vector-build.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["model_loaded", "build_failed"]
    assert records[-1]["error_type"] == "RuntimeError"
    assert records[-1]["error"] == "failure detail"
    assert records[-1]["pid"] > 0
