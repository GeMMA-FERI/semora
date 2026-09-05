"""Newline-delimited JSON protocol for a persistent Semora search process."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from semora.retrieval.engine import SearchEngine


def run_stdio(engine: SearchEngine, input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> None:
    for raw_line in input_stream:
        if not raw_line.strip():
            continue
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            operation = request.get("op", "search")
            if operation == "shutdown":
                _write(output_stream, {"id": request_id, "ok": True})
                return
            if operation == "health":
                _write(
                    output_stream,
                    {
                        "id": request_id,
                        "ok": True,
                        "semantic_loaded": engine.semantic_loaded,
                        "lemma_loaded": engine.lemma_loaded,
                    },
                )
                continue
            if operation != "search":
                raise ValueError(f"Unknown operation: {operation}")
            hits = engine.search(
                request["mode"],
                request["query"],
                limit=int(request.get("limit", 10)),
                before=int(request.get("before", 0)),
                after=int(request.get("after", 0)),
                context_lines=int(request.get("context_lines", 0)),
                ignore_case=bool(request.get("ignore_case", False)),
                newspaper=request.get("newspaper"),
                date_from=request.get("date_from"),
                date_to=request.get("date_to"),
                lemma_weight=float(request.get("lemma_weight", 1.0)),
            )
            _write(output_stream, {"id": request_id, "hits": [hit.as_dict() for hit in hits]})
        except Exception as exc:
            _write(
                output_stream,
                {"id": request_id, "error": {"type": type(exc).__name__, "message": str(exc)}},
            )


def _write(stream: TextIO, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()
