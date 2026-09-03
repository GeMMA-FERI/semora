from __future__ import annotations

import json
import pathlib
from typing import TypedDict

CHUNK_ID_SCHEMA = "semora.chunk_id.v1"


class ChunkId(TypedDict):
    schema: str
    path: str
    article: str
    chunk: str


def build_chunk_id(path: str, chunk: str) -> str:
    article = pathlib.Path(path).stem
    payload: ChunkId = {
        "schema": CHUNK_ID_SCHEMA,
        "path": path,
        "article": article,
        "chunk": str(chunk),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_chunk_id(chunk_id: str) -> ChunkId:
    # Structured schema (current format)
    try:
        obj = json.loads(chunk_id)
        if isinstance(obj, dict) and obj.get("schema") == CHUNK_ID_SCHEMA and "path" in obj and "chunk" in obj:
            path = str(obj["path"])
            return {
                "schema": CHUNK_ID_SCHEMA,
                "path": path,
                "article": str(obj.get("article") or pathlib.Path(path).stem),
                "chunk": str(obj["chunk"]),
            }
    except Exception:
        pass

    # Legacy fallback: <path>###<chunk>
    if "###" in chunk_id:
        path, chunk = chunk_id.rsplit("###", 1)
        return {
            "schema": CHUNK_ID_SCHEMA,
            "path": path,
            "article": pathlib.Path(path).stem,
            "chunk": str(chunk),
        }

    # Last-resort fallback for malformed ids.
    return {
        "schema": CHUNK_ID_SCHEMA,
        "path": chunk_id,
        "article": pathlib.Path(chunk_id).stem,
        "chunk": "0",
    }


def extract_chunk_path(chunk_id: str) -> str:
    return parse_chunk_id(chunk_id)["path"]
