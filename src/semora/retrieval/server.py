r"""Simple FastAPI server that exposes a /query endpoint for embeddings search.

POST /query JSON body:
    {"query": "your question", "topk": 10}

Response:
    {"results": [{"rank":1, "path": "...", "score": 0.9123}, ... ]}
"""

from __future__ import annotations

import argparse
import os
import re

import faiss
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from semora.embeddings.io import load_embeddings
from semora.embeddings.registry import get_embedder
from semora.text.ids import extract_chunk_path

app = FastAPI(title="Embedding Query Server")

# Globals populated at startup
_INDEX = None
_FILE_PATHS: list[str] = []
_EMBEDDER = None
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QueryRequest(BaseModel):
    query: str
    topk: int | None = 10


class QueryResult(BaseModel):
    rank: int
    path: str
    score: float


class QueryResponse(BaseModel):
    results: list[QueryResult]


class SnippetRequest(BaseModel):
    path: str | None = None
    content: str | None = None
    sentences: int | None = 3


class SnippetResponse(BaseModel):
    snippet: str


def _strip_markdown(md: str) -> str:
    # Remove code fences
    text = re.sub(r"```.*?```", "", md, flags=re.S)
    # Remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    # Remove images
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Replace links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Remove headings
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize newlines and whitespace but preserve paragraph breaks
    # Normalize CRLF -> LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces and tabs but keep newline boundaries
    text = re.sub(r"[ \t]+", " ", text)
    # Limit excessive blank lines to at most two (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim leading/trailing whitespace/newlines
    text = text.strip()
    return text


def _first_n_sentences(text: str, n: int) -> str:
    # Simple sentence splitter: split on sentence-ending punctuation followed by space
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    selected = parts[: max(1, n)]
    return " ".join(s.strip() for s in selected).strip()


def get_query_vector(text: str, embed_func):
    out = embed_func(text)
    if isinstance(out, tuple):
        out = out[0]
    emb = out.detach()
    emb = emb.unsqueeze(0)
    return emb.float().cpu().numpy().astype("float32")


def build_index(emb_matrix: torch.Tensor):
    mat = emb_matrix.numpy()
    d = mat.shape[1]
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    return index


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    global _INDEX, _FILE_PATHS, _EMBEDDER
    if _INDEX is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # embed query
    def _embed_fn(txt):
        return _EMBEDDER.embed_query(txt).to(_DEVICE).float()

    q_emb = get_query_vector(req.query, _embed_fn)
    faiss.normalize_L2(q_emb)
    topk = int(req.topk or 10)
    distances, indices = _INDEX.search(q_emb, topk)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0], strict=True), start=1):
        path = extract_chunk_path(_FILE_PATHS[int(idx)])
        results.append(QueryResult(rank=rank, path=path, score=float(dist)))
    return QueryResponse(results=results)


@app.get("/health")
async def health():
    return {"ready": _INDEX is not None}


@app.post("/snippet", response_model=SnippetResponse)
async def snippet(req: SnippetRequest):
    """Return the first N sentences from a supplied article path or raw content.

    POST body: { "path": "/abs/path/to/article.md", "sentences": 3 }
    OR: { "content": "raw markdown or text", "sentences": 2 }
    """
    if not req.path and not req.content:
        raise HTTPException(status_code=400, detail="Provide either 'path' or 'content'")

    text = ""
    if req.path:
        if not os.path.exists(req.path):
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        try:
            with open(req.path, encoding="utf-8") as fh:
                text = fh.read()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc
    else:
        text = req.content or ""

    cleaned = _strip_markdown(text)
    n = int(req.sentences or 3)
    snippet = _first_n_sentences(cleaned, n)
    return SnippetResponse(snippet=snippet)


def start_server(args):
    global _INDEX, _FILE_PATHS, _EMBEDDER, _DEVICE

    file_paths, emb_matrix = load_embeddings(args.embeddings)
    _FILE_PATHS = file_paths
    print(f"Loaded {len(file_paths)} embeddings, shape={emb_matrix.shape}")

    print("Building FAISS index...")
    _INDEX = build_index(emb_matrix)
    print("Index ready.")

    # load embedder
    _EMBEDDER = get_embedder(args.model_id, kind=args.model_kind)
    _EMBEDDER.load()
    print(f"Loaded embedder {args.model_id}")

    # TODO: Remove this later
    if getattr(args, "allow_all", False):
        origins = ["*"]
    elif getattr(args, "allow_origins", ""):
        origins = [o.strip() for o in args.allow_origins.split(",") if o.strip()]
    else:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]  # SvelteKit dev frontend

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print(f"Configured CORS for origins: {origins}")

    # Pass the ASGI app instance directly to uvicorn.run
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to .pt file or folder with .pt batches",
    )
    parser.add_argument("--model-id", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--model-kind", type=str, choices=["causal", "sentence"], default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allow-origins",
        type=str,
        default="",
        help="Comma-separated list of allowed CORS origins (dev: http://localhost:5173)",
    )
    parser.add_argument("--allow-all", action="store_true", help="Allow all origins (CORS wildcard)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # sanity checks
    if not os.path.exists(args.embeddings):
        raise SystemExit(f"Embeddings path not found: {args.embeddings}")

    start_server(args)


if __name__ == "__main__":
    main()
