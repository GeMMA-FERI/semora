"""Corpus ingestion with exact source spans."""

from semora.corpus.indexer import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKEN_COUNT,
    DEFAULT_TOKEN_OVERLAP,
    CorpusStats,
    ingest_corpus,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_TOKEN_COUNT",
    "DEFAULT_TOKEN_OVERLAP",
    "CorpusStats",
    "ingest_corpus",
]
