# Semora

Semora provides reusable building blocks for newspaper-corpus ingestion,
document storage, text chunking, embedding generation, dimensionality
reduction, and lexical or dense retrieval. Its core SQLite storage package has
no third-party runtime dependencies.

## Installation

Install the base package from a source checkout:

```sh
pip install .
```

Install optional capabilities as needed:

```sh
pip install ".[chunking,embeddings,openai-batch,projection]"
pip install ".[classla,retrieval,server]"
```

For development:

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

## SQLite storage and migrations

The database schema and numbered SQL migrations are package resources. Calling
`Database.initialize()` creates a new database or applies pending migrations:

```python
from semora import Database

database = Database("data/semantic-search.sqlite")
try:
    database.initialize()
finally:
    database.close()
```

Domain repositories are available as `database.documents`, `database.chunks`,
`database.embeddings`, `database.projections`, and `database.runs`. The direct
database methods remain available for advanced queries.

## Text and embedding APIs

```python
from semora.text import TokenWindowProcessor, remove_markdown_images
from semora.embeddings import get_embedder

clean_text = remove_markdown_images(markdown)
chunks = TokenWindowProcessor(
    model_id="intfloat/multilingual-e5-large",
    token_count=256,
    token_overlap=64,
).process("document-id", clean_text)

embedder = get_embedder("intfloat/multilingual-e5-large").load()
vectors = embedder.embed_documents([text for _, text in chunks])
```

## Newspaper corpus search

Place source files below `corpus/`. A newspaper issue is a Markdown file and
may have a matching `_metadata.json` file:

```text
corpus/
└── Jutro_Ljubljana/
    ├── URN_NBN_SI_doc-0L8XYEOC.md
    └── URN_NBN_SI_doc-0L8XYEOC_metadata.json
```

Level-one Markdown headings delimit articles. The following commands build the
SQLite corpus, BM25 index, and EmbeddingGemma FAISS index under `indexes/`:

```sh
semora ingest --newspapers --replace
semora ingest --articles
semora ingest --chunks
semora index bm25 --max-articles 100000
semora index bm25
semora models download-classla
semora index lemma
semora index semantic
semora search bm25 "search terms"
semora search bm25-combined "iskanje po pregibnih oblikah"
semora search regex "regular expression" --context-lines 3
semora search semantic "natural-language query" --before 1 --after 1
```

Search results use relevance-centered snippets limited to 600 characters by
default; use `--max-snippet-chars` to choose another limit.

By default, paths are relative to the current directory. Use `--root PATH`
before the subcommand to select a different repository root. See the
[corpus-search guide](docs/corpus-search.md) for filters, result fields, and the
persistent NDJSON interface intended for agent integrations.

Running `semora ingest --replace` without stage flags performs all three
ingestion stages in the same order.

BM25 indexing is contentless, article-oriented, and resumable. `--max-articles` specifies the total
desired index size, so increasing it continues from the last committed batch;
omitting it indexes all remaining valid articles. Chunks are used only by the
semantic index and semantic-search context expansion.

The optional CLASSLA index lemmatizes Slovene articles in multi-document
batches and stores a second contentless FTS index. It supports lemma-only and
combined surface-plus-lemma BM25 search without changing the original text or
snippets. Use `semora index lemma --profile` to inspect processor throughput
and CUDA-memory usage. With one CLASSLA worker, tokenization runs in a
lightweight persistent subprocess while the parent retains the only copy of
the POS model, lemma model, CUDA context, and lexicon. Set both
`--pipeline-depth 1` and `--tokenizer-workers 0` to disable this staged path.

Lemma indexing also pipelines SQLite work with CLASSLA inference. Semora fetches
all chunk mappings for an article group in one query, prefetches the next group
while CLASSLA is running, and commits completed groups through one ordered writer.
Each FTS insert and resume checkpoint remains in the same transaction.

For read-only performance diagnosis, use `semora profile classla` to create a
PyTorch CPU/CUDA trace and `semora benchmark classla --workers 1 2 3 4` to
measure whether multiple CLASSLA processes improve steady-state throughput.
When the benchmark supports it, `semora index lemma --workers N` enables the
same architecture with a single ordered SQLite writer.

To benchmark the shared-memory staged path separately, run:

```sh
semora benchmark classla --workers 1 --pipeline-depth 3 --tokenizer-workers 1 \
  --classla-lemma-batch-size 200
```

## Command-line tools

The package also keeps focused commands for lower-level workflows:

```sh
semora-project-embeddings --help
semora-query --help
semora-serve --help
```

The underlying modules can also be executed with `python -m`, for example:

```sh
python -m semora.projection.projector --help
```

Additional usage notes are available in [`docs/`](docs/).
