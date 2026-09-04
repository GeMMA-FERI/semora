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
pip install ".[retrieval,server]"
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
semora index bm25
semora index semantic
semora search bm25 "search terms"
semora search regex "regular expression" --context-lines 3
semora search semantic "natural-language query" --before 1 --after 1
```

By default, paths are relative to the current directory. Use `--root PATH`
before the subcommand to select a different repository root. See the
[corpus-search guide](docs/corpus-search.md) for filters, result fields, and the
persistent NDJSON interface intended for agent integrations.

Running `semora ingest --replace` without stage flags performs all three
ingestion stages in the same order.

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
