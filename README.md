# Semora

Semora provides reusable building blocks for document storage, text chunking,
embedding generation, dimensionality reduction, and dense retrieval. Its core
SQLite storage package has no third-party runtime dependencies.

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

## Command-line tools

The package installs command-line tools for common workflows, including:

```sh
semora-embed-files --help
semora-count-embeddings --help
semora-project-embeddings --help
semora-query --help
semora-serve --help
```

The underlying modules can also be executed with `python -m`, for example:

```sh
python -m semora.projection.projector --help
```

Additional usage notes are available in [`docs/`](docs/).
