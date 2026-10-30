# Semora

Semora is a Python toolkit for turning newspaper archives into searchable
collections. It takes newspaper text in Markdown, separates issues into articles,
and lets you search by words, meaning, or the names mentioned in the text.

## What you can do

- **Build a corpus.** Import newspaper issues with their metadata and keep
  articles and text chunks in SQLite.
- **Search the archive.** Find articles with keyword, regular-expression, or
  semantic search. For Slovene text, lemma search also matches inflected words.
- **Explore names.** Extract named entities and look up the articles where
  those names appear.
- **Work with embeddings.** Split text into token windows, generate embeddings,
  and reduce their dimensions for analysis.
- **Add new material.** Import reviewed batches while reusing existing text
  processing and indexes.

Use the `semora` command for the newspaper workflow, or import the Python modules
when you only need part of it. Model dependencies are optional, so you can install
the features you need.

## Getting started

You'll need Python 3.10 or later. From a source checkout, install Semora with:

```sh
python -m pip install .
```

### Search your first articles

Start with keyword search. It uses SQLite's FTS5 support and does not need an
embedding model or a GPU.

Place your source files in a `corpus/` folder. Each Markdown file represents a
newspaper issue, with level-one headings (`# Article title`) marking articles.
A matching metadata file is optional:

```text
corpus/
└── Jutro_Ljubljana/
    ├── URN_NBN_SI_doc-0L8XYEOC.md
    └── URN_NBN_SI_doc-0L8XYEOC_metadata.json
```

From the directory containing `corpus/`, import the issues and articles, then
build the keyword index:

```sh
semora ingest --newspapers
semora ingest --articles
semora index bm25
semora search bm25 "gledališče"
```

Semora stores the database and indexes under `indexes/`. To work in another
directory, put `--root PATH` before the subcommand, for example
`semora --root /path/to/archive search bm25 "gledališče"`.

This is the starting workflow for a new corpus. For recurring imports into an
existing archive, follow the [additive indexing guide](docs/additive-indexing.md),
which covers batch planning, resume behavior, and publishing updated indexes.

### Add semantic or Slovene lemma search

Install the relevant extras from the source checkout:

```sh
python -m pip install ".[chunking,retrieval]"
python -m pip install ".[classla]"
```

Semantic search needs text chunks and an embedding index. Slovene lemma search
needs CLASSLA's language resources and a lemma index. The
[corpus search guide](docs/corpus-search.md) walks through both, including model
access, CPU/GPU options, and trying a small sample before indexing a full archive.

## Use Semora in Python

The same tools are available without the newspaper command-line workflow.
For example, you can turn text into overlapping token windows and embed them:

```sh
python -m pip install ".[chunking,embeddings]"
```

```python
from semora.embeddings import get_embedder
from semora.text import TokenWindowProcessor

chunks = TokenWindowProcessor(
    model_id="intfloat/multilingual-e5-large",
    token_count=256,
    token_overlap=64,
).process("article-1", "Your article text goes here.")

embedder = get_embedder("intfloat/multilingual-e5-large").load()
vectors = embedder.embed_documents([text for _, text in chunks])
```

See the guides for [text processing](docs/text.md),
[embeddings](docs/embeddings.md), and [SQLite storage](docs/storage.md) for more
examples.

## Further reading

| If you want to…                                               | Read                                                |
| ------------------------------------------------------------- | --------------------------------------------------- |
| Choose dependencies and set up models                         | [Installation](docs/installation.md)                |
| Build indexes, filter results, or connect an agent over stdio | [Corpus indexing and search](docs/corpus-search.md) |
| Add batches to an existing archive                            | [Additive indexing](docs/additive-indexing.md)      |
| Extract named entities                                        | [NER indexing](docs/ner-indexing.md)                |
| Find names and their articles                                 | [Entity search](docs/entity-search.md)              |
| Compare entity extraction models                              | [NER benchmarks](docs/ner-benchmark.md)             |
| Use projection, query, and server commands                    | [Command-line tools](docs/cli.md)                   |

## Development

Create and activate a virtual environment, then install the development tools
and run the tests:

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -e ".[dev]"
python -m pytest
```

Tests for optional features need their corresponding extras installed.
