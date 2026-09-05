# Installation

Install the lightweight storage core with:

```sh
pip install semora
```

Install only the feature groups required by your application:

```sh
pip install "semora[chunking]"
pip install "semora[classla]"
pip install "semora[embeddings]"
pip install "semora[openai-batch]"
pip install "semora[projection]"
pip install "semora[retrieval]"
pip install "semora[server]"
```

For a source checkout used during development:

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

Semora supports Python 3.10 through 3.12. Optional model and retrieval groups
are intentionally absent from the base installation.

Corpus ingestion requires the `chunking` extra. Slovene lemmatization requires
the separate `classla` extra and downloaded CLASSLA language resources.
Semantic indexing and search require the `retrieval` extra. Surface BM25 and
regular-expression search use the base package, provided Python's SQLite build
includes FTS5.

The default semantic model, `google/embeddinggemma-300m`, is gated on Hugging
Face. Accept its license terms and authenticate with Hugging Face before the
first ingestion or semantic-indexing run.
