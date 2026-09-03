# Installation

Install the lightweight storage core with:

```sh
pip install semora
```

Install only the feature groups required by your application:

```sh
pip install "semora[chunking]"
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
