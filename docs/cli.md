# Command-line tools

The unified `semora` command builds and searches a repository-local corpus:

```sh
semora --help
semora ingest --help
semora index --help
semora search --help
semora stdio --help
```

It expects `corpus/` and `indexes/` below the current directory unless the
global `--root` option is specified. See [Corpus indexing and
search](./corpus-search.md) for the complete workflow.

Ingestion can run as one command or as three explicit stages:

```sh
semora ingest --newspapers --replace
semora ingest --articles
semora ingest --chunks
```

BM25 indexing supports bounded experiments and continuation:

```sh
semora index bm25 --max-chunks 100000
semora index bm25 --max-chunks 1000000
semora index bm25
```

The limit is the desired total index size, not the number added by that one
command. Use `--rebuild` when a fresh lexical index is required.

Slovene lemma search is an optional second lexical index:

```sh
semora models download-classla
semora index lemma --max-articles 10000 --batch-articles 50 --profile
semora index lemma
semora search bm25-lemma "gledališča"
semora search bm25-combined "ljubljanska gledališča" --lemma-weight 1.0
```

The article limit is a total target, so lemma indexing resumes after its last
committed article. Each article batch is processed in one CLASSLA call and one
SQLite transaction. `--classla-pos-batch-size` and
`--classla-lemma-batch-size` override CLASSLA's internal inference batches;
`--profile` prints processor timings and CUDA-memory usage. Run
`semora index lemma --rebuild` after rebuilding or expanding the surface BM25
index.

Semora also installs lower-level commands:

```sh
semora-project-embeddings --help
semora-query --help
semora-serve --help
```

Each command requires the optional dependency group associated with its domain.
