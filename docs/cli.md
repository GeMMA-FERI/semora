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

After benchmarking, enable independent CLASSLA processes explicitly:

```sh
semora index lemma --workers 4 --batch-articles 25 --profile
```

`--batch-articles` is the number processed per worker. The parent fetches up to
`workers × batch-articles`, sends one batch to each worker, then commits all
results in source order through its single SQLite connection. Four workers use
approximately 35 GB of host RAM with the current Slovene models.

Two read-only commands diagnose CLASSLA without modifying the lemma index:

```sh
semora profile classla --articles 1
semora benchmark classla --articles 500 --batch-articles 50 --workers 1 2 3 4
```

The profiler writes a PyTorch CPU/CUDA Chrome trace and operator summary below
`indexes/profiles/`. The benchmark uses the same deterministic article sample
for every worker count and writes `indexes/classla_benchmark.json`. It reports
steady-state throughput, worker initialization and memory, and sampled NVIDIA
GPU utilization, memory, and power.

Operator profiling is limited to five articles because CLASSLA's recurrent POS
model generates very large event traces. For larger samples, use the lightweight
`semora index lemma --profile` stage timings instead.

Semora also installs lower-level commands:

```sh
semora-project-embeddings --help
semora-query --help
semora-serve --help
```

Each command requires the optional dependency group associated with its domain.
