# Corpus indexing and search

Semora treats each Markdown file as a newspaper issue and every level-one
heading (`# Article title`) as the start of an article. A source file may have
a JSON metadata sidecar with the same stem and `_metadata.json` suffix:

```text
corpus/Jutro_Ljubljana/URN_NBN_SI_doc-0L8XYEOC.md
corpus/Jutro_Ljubljana/URN_NBN_SI_doc-0L8XYEOC_metadata.json
```

When present, the JSON object is preserved in `metadata_json`; known fields
such as date, publisher, source, rights, title, URN, issue, and volume are also
normalized into database columns. A missing sidecar does not prevent ingestion:
Semora infers the newspaper name from its parent folder by replacing
underscores with spaces. All other metadata fields remain optional.

## Build from scratch

Run these commands from the repository containing `corpus/`:

```sh
semora ingest --newspapers --replace
semora ingest --articles
semora ingest --chunks
semora index bm25 --max-chunks 100000
semora index bm25
semora models download-classla
semora index lemma
semora index semantic
```

The three ingestion stages operate on `indexes/semora.sqlite`. The newspaper
stage creates the database and stores source issues in batches. The article
stage reads those stored issues and extracts all articles. The chunk stage then
uses `google/embeddinggemma-300m` to produce 256-token chunks with a 64-token
overlap. Each stage has its own progress bar and can be run as a separate
process. Stage flags may also be combined, and omitting all stage flags runs all
three in order. `--replace` is only valid when `--newspapers` is selected and
removes the previous database before rebuilding it.

`index bm25` builds a contentless SQLite FTS5 index, so indexed title and chunk
text are not stored for a second time. A small mapping table connects FTS row
IDs to canonical chunk IDs. Writes commit in batches and resume after the last
committed chunk. `--max-chunks N` is a total target: rerunning the same target
is a no-op, increasing it adds more chunks, and omitting it indexes everything
remaining. Use `--batch-size` to control transaction size or `--rebuild` to
clear the lexical index first. A target below the current indexed count does
not remove rows.

`index lemma` uses CLASSLA's Slovene tokenizer, POS tagger, and lemmatizer. It
combines each `--batch-articles` group into one CLASSLA call, separates source
documents with an explicit EOD sentinel, and validates token positions while
mapping the lemmas back onto the original articles and chunks. The same group
is then committed in one SQLite transaction. `--max-articles N` is a resumable
total target. If the surface BM25 sample or CLASSLA pipeline type changes,
rebuild the lemma index with `--rebuild`.

Install the `classla` extra and download its language resources before the
first lemma-indexing run:

```sh
pip install "semora[classla]"
semora models download-classla
```

Use `--classla-device cpu` to force CPU execution or `cuda` to require a CUDA
device. `--classla-resources-dir` selects a non-default resource directory.
The default CLASSLA pipeline targets standard Slovene; historical spelling and
OCR errors will not always normalize correctly, so surface BM25 remains useful.

CLASSLA's model initialization can take roughly two minutes on a large model
installation, but happens only once per indexing or persistent search process.
Use profiling while tuning inference batches:

```sh
semora index lemma --batch-articles 50 --profile
semora index lemma --batch-articles 50 --classla-pos-batch-size 10000 \
  --classla-lemma-batch-size 200 --profile
```

The profile reports database fetch/write time, tokenizer/POS/lemma time,
tokens per second, and peak CUDA memory. Larger processor batches can improve
GPU utilization but require more GPU memory. Omitting the overrides retains
CLASSLA's model defaults.

`index semantic` encodes valid chunks and creates a normalized inner-product FAISS index in
`indexes/semantic/`; its manifest records the model and chunk configuration.

The default model is gated. Accept the
[EmbeddingGemma model terms](https://huggingface.co/google/embeddinggemma-300m)
and authenticate with Hugging Face before using it. Both ingestion and semantic
indexing accept model and batching overrides; use `--help` for details.

## Search

```sh
semora search bm25 "Ljubljana gledališče" --limit 20
semora search bm25-lemma "ljubljanska gledališča" --limit 20
semora search bm25-combined "ljubljanska gledališča" --lemma-weight 1.0
semora search regex "Ljubljan[ae]" --ignore-case --context-lines 4
semora search semantic "reports about theatre in Ljubljana" --before 1 --after 1
```

All modes return the same JSON shape. A hit contains the normalized newspaper
name and date, source document identifier, relative Markdown path, score,
original source lines, article title, and snippet. `--before` and `--after`
include adjacent chunks from the same article. `--context-lines` expands the
snippet using lines from the original newspaper issue. Searches may be filtered
with `--newspaper`, `--date-from`, and `--date-to`.

`bm25-lemma` lemmatizes the query and searches only normalized terms.
`bm25-combined` merges surface and lemma BM25 scores; `--lemma-weight` controls
the lemma contribution. Lemma queries are interpreted as natural text, whereas
surface `bm25` queries retain SQLite FTS5 query syntax.

## Persistent agent process

`semora stdio` loads the database, FAISS index, and embedding model once, then
accepts one JSON request per input line and writes one JSON response per line.
Diagnostics go to stderr, leaving stdout machine-readable.

```json
{"id":"q1","op":"search","mode":"semantic","query":"reports about theatre","limit":5,"before":1,"after":1}
{"id":"q2","op":"search","mode":"bm25","query":"Ljubljana","date_from":"1934-01-01"}
{"id":"q3","op":"search","mode":"bm25-combined","query":"ljubljanska gledališča","lemma_weight":1.0}
{"id":"health","op":"health"}
{"id":"done","op":"shutdown"}
```

Keep the process open and communicate through its stdin and stdout using
`subprocess.Popen` or an equivalent long-lived process API. A one-shot shell
pipe closes stdin after its input is exhausted and therefore cannot be reused.
`semora stdio --no-semantic` skips startup preloading; the first semantic query
still loads the model and index lazily.

Source character offsets are zero-based and end-exclusive. Source line numbers
are one-based and inclusive, and always refer to the original Markdown file.
