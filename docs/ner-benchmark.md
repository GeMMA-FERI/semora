# Offline NER comparison

The benchmark lives in `semora.diagnostics.ner`, separately from production
indexing. Source SQLite is opened read-only. All outputs are new files; existing
samples and run directories cannot be overwritten accidentally.

## Environment

Use a separate environment, especially while Nora is running. Example for Windows
and the CUDA 13.0 wheels used on the development machine (choose the appropriate
CUDA wheels for your machine):

```powershell
uv venv artifacts/ner-benchmark/.venv --python 3.14
$py = "$PWD/artifacts/ner-benchmark/.venv/Scripts/python.exe"
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $py -e . classla transformers psutil
# Alternatively, use the local CLASSLA fork instead of the published package:
uv pip install --python $py -e C:/GitHub/classla
```

No model dependencies are imported by sampling/reporting. Model runs require
PyTorch and psutil, plus the selected adapter's dependencies. Each run records
package versions, device, precision, and model configuration. Model download/load
time is separate from steady-state timing. CLASSLA defaults to its standard
Slovenian package; the adapter runs only tokenization and NER, not POS/lemma.

## 1. Create one reusable sample

```powershell
& $py -m semora.diagnostics.ner sample `
  --database C:/GitHub/nora/indexes/semora.sqlite `
  --output artifacts/ner-benchmark/sample.sqlite --articles 10000 --seed 42
```

This is a uniform sample of valid nonempty articles, selected by random rowid
rejection without scanning all article bodies or favoring articles after rowid
gaps. It preserves original content, capitalization and metadata. The sample
stores a fingerprint and summaries of decades, newspapers and length buckets.
Uniform sampling represents the corpus mix; it does not guarantee coverage of
rare decades or poor OCR. OCR quality is not automatically labelled.

## 2. Smoke-test both models

```powershell
& $py -m semora.diagnostics.ner run --sample artifacts/ner-benchmark/sample.sqlite `
  --adapter classla --download --device cuda --batch-size 32 --limit 100 `
  --output artifacts/ner-benchmark/classla-smoke
& $py -m semora.diagnostics.ner run --sample artifacts/ner-benchmark/sample.sqlite `
  --adapter transformers --model ivlcic/xlmr-ner-slavic-pline `
  --device cuda --batch-size 32 --limit 100 `
  --output artifacts/ner-benchmark/xlmr-smoke
```

`--download` explicitly downloads CLASSLA tokenization/NER resources. An optional
`--resources` selects another CLASSLA resource directory. XLM-R downloads through
Hugging Face's cache. Its pipeline-compatible checkpoint uses the author's
lossless pretokenization and simple aggregation. Full articles use overlapping
windows (`--max-length 256 --stride 64`); nothing is silently truncated. Entity
offsets are half-open Python character offsets into `articles.content`, not
newspaper-level source coordinates. Every span is checked against the source.

## 3. Compare throughput

Stop competing semantic indexing/llama jobs yourself before measuring. Run models
sequentially. Use three batch sizes with one repeat to select settings, then three
repeats of the best setting:

```powershell
& $py -m semora.diagnostics.ner sweep --sample artifacts/ner-benchmark/sample.sqlite `
  --adapter classla --device cuda --batch-sizes 16 32 64 --repeats 1 `
  --corpus-articles 6318092 --output artifacts/ner-benchmark/classla-sweep
& $py -m semora.diagnostics.ner sweep --sample artifacts/ner-benchmark/sample.sqlite `
  --adapter transformers --device cuda --batch-sizes 16 32 64 --repeats 1 `
  --corpus-articles 6318092 --output artifacts/ner-benchmark/xlmr-sweep
```

Each configuration runs in a fresh child process, with its own log, indexed
`predictions.sqlite` and `metrics.json`. A failure or per-process timeout (default
two hours, `--timeout`) does not prevent subsequent configurations running. The
sweep exits unsuccessfully if any configuration fails. Incomplete runs cannot
be used in the report. Committed partial predictions remain for diagnosis, but
timed benchmarks do not resume: run again into a new output directory.

`--batch-size` means NER sentence batches for CLASSLA and model-window batches for
Transformers. `--batch-articles` (default 32) controls the outer read/write batch.
Use each model's fastest stable settings; equal batch numbers are not equal work.
`--precision` controls Transformers only; CLASSLA uses its native precision.

Timing includes sample SQLite reads, prediction, validation, indexed mention
writes and commits. Startup and warm-up are separate. This is **not a measurement
of reads from the full production database**; the sample may fit in the OS cache.
Transformers combines tokenization/inference/postprocessing timing because its
pipeline interleaves them. CLASSLA reports those stages separately. Peak RSS is
sampled at 100ms; CUDA allocated/reserved peaks include load and warm-up and do
not equal device-wide memory usage. Throughput estimates assume the uniform sample
represents article lengths; they are not guaranteed production runtimes.

## 4. Review cached disagreements

```powershell
& $py -m semora.diagnostics.ner report --sample artifacts/ner-benchmark/sample.sqlite `
  --runs artifacts/ner-benchmark/classla-smoke artifacts/ner-benchmark/xlmr-smoke `
  --names CLASSLA XLM-R --limit 40 --output artifacts/ner-benchmark/review
```

Open `review/report.html` in a browser; no server or external assets are needed.
It shows one short highlighted passage from each of up to 40 sampled disagreeing
articles, preferring person disagreements within each article. Metadata and exact
offsets are in `review.json`; per-run timings are in `timings.csv`. The default
comparison uses PER/LOC/ORG, configurable with `--labels`. Full predictions retain
all labels. All runs must have completed the same sample prefix.

Disagreement rates and entity counts are **not precision, recall or F1**. Shared
omissions are invisible. Reports expose practical differences without implying
that one model is the reference answer. Scores are model-specific and should not
be compared directly. A further one-hour stability run is advisable before a
full-corpus job.

## Adding models

Use `--adapter transformers --model organization/checkpoint` for another
pipeline-compatible token classifier. Adjust window length, stride and optional
pretokenization to the checkpoint documentation. For another architecture, supply
`--adapter your_package.your_module:factory`. The factory accepts the run options
and returns an object implementing `base.Adapter`: `metadata` plus
`predict(texts, batch_size) -> Prediction`, with one entity list per input.
Adapters load lazily; the benchmark/report code needs no changes. An adapter is
responsible for complete document coverage and correct source-relative offsets.
