# Production article entities

`semora index ner` saves extraction results directly in `indexes/semora.sqlite`.
It does not create a new BM25 index or expose a Nora entity-search tool.
It processes valid, nonempty **article content**, not chunks or article titles.

## Run and resume

Use the tested editable CLASSLA fork and the benchmark environment for now.
The Numba decoder requires Numba installed in that same environment. NumPy is
the CLI default; explicitly request Numba for the optimized path.

```powershell
$py = "C:\GitHub\semora\artifacts\ner-benchmark\.venv\Scripts\python.exe"

& $py -m semora.cli.main --root C:\GitHub\nora index ner `
  --run-id classla-sl --device cuda `
  --batch-articles 32 --batch-size 16 `
  --pipeline-depth 3 --tokenizer-workers 2 `
  --ner-pipeline-depth 1 --ner-decoder numba `
  --max-articles 10000
```

Remove `--max-articles` to continue through all eligible articles. Otherwise the
limit is the **total completed article target for this run**, including work
already saved. Running the same target twice adds nothing. Ctrl+C or a failed
batch leaves earlier committed batches available for resume; uncommitted work
is recomputed. Errors are reported, not silently skipped. Do not run competing
GPU workloads during extraction.

The model identity, editable CLASSLA source fingerprint and inference settings
are recorded with the run. Reusing an ID with different recorded settings is
rejected; choose a new `--run-id` to retain both outputs. Limits and outer
pipeline depth may change on resume. Model metadata also includes tokenizer
worker and inner-pipeline settings; keep those unchanged for the same run.

Use an outer-article limit divisible by `--batch-articles` when comparing exact
outputs: cutting off a batch can change its neural batch composition, just as
in the benchmark. Extraction always uses original sentence order.

## Storage

- `ner_runs`: extraction identity and JSON configuration.
- `article_ner_state`: one completion row per article/run, content SHA-256 and
  entity count. Articles with zero entities are recorded too.
- `article_entities`: one row per mention, with text, label, optional score and
  `char_start`/`char_end`. End offsets are exclusive Python character offsets
  within `articles.content`, **not original newspaper Markdown offsets**.
- Existing `runs`/`logs`: one attempt per invocation, start/finish/failure events.

```sql
SELECT a.article_id, n.urn, n.date, e.text, e.label, e.char_start, e.char_end
FROM article_entities e
JOIN articles a ON a.article_id = e.article_id
JOIN newspapers n ON n.newspaper_id = a.newspaper_id
WHERE e.ner_run_id = 'classla-sl' AND e.text = 'Marija Abram'
LIMIT 100;
```

CLASSLA does not provide mention confidence here, so `score` is NULL.
Mention lookup is exact and case-sensitive; this is not fuzzy name search or
entity linking (different spellings are not automatically the same person).

## Safety and concurrency

Migration 013 adds tables/indexes and a trigger; it does not rebuild or delete
existing BM25/lemma/semantic indexes. Back up the database before migrating.
Each batch validates spans and saves completion records and mentions in one
transaction. A second NER builder for the same database is rejected by an OS
file lock, which is released even when its owning process exits unexpectedly.
The `.ner.lock` sidecar may remain on disk; its presence alone is not a lock.

Preparation, single-owner inference and writes use the shared bounded pipeline.
Reads use rowid keyset iteration and short-lived connections; no OFFSET, full
content preload, or initial corpus text count. Searches can continue, although
SQLite may briefly serialize reads/writes depending on journal mode.

Avoid modifying/ingesting articles during extraction. Text and validity are
checked again before commit. Updating content/validity invalidates the affected
article's NER state and entities; deletion cascades too (requires SQLite foreign
keys enabled, as with Semora connections). A subsequent run re-extracts changed
eligible articles. Appended articles beyond the invocation's initial rowid upper
bound are picked up on the next invocation. `exhausted_snapshot` reports whether
the reader reached the end, rather than merely stopping at a requested limit.

## Validation (2026-09-15)

An isolated production-schema database was populated with 64 articles from the
existing benchmark sample. The actual CLI and CUDA/Numba adapter were run three
times: 32 articles (562 mentions), resume to 64 (415 additional mentions), then
a no-op resume (zero articles/entities added). All 977 saved mentions matched
the validated benchmark exactly, including text, labels, offsets, ordering and
scores. SQLite integrity and foreign-key checks passed. This was a correctness
smoke test, not a throughput benchmark.

The focused suite passed 29 tests covering shared benchmarking and production
extraction, including zero-entity articles, invalid/empty articles, mismatched
configurations, failures, rollback, source changes, deletion, lock contention
and resume. Including existing corpus/search and public API regressions, 42
tests passed (two optional benchmark model tests were not selected). The live
Nora database was not modified.
