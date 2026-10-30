# Additive newspaper indexing

Agreed with Mitko Nikov on 2026-10-29. Semora owns the batch state and indexing;
source adapters supply verified input manifests. NORA owns deployment and source
preparation. A batch uses existing data plus selected new issues without copying
the historical source corpus into a new build directory.

## Current scope

The implemented path covers strict manifests, read-only planning, additive
newspapers/articles/chunks, explicit historical baseline audit/adoption,
batch surface/lemma indexing, private semantic vector/FAISS generations and
audited batch NER extraction, additive entity dictionary updates and a full
publication gate. Linux/Docker packaging and production qualification remain.
Each indexing stage reports `retrieval_ready=false` until a complete serving
revision is selected and readers restart.

```text
reviewed CSVs -> source adapter -> batch.json + existing source files
                                      |
                             read-only plan
                                      |
                       SQLite batch/corpus ledger
                                      |
                                  articles
                      +---------------+---------------+
                      |               |               |
                surface -> lemma    chunks          NER
                                      |               |
                                vectors          dictionary
                                      |
                                 FAISS
```

Historical article/lexical records enter the ledger through explicit baseline
adoption. Source paths and original IDs stay unchanged. Missing issues in a new
batch are never deletion requests; changed OCR requires a future coordinated
replacement operation. Metadata-only corrections do not repeat text processing.

## Inputs, paths and command I/O

All relative CLI paths resolve against the current working directory.
`--root` is only the fallback for `--database`: `<root>/indexes/semora.sqlite`.
It does not rebase `--manifest`, `--input-root`, `--database` or `--output`.

| Input/output | Purpose |
| --- | --- |
| `--manifest` | Explicit `semora.batch.v1` JSON contract. No directory scanning. |
| `--input-root` | Resolves only `markdown_path` and `metadata_path` inside that manifest. |
| `relative_path` | Canonical logical source path used for stable IDs; independent of host/data/input location. |
| `--database` | Existing or new corpus SQLite path; baseline commands require an existing archive. |
| `--semantic-dir` | Semantic artifact root; default `<database parent>/semantic`, independent of `--root`. Holds `mapping.sqlite`, `vectors/`, and private `generations/`. |
| `--semantic-generation` | Exact 64-character directory ID of a complete private FAISS generation; publication never chooses one implicitly. |
| `--entity-dir` | Dictionary root; default `<database parent>/entities`. The adopted SQLite file stays in place. |
| Models/resources | Loaded only by a selected stage that has pending model work. Planning, status, baseline audit/adoption and newspaper storage require no NLP model. |
| stdin | Unused. Commands read their explicit files/database/configuration. |
| stdout | Full JSON report; with `--output`, only `{"report_path":"<absolute path>"}`. |
| stderr | Structured JSON progress events; keep separate from report stdout. |
| `--output` | Optional full JSON report, published atomically to a new file. Existing outputs are refused before command execution; parent directories are created. |

The runner never copies/downloads issue files, assembles a complete source tree,
or starts/stops services. `--maintenance` is an operator acknowledgement that
readers and older writers have stopped, not process detection or service control.
Keep writable SQLite and its journal/WAL on local storage shared by same-host
processes. A source batch can live elsewhere; it must remain immutable while used.

| Command | Database/filesystem side effects |
| --- | --- |
| `pipeline plan` | Reads and hash-verifies every selected source pair; compares the database read-only without migration. Only an explicit report file is written. |
| `pipeline baseline audit` | Reads existing database records, validates stored article derivation and lexical evidence. No migration, adoption or model inference. Only an explicit report file is written. |
| `pipeline baseline adopt --maintenance` | Repeats the audit under writer exclusion and registers verified historical article/lexical completion. Adds ledger/schema state; preserves historical source and index rows. |
| `pipeline run --maintenance` | Validates all inputs before corpus changes, then commits selected source/article/chunk outputs and stage markers per issue. Creates database/ledger and lock files as needed. No retrieval publication. |
| `pipeline index --maintenance` | Adds selected batch surface/lemma outputs and durable completion; preserves completed compatible work. Acquires writer locks. No semantic/entity publication. |
| `pipeline semantic audit` | Reads the existing v2 semantic manifest, mapping, corpus and every raw shard; reports eligibility and evidence without migration, model loading or file changes. |
| `pipeline semantic adopt --maintenance` | Repeats the audit, records historical chunk input hashes and corpus identity in the mapping database, and leaves old vectors and serving FAISS files unchanged. |
| `pipeline semantic vectors --maintenance` | Embeds only unmapped chunks in the selected completed batch, writes immutable float16 shards and mapping checkpoints; resumes interrupted publication from its journal. |
| `pipeline semantic faiss --maintenance` | Builds or extends a private, complete FAISS generation from verified shards, with resumable checkpoints; does not replace the serving index or make retrieval ready. |
| `pipeline ner audit` | Reads one historical NER run, eligible article contents and every mention in a snapshot; reports exact coverage and structural evidence without migration or model load. |
| `pipeline ner adopt --maintenance` | Repeats the full audit under writer exclusion and records a historical NER receipt/model identity; preserves extraction rows. |
| `pipeline ner index --batch <id> --maintenance` | Extracts only pending valid nonblank articles in the selected batch, with atomic completion and mention checkpoints. Loads the adapter only when work is pending. |
| `pipeline entities audit` | Reads the legacy entity manifest, dictionary, spellfix vocabulary and current NER output; reports valid existing pairs and pending pairs without mutation. Requires a trusted installed spellfix library. |
| `pipeline entities adopt --maintenance` | Requires complete audited legacy dictionary and adopted NER; binds the existing dictionary file to the corpus/NER profile and builds a one-time article lookup index. Does not copy the dictionary or publish search. |
| `pipeline entities index --batch <id> --maintenance` | Adds only missing selected-batch name/article memberships, affected counts and spellfix ranks in bounded transactions. Preserves entity IDs and root serving manifest. |
| `pipeline publication --batch <id> --semantic-generation <id>` | Read-only full gate over ingestion, both lexical paths, NER, all vector shards/FAISS IDs and the complete entity dictionary. Returns a candidate pointer only when eligible; optional new JSON report. |
| `pipeline publication --batch <id> --semantic-generation <id> --publish --maintenance` | Repeats the full gate under writer locks, then atomically replaces `<database parent>/publication.json`. A failed gate returns an ineligible report and leaves the pointer untouched. Restart retrieval readers after success. |
| `pipeline status` | Reads committed batch/stage state. Does not create/migrate a missing database or verify every external artifact. |

The gate checks the selected batch and verifies that no other registered
batch has unfinished ingestion or lexical coverage; its semantic generation
must cover every eligible chunk in the current corpus. It also checks global
NER/entity dictionary consistency. `pipeline publication` accepts `--database`, `--semantic-dir`,
`--entity-dir`, `--run-id` (default `classla-sl`), and optional trusted
`--spellfix-path`. Without `--publish` it is read-only and does not require
`--maintenance`. It can take substantial time: it streams the full 15.7 GB
legacy dictionary and validates every vector shard and FAISS ID. The audit
returns `candidate` only when all six stage checks pass. `--publish` returns
`published=true`, `publication_id` and `restart_readers=true` on success; an
idempotent retry preserves the same publication ID. It does not restart NORA.

The pointer selects one immutable FAISS generation and one revision token for
entity cursors. The entity SQLite file and corpus/semantic mappings are mutable,
so each writer commits a corpus publication epoch change before its first
output. Readers reject a missing or stale pointer after that change. Entity
checkpoints also advance a matching corpus/dictionary token; this detects a
restored older corpus even when its logical UUID is unchanged. Stop existing
readers and legacy writers, take a consistent backup, then use `--publish` and
restart the gateway. Previously deployed readers do not understand this gate.
After restart, reader health reports the audited publication ID and current
published article totals for BM25 and lemma; legacy pre-publication counters
are no longer used as the active coverage count.

## Manifest contract

A minimal illustrative record (replace the zero hashes with actual byte hashes):

```json
{
  "format": "semora.batch.v1",
  "batch_id": "newspapers-2026-10-27",
  "issues": [{
    "relative_path": "Straza/GF-URN_example/GF-URN_example.md",
    "markdown_path": "Straza/GF-URN_example/GF-URN_example.md",
    "metadata_path": "Straza/GF-URN_example/GF-URN_example_metadata.json",
    "markdown_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "metadata_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "source_identifier": "GF-URN:example"
  }]
}
```

The input paths may differ from `relative_path`. They must be portable relative
POSIX paths that resolve inside the input root, including after symlink
resolution. Markdown must be nonempty UTF-8 and metadata a JSON object. Both
exact byte hashes are checked; stored newlines normalize like the historical
reader so Windows line endings do not alter existing identity/span conventions.

Manifest identity is independent of its file location and issue ordering. Reuse
a batch ID only with the same manifest and processing configuration. Case-colliding
canonical paths, duplicate source identifiers, changed text, identifier disagreement
and stale replay after a later metadata correction are rejected. A source
identifier omitted from the manifest is derived from unambiguous metadata.
Straza retains `GF-URN` and provenance; the normalized dLib `urn` stays absent.

Source identity registration covers historical adoption as well as new imports.
Migration 016 stores case-folded dLib/GF identifiers independently of canonical
paths, so an adopted issue cannot return under another folder or identifier case.
Read-only planning against older databases checks stored metadata without
requiring that migration first.

## Operator sequence

Use disposable databases to qualify each milestone. The following examples show
explicit paths; they do not authorize production adoption or import.

```sh
semora pipeline plan --manifest /imports/batch.json --input-root /imports/issues \
  --database /data/indexes/semora.sqlite --output /reports/batch-plan.json
semora pipeline baseline audit --database /data/indexes/semora.sqlite \
  --output /reports/baseline-audit.json
# After reviewing the audit, stopping readers/writers and taking consistent backups:
semora pipeline baseline adopt --database /data/indexes/semora.sqlite --maintenance
semora pipeline run --manifest /imports/batch.json --input-root /imports/issues \
  --database /data/indexes/semora.sqlite --through chunks --maintenance
semora pipeline index --database /data/indexes/semora.sqlite --batch newspapers-2026-10-27 \
  --stage surface --maintenance
semora pipeline index --database /data/indexes/semora.sqlite --batch newspapers-2026-10-27 \
  --stage lemma --classla-device cuda --maintenance
semora pipeline semantic audit --database /data/indexes/semora.sqlite \
  --semantic-dir /data/indexes/semantic --output /reports/semantic-audit.json
semora pipeline semantic adopt --database /data/indexes/semora.sqlite \
  --semantic-dir /data/indexes/semantic --maintenance
semora pipeline semantic vectors --database /data/indexes/semora.sqlite \
  --semantic-dir /data/indexes/semantic --batch newspapers-2026-10-27 --maintenance
semora pipeline semantic faiss --database /data/indexes/semora.sqlite \
  --semantic-dir /data/indexes/semantic --maintenance
semora pipeline ner audit --database /data/indexes/semora.sqlite \
  --run-id classla-sl --output /reports/ner-audit.json
semora pipeline ner adopt --database /data/indexes/semora.sqlite \
  --run-id classla-sl --maintenance
semora pipeline ner index --database /data/indexes/semora.sqlite \
  --batch newspapers-2026-10-27 --run-id classla-sl --device cuda --ner-decoder numba --maintenance
semora pipeline entities audit --database /data/indexes/semora.sqlite \
  --entity-dir /data/indexes/entities --run-id classla-sl --output /reports/entities-audit.json
semora pipeline entities adopt --database /data/indexes/semora.sqlite \
  --entity-dir /data/indexes/entities --run-id classla-sl --maintenance
semora pipeline entities index --database /data/indexes/semora.sqlite \
  --entity-dir /data/indexes/entities --batch newspapers-2026-10-27 --maintenance
semora pipeline status --database /data/indexes/semora.sqlite --batch newspapers-2026-10-27
```

Semantic vector defaults match the existing scripts: `google/embeddinggemma-300m`,
unknown model revision, 256 dimensions, 64 texts per encode call and 10,000 vectors
per shard. `--model-revision` records a pinned revision when supplied; an omitted
historical revision is never described as verified. The encode path uses
`encode_document` where available and L2-normalizes float32 output before
storing little-endian float16 vectors. A semantic audit checks stored IDs,
source spans, configuration and shard bytes; it cannot reproduce the exact
historical model or tokenizer revision from absent metadata. FAISS defaults to
IVFPQ (16,384 lists, 32 PQ subvectors, 8 bits, one million training samples,
`nprobe=32`); `--index-type flat` is useful for small pilots. `--checkpoint-shards`
defaults to 10. Completed private generations live under
`<semantic-dir>/generations/<fingerprint>/`; serving root files remain untouched.

NER `--run-id` selects the extraction profile (default `classla-sl`). The
historical audit checks the producer's valid nonblank article membership,
current content hashes, mention counts, indices, spans and text, including
explicit zero-mention completions. Stored model/source hashes are evidence;
old execution is not rerun or claimed proven. A populated legacy run requires
`ner adopt` before additive extraction. New runs store stable model identity
separately from device, resource paths and batch tuning; incompatible hashes
require a new run ID. `ner index` accepts `--adapter`, `--device`, `--resources`,
`--batch-articles`, `--batch-size`, `--pipeline-depth`, `--tokenizer-workers`,
`--ner-pipeline-depth` and `--ner-decoder`. It does not update the entity
dictionary or serving search index.

Entity commands take `--entity-dir` (default `<database parent>/entities`) and
optional `--spellfix-path` (default `<database grandparent>/extensions/spellfix1.dll`
on Windows or `.so` on Linux). The historical audit validates stored normalized
names, per-article earliest offsets/content hashes, counts and spellfix entries
against the selected NER run. `eligible` permits valid but missing expected
memberships; adoption additionally requires `complete=true`. Adoption changes
only the existing dictionary's control tables and creates an article-first
lookup index. It is a one-time index build over the existing large file, so plan
disk space and backup accordingly. `entities index` uses `--batch-articles`
(default 1000) as the transaction checkpoint size. It mutates the adopted
dictionary file in place while readers are stopped; it does not clone the 15 GB
legacy artifact or replace the root manifest. A failure rolls back the current
checkpoint; earlier checkpoints remain and retry checks per-article completion.
Publication must issue a new reader token after all index gates pass. Until then,
the dictionary report says `retrieval_ready=false`.

`--through` selects an ingestion prefix: `newspapers`, `articles` or `chunks`
(default). Rerun the same manifest/settings to advance or resume. Selected stages
for one issue commit together; failure rolls back that issue while preserving
previously committed issues. Zero-output stages also get completion records.
`state=ingested` means the selected prefix finished; `ingestion_complete` requires
all three corpus stages. Neither means all retrieval indexes are ready.

Chunk settings are `--model-id`, `--tokenizer-revision`, `--token-count` and
`--token-overlap`. Pin an immutable tokenizer revision for new reproducible work;
unknown historical revisions remain unknown. The runner rejects incompatible
completed stage settings instead of silently rechunking historical text.

Lexical `--stage` is `surface` or `lemma`; `--batch-size` controls the bounded
write/inference batch. Lemma options include `--classla-type`,
`--classla-device`, `--classla-resources-dir`, `--classla-pos-batch-size`,
`--classla-lemma-batch-size` and optional `--lemma-model-fingerprint`. Throughput
settings do not identify model compatibility. Supply a verified model fingerprint
when known; this argument records an operator assertion and does not compute or
verify model weights. Adoption does not invent one for historical outputs.

`pipeline status` reports eligible articles and recorded/pending surface/lemma
completion for the selected batch. These are bookkeeping counts, not an audit
of FTS text. The publication gate checks stored membership and configured
index structures before selecting a serving revision. Do not reopen the
gateway using a completed lexical stage alone.

## Historical baseline and compatibility limits

Baseline audit is a full stored-corpus validation, not a cheap status lookup.
It checks canonical paths/IDs/source identities, re-splits stored issue text and
compares exact article identity/content/spans/run associations, then checks lexical
membership/rowids and historical completion evidence. It does not read the old
source tree or run tokenization, CLASSLA, embeddings or NER. Plan separately
verifies incoming files and their hashes.

Corpus adoption records only validated article and eligible lexical outputs.
Semantic, NER and entity dictionaries have separate audits and adoption receipts; none infers
historical model execution from stored declarations. Historical contentless FTS text is not
reconstructed or cryptographically verified by this audit. Supported FTS schema
and tokenizer settings are checked against the actual table definitions. Counts and legacy
`complete` flags alone do not prove coverage or model compatibility. Audit the
schema first: historical migration 012 clears lemma state when initially applied;
additive ingestion requires migrations 001-013 already present before migration
014 rather than silently applying older destructive migrations.

Top-level `eligible` describes article-baseline eligibility. Inspect the separate
`lexical.surface.eligible` and `lexical.lemma.eligible` fields before assuming
those outputs will be adopted. Adoption may add the new ledger schema even when
the content audit subsequently fails; failed audit/marker transactions preserve
historical source/index rows and add no completion markers.

Database checkpoints and JSON report publication are separate. A report write
failure does not undo completed database work; inspect status and retry with a
fresh report path. Planning conflicts and an ineligible article audit save their
reports before returning exit code 1; successful commands return 0.
Model/backend diagnostics are redirected to stderr so stdout remains valid JSON.
Baseline receipts retain the full evidence report; distinct audited snapshots
receive separate receipts rather than overwriting earlier adoption evidence.

Legacy full-build APIs reject archives containing registered additive batches.
Use the batch-aware lexical entry point rather than resetting old global flags.
The low-level artifact-only FAISS builder cannot inspect a corpus database from
its saved basename; it is not a route around the corpus guard. Locks do not
protect against an older binary already running: maintenance still requires
stopping those processes explicitly.

## Remaining roadmap and acceptance

1. **Native qualification:** exercise A/B overlap, interruption/recovery and
   relocation on a disposable archive, then audit and publish the reviewed
   production batch in a stopped-reader maintenance window. SQLite and
   FAISS/dictionary files cannot share one transaction.
2. **Semantic lifecycle:** define eventual compaction or
   retraining from retained vector shards, without repeating old embeddings.
3. **Portability/package:** keep one Python runner, thin platform wrappers,
   independent model/data/report roots and a GF adapter emitting this contract.
   Build pinned Semora/CLASSLA wheels, qualify Python/PyTorch/CUDA, package Linux
   spellfix, use persistent local data/model volumes and runtime credentials.
   No distributed queue is required for the current single-host coordinator.

Acceptance uses small batches A/B before the reviewed production batch: import A
twice; overlap B; interrupt/resume around transactions and artifact publication;
include new hashed IDs sorting before old checkpoints; verify no old NLP repeats,
no duplicates/missed work, stable IDs/citations, metadata corrections, Straza
filter/source/citation behavior and input/data-root relocation. Qualify the same
logical results on native Windows and Linux/container runs before claiming Docker
portability. No production corpus mutation, baseline adoption or cutover has been
performed merely by implementing these commands.
