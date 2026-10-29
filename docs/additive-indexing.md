# Additive newspaper indexing

Design agreed with Mitko Nikov on 2026-10-29. This is the implementation roadmap
for recurring newspaper imports, including sources without dLib URNs. It does
not assert that the complete pipeline or Docker packaging is implemented.

The first milestone introduces manifest-driven batch planning and additive
corpus ingestion. Downstream lexical, semantic and entity stages need the
changes below before an imported batch can be published for retrieval. The
public legacy BM25, lemma, semantic, vector, NER and entity-dictionary builders
now reject a corpus containing registered `pipeline_batches` before creating
stage outputs or loading models. An empty batch ledger does not block legacy
databases. Do not interpret old global completion flags as proof of coverage
for an appended corpus.

## First milestone: available commands

`pipeline plan` verifies every selected file/hash and compares the issues with
an existing database using a read-only connection. It neither initializes nor
migrates SQLite. Input paths are relative to `--input-root`; canonical
`relative_path` values determine corpus IDs independently of that input root.

```sh
semora pipeline plan --manifest /imports/batch.json --input-root /imports/issues \
  --database /data/indexes/semora.sqlite
```

`pipeline run` currently implements corpus ingestion only. Use disposable test
databases until the downstream milestones and baseline audit are complete.
The maintenance flag acknowledges that readers and legacy writers are stopped;
it does not detect or stop those processes.

```sh
semora pipeline run --manifest /imports/batch.json --input-root /imports/issues \
  --database /test/indexes/semora.sqlite --maintenance --through newspapers
semora pipeline status --database /test/indexes/semora.sqlite --batch example-batch
```

The `--through` choices are `newspapers`, `articles`, and `chunks` (default).
Rerun the same manifest and configuration to advance through those stages or
resume after failure. Each issue's selected stages commit together; previous
issues remain committed when a later issue fails. Empty article/chunk outputs
also receive completion records. `state=ingested` means the requested ingestion
prefix finished; `ingestion_complete` requires all three stages for every issue.
`retrieval_ready` is always false in this milestone.

For chunks, configure `--model-id`, `--tokenizer-revision`, `--token-count`, and
`--token-overlap`. Pin an immutable tokenizer revision for reproducible runs.
The old full-build command remains available for legacy databases. Before
applying migration 014 to an existing database, the runner requires historical
migrations 001-013 to have already been explicitly applied; it will not silently
run older destructive migrations.

A batch ID cannot be reused with a different manifest or processing config.
Text replacements, case-colliding canonical paths, duplicate source identifiers,
and stale replays of metadata superseded by a later batch are rejected.
Existing historical article outputs require explicit baseline adoption before
any additive run is allowed. The runner also rejects a selected stage whose
configuration differs from already completed pipeline stages. Adoption is a
later milestone, not an implicit success. This prevents using the corpus-only
milestone to change the active legacy database accidentally.

The manifest format is `semora.batch.v1`, with `batch_id` and an `issues` array.
Each issue has `relative_path`, `markdown_path`, `metadata_path`,
`markdown_sha256`, `metadata_sha256`, and optional `source_identifier`.
Source files must be nonempty UTF-8 Markdown plus JSON-object metadata, resolve
inside the input root, and match their byte hashes. Stored Markdown normalizes
newlines exactly as the legacy reader did, preserving its ID/span conventions.
The source identifier is checked against recognized metadata identifiers; a
missing declared identifier is derived from an unambiguous dLib/GF identifier.

The migration adds only the batch ledger and completion tables; it does not
modify historical corpus/index rows. Planning and status need no NLP models.

The low-level `build_faiss_index(output_dir)` API receives only vector artifacts;
their saved database name is a basename, not a trustworthy source database
location. It cannot apply the corpus guard, and remains an artifact-only API.
Use the guarded `build_semantic_index(database_path, ...)` entry point for corpus
indexing. A guard is not mutual exclusion: operators must stop legacy writers
before beginning an import, including any job started before the guard check.

## Scope and invariants

- Import only the explicitly selected issues. Missing issues in a later batch
  are not deletions. Do not scan arbitrary Markdown beside a manifest.
- Preserve existing newspaper, article and chunk IDs and source spans. Corpus
  identity uses the canonical relative path; absolute input/data directories
  and the working directory are not identity inputs. Input paths may differ
  from the canonical corpus path.
- Add new issues, skip unchanged issues and update metadata without repeating
  text processing. Reject changed OCR, identity conflicts and deletion requests
  until replacement/invalidation is implemented across every derived index.
- Preserve Straza's `GF-URN` and provenance separately from an absent dLib URN.
  Do not manufacture a dLib link. Metadata normalization already leaves the
  normalized `urn` empty for `GF-URN` identifiers.
- A retry must not duplicate records or skip uncommitted work. Completion is
  attached to the input revision and semantic processing configuration.
- Batch IDs identify imports; corpus/chunking/NER runs identify processing.
  Several compatible runs must be able to contribute to one searchable corpus.
- Stop retrieval readers for the initial implementation's maintenance window.
  Restart them only after all required stages and coverage checks succeed.
  Continuous availability needs additional reader/publication changes.

The existing `_stable_id`, `_article_record` and `_token_chunks` functions in
`src/semora/corpus/indexer.py` define the current ID conventions. Changing a
splitter or tokenizer can change boundaries even when source text is unchanged;
that is a processing revision, not an ordinary additive import.

## Batch state and orchestration

Use SQLite to record immutable manifest identity, batch membership, input hashes,
processing configurations, attempts and stage completion. A batch may be resumed
with the same manifest; reusing its ID for different contents must fail.
Validate paths, hashes, UTF-8 source text and JSON metadata at the input boundary.
Paths must remain within the selected input root after resolving symlinks.

Select pending work through batch membership and missing completion records,
not a global last hashed ID. A new ID may sort before the old checkpoint.
Bound each attempt to an explicit input set so later batches cannot silently
extend a running job. Successful zero-output processing must also be recorded.

Keep each issue's outputs and completion in one transaction for SQLite-only
stages. Existing `Database.insert_*` helpers open and commit their own
transactions; they cannot simply be nested inside an outer per-issue transaction.
Reuse transaction-aware primitives or issue the transactional writes directly.

Use one coordinator lock for mutations of a data root, with bounded internal
workers and one controlled SQLite writer. Existing `.vectors.lock`, `.ner.lock`
and entity `.build.lock` files exclude only their own stages; they do not prevent
ingestion from racing with indexing. Legacy entry points must participate in the
same exclusion policy, or be disallowed while a pipeline job is active. An
unrelated lock file by itself cannot protect against an older binary.

The dependency graph is:

```text
validate -> newspapers -> articles
                           |-> surface BM25 -> lemma BM25
                           |-> chunks -> vectors -> FAISS
                           `-> NER -> entity dictionary
```

CPU/GPU scheduling may serialize these branches without changing their logical
dependencies. Do not add a distributed broker until there is a demonstrated
need for work across several hosts.

## Changes required by stage

| Stage | Current implementation | Additive work |
| --- | --- | --- |
| Newspapers | `corpus/indexer.py` rejects an existing database and requires an empty table. | Validate selected inputs; compare hashes/content and metadata; transact only new issues or metadata changes. |
| Articles | Requires an empty table and processes all newspapers. | Process pending selected issues, preserving IDs/spans and recording completed empty issues. |
| Chunks | Requires an empty table and exactly one corpus run. | Process pending eligible articles using a compatible tokenizer/chunking specification across runs. |
| Surface BM25 | `retrieval/indexing.py` uses `last_article_id`, a global `complete`, and `indexed + 1` for the next FTS ID. | Select missing eligible mappings, allocate collision-free FTS IDs, and commit FTS rows/mapping/completion together. Report batch and corpus coverage separately. |
| Lemma BM25 | Requires the stored surface count to remain unchanged, checkpoints by article ID and stops at global completion. | Reuse existing `article_lemmas`, process only pending compatible articles, and transact materialized text/FTS/completion together using the surface FTS mapping. |
| Vectors | `semantic_vectors.py` and `semantic_store.py` describe a fixed chunk count, one chunking run and hashed-ID cursor. | Retain immutable shards; append only pending chunk revisions to a durable vector-ID map, with batch membership and compatible model/tokenizer settings. |
| FAISS | `semantic_faiss.py` already uses `add_with_ids`; its checkpoint fingerprint embeds a fixed semantic specification. | Extend a compatible trained index in a private generation and journal included shards/vector IDs; publish only when validated. |
| NER | `ner/indexing.py` already selects articles lacking `article_ner_state` and bounds each attempt by an upper rowid. | Add batch selection, harmonize configuration/revision rules and coordinator locking, and verify exact pending coverage. |
| Dictionary | `ner/names.py` scans all mentions into a fresh immutable database. | Clone a compatible generation and apply new memberships, affected article counts and spellfix entries once; publish a complete next generation. |

Configuration fingerprints must distinguish output compatibility from throughput
tuning. Model/tokenizer revisions, splitter behavior, dimensions, normalization
and relevant CLASSLA configuration are compatibility inputs. Worker counts and
batch sizes should not invalidate compatible completed outputs. Currently NER's
run configuration also includes device and batching fields; separate these before
allowing tuning changes on resume. A device change still needs qualification if
it changes the promised output semantics.

NER already commits mentions and completion together, including zero mentions.
Migration `013_article_entities.sql` invalidates NER state when article content or
validity changes. This does not make arbitrary OCR replacement safe: lexical,
semantic and dictionary data need corresponding invalidation too.

### Semantic compatibility and recovery

The existing semantic resume path loads its saved specification and checks the
database basename, model, model revision and dimensions. It can return the old
fixed target as complete after the corpus grows. A fresh build rejects multiple
valid chunking runs. Neither path is an additive coverage check.

Separate an embedding compatibility specification from the batch input set and
artifact generation. Compatible chunking runs may have different run IDs. Every
chunk mapping must identify the exact source/configuration revision and vector
location. Never renumber historical vector IDs when new chunks arrive.

Write new shard data to a temporary file, flush and verify it, publish the file,
then commit its mapping/completion. Recover an orphaned published file by
verifying it against the journal or removing it before retry. Never record a
completed shard whose durable bytes do not exist. The corpus and mapping SQLite
databases and external files do not form one atomic transaction.

Build FAISS from the previous compatible generation plus verified new shards.
Check actual vector IDs and shard membership, not just `ntotal`: a count cannot
detect duplicate IDs or missing/replaced membership. Never mutate a FAISS file
currently opened by readers. Retain the previous generation until acceptance.
Periodic retraining/compaction may read stored vectors; it must not require
embedding the historical corpus again.

## Adopt the existing archive

Adoption is an explicit, audited migration, not an automatic declaration that
all historical records are complete.

1. Stop writers/readers and take consistent backups of the corpus database,
   mapping database and artifacts. Record code/schema/model versions.
2. Inspect schema migrations before calling initialization. Historical migration
   `012_materialized_article_lemmas.sql` clears lemma FTS/state when first applied;
   do not mistake `Database.initialize()` for a read-only audit.
3. Reconcile source paths/IDs, article spans and chunk configurations. Preserve
   legacy IDs and record available source/configuration evidence. Unknown model
   revisions remain unknown; do not infer them from today's cache.
4. Audit surface mappings against eligible articles, materialized lemmas against
   surface mappings, and NER state/mentions against article content hashes.
5. Audit semantic shard checksums and sizes, exact chunk/vector membership,
   dimensions, normalization, FAISS IDs and dictionary generation metadata.
6. Register verified outputs as the baseline. Report gaps separately and process
   only missing compatible work; reject incompatible artifacts rather than
   rewriting their metadata to claim compatibility.

No baseline migration, live corpus mutation, expensive NLP or production cutover
is implied by writing this design document.

## Portability and Docker boundary

Implement the pipeline once in Python under Semora. The source adapter handles
GF inventory/download/authentication and emits the same manifest contract that
local file imports use. NORA launches/configures the pipeline; its PowerShell
scripts become wrappers rather than orchestration logic.

Configuration supplies independent input, data, artifacts, model-cache and log
roots plus model identities, device and resource limits. Never persist a host's
absolute input location as document identity. Load dependencies/models only for
the selected stage, so planning and basic ingestion do not require CUDA.

Package a separate indexing image using pinned Semora/CLASSLA wheels and a tested
Python/PyTorch/CUDA dependency set. Replace editable sibling paths and the Windows
spellfix DLL with packaged dependencies and a tested Linux extension. Models,
database/artifacts and logs live in persistent volumes; credentials are runtime
inputs. Handle termination by finishing/rolling back the current checkpoint,
and emit structured progress and meaningful exit codes. Keep writable SQLite on
a local filesystem shared by same-host processes; do not place WAL databases on
a network filesystem.

Docker packaging should wrap the tested native runner. A container restart uses
the same durable batch state rather than starting another independent import.

## Delivery milestones and acceptance

1. **Manifest and corpus:** read-only plan, durable batch identity, additive
   newspaper/article/chunk stages, metadata-only updates, retry and rejection
   tests. Keep production untouched.
2. **Lexical and baseline audit:** adopt verified historical records, append
   surface/lemma indexes and prove pending work is complete by membership.
3. **Semantic:** append compatible shards and FAISS generations with failure
   injection around every file/database boundary.
4. **Entities and publication:** batch-aware NER, incremental dictionary and
   complete maintenance-window verification/restart.
5. **Packaging:** configuration, source adapter, dependency locks and image;
   exercise the same acceptance scenario natively and in Linux containers.

For each milestone, use small controlled batches before the approved production
batch. Import batch A twice, import overlapping batch B, and interrupt/resume at
transaction and artifact boundaries. Include a new hashed article/chunk ID that
sorts before the old checkpoint. Verify unchanged IDs/citations, no duplicate
outputs, no old NLP reprocessing, exact completion sets, Straza filtering and
source reading, metadata-only updates, conflicting IDs/text rejection, and
relocation of input/data roots. Model/configuration changes must fail clearly
or schedule an explicit compatible new generation.

Final publication requires corpus/index integrity, exact expected membership,
old/new lexical, semantic and entity smoke searches and source/citation checks.
Do not substitute a count, a successful process exit or a legacy `complete` flag
for those checks.
