# Optional CLASSLA execution backends

Local processing remains the default. NER and lemma indexing need no worker
service, Docker installation, network access or HTTP/server dependency.
Install the same task-specific model extras as before for local inference.

Additive CLASSLA NER can also use one or more authenticated [workers](classla-worker.md).
The coordinator uses only Python's standard library: it needs neither CLASSLA,
Torch nor the worker/server extra. Workers receive text and return predictions;
the coordinator alone owns the corpus, SQLite writes and publication state.
Remote lemmatization remains unavailable. Legacy `index ner` and `index lemma`
commands retain their existing local behavior.

## Run distributed NER

Start compatible workers with existing model resources and the same private token
file. Use private-network/VPN connectivity; use HTTPS on untrusted networks.
Every configured URL is an explicit destination for article text and credentials.
HTTP(S) origins only are accepted; embedded credentials, paths, queries, fragments
and duplicate origins are rejected. The client follows no redirects or environment
proxies, and HTTPS verifies certificates using the platform trust configuration.

```console
semora pipeline ner index --database /data/semora.sqlite --batch reviewed-batch \
  --maintenance --backend remote --run-id classla-sl-remote-v1 \
  --worker-url http://worker-a:8000 --worker-url http://worker-b:8000 \
  --worker-token-file /private/classla-token \
  --batch-articles 32 --worker-connect-timeout 5 --worker-read-timeout 300 \
  --worker-attempts 3 --output /reports/ner-remote.json
```

The example assumes a maintenance window with retrieval readers and legacy
writers stopped. Choose a new run ID for a remote model profile. This command
finishes NER only: entity dictionary construction and serving publication remain
separate steps. It never falls back to local inference.

```python
from pathlib import Path
from semora.execution import ExecutionConfig
from semora.pipeline.ner import index_batch_ner

result = index_batch_ner(
    database, batch_id, run_id="classla-sl-remote-v1",
    execution=ExecutionConfig(
        backend="remote",
        worker_urls=("http://worker-a:8000", "http://worker-b:8000"),
        token_file=Path("/private/classla-token"),
    ),
)
```

Omit `execution` / `--backend` to keep local behavior. Local mode rejects worker
URLs/token files. Remote mode rejects injected local models and adapters other
than CLASSLA. Lemma execution still uses the existing local process pool; remote
lemma requests fail before writes. The token must contain at least 32 visible
ASCII characters; a trailing newline is accepted. Secret contents never enter
run metadata or reports.

## Scheduling and recovery

Before any NER writes, all configured endpoints must be ready and advertise the
same v1 NER identity/fingerprint. Failure of even one configured endpoint aborts
preflight. Remove an unavailable endpoint explicitly before retrying. A fully
completed or empty batch returns from durable coverage without contacting workers.

At most one HTTP inference call per endpoint and one logical task per configured
worker are outstanding. The coordinator keeps only these tasks plus one bounded
`batch_articles` page of split work. Results may commit out of order. Worker
limits determine fixed subbatches using the smallest advertised document, UTF-8
text and raw JSON limits, capped by independent client ceilings. An oversized
single article fails clearly; it is never truncated. Local model tuning flags
(`--device`, `--batch-size`, `--pipeline-depth`, tokenizer/decoder settings and
`--resources`) do not configure remote services; configure each worker itself.

Each request keeps its ID, documents, hashes and composition across retries.
Connection failures/timeouts and HTTP 408/429/502/503/504 retry up to
`--worker-attempts` total attempts (default 3, maximum 10), rotating through
configured endpoints with a short delay. Other HTTP errors, identity changes,
malformed responses and invalid spans fail immediately. Responses are byte
bounded and must exactly match the request/task/fingerprint/document hashes;
NER offsets are checked against submitted text and restored to input order.

Connect and read timeouts default to 5 and 300 seconds. An overall socket deadline
of their sum also stops continuously dribbling peers. DNS resolution is subject
to the operating system's resolver deadline. Exiting the coordinator closes its
bounded executor after active calls finish or time out. It cannot cancel inference
already running on another computer; that worker keeps its slot until inference
finishes and can return busy to a retry. Inference may therefore execute twice.

Database completion is recorded once: the write transaction rechecks original
article content/validity, batch membership/state and the table-policy receipt,
then atomically saves mentions and completion, including
zero-entity articles. Identical duplicate results are skipped; different payloads,
changed input and orphan mentions are errors. Existing batch membership locks and
table masking rules remain in force. On restart, only articles without durable
completion are scheduled. There is no remote queue, Redis or shared SQLite mount.

Progress callbacks report durable counts after commits. The JSON report includes
workers attempted, attempts/retries/failures and sanitized last failure, plus
worker runtime/limit provenance. `--progress-file /path/ner-progress.json` writes
atomic Pebble snapshots from these counts, marking completion or failure. An
operator wrapper should remove only its task file after recording the producer's
outcome; unfinished work resumes from database state rather than that file.

## Model compatibility

Remote run metadata records `worker_identity` and `worker_fingerprint`, including
CLASSLA/Obeliks code, alignment code and model resources. Endpoint URLs, runtime
versions/device/decoder and throughput settings are provenance, outside model
identity. The first run record retains its execution settings; each invocation's
report records the workers used. Changing endpoints with identical semantics can
resume a remote run. Persist the reports when qualifying different runtime tuning.

Historical local metadata has weaker evidence. It is **not** automatically
compatible with the full worker identity, even when model hashes overlap. Use a
new NER run ID; there is no automatic migration/adoption override. Switching a
populated remote run back to the current local adapter likewise requires a new
run or a future explicit audited compatibility procedure. Matching fingerprints
do not prove identical predictions across hardware or changed batch composition.
CPU reference checks passed; GPU and clean second-machine qualification remain
separate milestones.

## Shared v1 contracts

`semora.execution.contracts` uses only standard-library dependencies and existing
entity data types. `Document` carries an ID, exact text and UTF-8 SHA-256;
`Request` carries the API version, request ID, task, fingerprint and documents.
`DocumentResult` carries the input ID/hash with entities or a lemma string, and
`Response.validate_against(request)` checks correlation, exact membership,
hashes, task payload and spans. Unicode offsets are end-exclusive Python
character positions; scores are finite numbers or null. Empty strings,
zero-entity results and empty document batches are valid contracts.

`semantic_fingerprint(identity)` hashes canonical versioned JSON, rejecting
nonstring keys, Python-only types and non-finite numbers. It does not discover
models or establish historical compatibility. Lemma contracts are defined now;
the future remote Lemmatizer will preserve the existing `lemmatize` /
`lemmatize_many` boundary after its own normalization and compatibility checks.


## Qualify an adopted historical CLASSLA run

Remote execution normally requires a new NER run because the worker fingerprint
contains more information than legacy metadata. For an adopted historical run,
`pipeline ner qualify` can record an explicit continuation receipt without
rewriting the historical run configuration or its adoption evidence. Stop
readers and writers first and pass `--maintenance`.

Provide `--reference` as JSON containing `metadata` from a loaded CLASSLA
adapter, its full `identity` from `build_classla_ner_identity`, and the
corresponding `fingerprint`. They must describe the same reference execution.
Historical model SHA-256, CLASSLA source SHA-256, version, package and normalized
processor metadata must match. The worker's model hash, version, package and
processor configuration are independently checked; its complete identity must
match the reference fingerprint. Source hash linkage to that full identity is
operator-supplied, not independently reconstructed from per-file digests.

Provide `--comparison` as JSON with `format` equal to
`semora.ner-comparison.v1`, positive integer `documents`, zero `mismatches`,
a nonempty `description`, and `evidence_sha256`. The description must state the
sample scope and any observed differences outside the compared pair. `--evidence`
points to the complete comparison artifact whose bytes must match that digest.
These values are operator-supplied; the command verifies the artifact digest but
does not replay inference or establish historical execution. Do not describe
a subset as complete parity or omit known counterexamples. Preserve the
artifact separately for review.

```text
semora pipeline ner qualify --database corpus.sqlite --maintenance \
  --backend remote --worker-url http://127.0.0.1:18081 \
  --worker-token-file token.txt --reference reference.json \
  --comparison comparison.json --evidence comparison-artifact.json
```

The receipt records worker identity and runtime, reference metadata, comparison
summary, and hashes of the historical configuration and adoption. Every remote
append checks the current fingerprint, qualified endpoint and exact advertised
runtime and rechecks those historical records. A selected subset of the
qualified endpoints is allowed. Hardware details not advertised by the API are
not independently verified. A changed worker requires a new explicit
qualification; the previous receipt is retained and later receipts are appended
to qualification history. Changed historical evidence invalidates those receipts.
Sample parity is not a guarantee of parity on all inputs.


## Trusted NER resume

`pipeline ner index --trust-completed-state --trusted-eligible-articles N`
skips the full revalidation of previously completed article text and mention
counts. `N` must come from a prior full preflight of the same unchanged batch.
The command counts existing batch completion markers using indexes unless
`--trusted-completed-articles M` supplies the exact last durable count from a
checkpoint of that same run and batch. Supplying the count avoids that scan.
This is an operator trust decision: completed rows, batch membership, article
validity and source text must remain unchanged since the trusted measurements.

Pending articles still undergo worker identity and qualification checks and
transactional content, batch membership, table policy and mention validation.
The command fails if supplied totals disagree with the observed end of pending
work. This faster retry is not a substitute for a full publication audit.


## Bounded preparation tuning

Worker startup can enable `--prepare-batch-documents 32` with a128-document
request limit. One CPU thread prepares only the next chunk while the original
GPU owner infers the current chunk. Cancellation retains the inference slot
until both inference and prefetched preparation finish; shutdown closes the
model afterward. Default0 retains sequential document preparation. Tokenizer
processes remain separately bounded by `--tokenizer-workers`.

The coordinator sends work to the endpoint that becomes available; a slower
worker does not prevent a faster endpoint from receiving its next batch. Raw
request sizing encodes each document once, accounts exactly for JSON commas and
headers, and still rejects oversized inputs without truncation. Neither tuning
changes the v1 request schema or permits per-request model settings.


## Coordinator dispatch and write batching

Remote dispatch runs independently of the caller's database writes. A single
scheduler owns the reader callback (which must create its SQLite connections in
that thread), maintains at most one submitted request per endpoint and delivers
through a queue bounded by endpoint count, plus one result at the handoff. Slow
writers apply backpressure. Closing the iterator stops dispatch, drains in-flight
calls and joins its threads before releasing pipeline locks. No worker changes
or additional model processes are needed.

`pipeline ner index --write-batch-articles 512` optionally combines or splits
validated result batches into transactions of at most 512 articles. The default
zero retains each inference result's commit boundary. This limit is independent
of HTTP document/byte limits and the `--batch-articles` reader page size. The
maximum is 8192 articles. One writer still owns SQLite; source, membership,
table-policy, span and retry-conflict checks remain in place. Completion and
mentions commit atomically; progress counts advance only afterward. A failure
discards the current uncommitted group; restart skips durable prior groups.
Grouped transactions reduce commit frequency without weakening durability or
changing stored predictions, model identity or worker qualification.


## Explicitly trusted publication retry

After a full audit passed NER, semantic and entities but failed only lexical
checks, publication can accept `--trust-passed-stages-from <audit.json>`.
This is an operator assertion that the passed outputs have not changed since
that report. Old reports have no database snapshot watermarks; identity/epoch
checks cannot prove the absence of direct edits. Never use this after restoring
or modifying those outputs. Full validation remains the default.

The retry retains writer exclusion, repeats ingestion and strict surface/lemma
validation using the active table policy, and checks matching corpus/batch/run,
epoch, complete scope, dictionary generation/manifest/current advancement binding,
semantic store/shard-ledger identity and the serving FAISS checksum. It reuses
only successful NER/semantic/entity stages. It skips their bulk row, membership
and vector-file scans. A historical-only report or incompatible/incomplete
receipt is rejected. The final report records the receipt SHA-256, reused stages
and trust assumption; the serving pointer still changes only after the gate.
