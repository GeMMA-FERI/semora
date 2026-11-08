# Entity-name discovery and article lookup

After `index ner`, build a separate name dictionary. Labels are ignored: names
are grouped by Unicode NFC, case folding and whitespace normalization. Original
spelling is retained for display; different inflections remain separate names.

## Windows setup

Use a trusted local spellfix library. Installation validates it in an in-memory
SQLite connection and copies it under the corpus root; it never downloads code
automatically. Optionally provide `--sha256` to verify the binary before loading.

```powershell
$py = "C:\GitHub\nora\.venv\Scripts\python.exe"
& $py -m semora.cli.main --root C:\GitHub\nora extensions install-spellfix `
  --source C:\GitHub\aidhch\geo-data\spellfix1.dll

& $py -m semora.cli.main --root C:\GitHub\nora index entities --run-id classla-sl
```

The installer reports the SHA-256 for reproducibility. `SEMORA_SPELLFIX_PATH`
overrides the default `ROOT/extensions/spellfix1.dll` (Linux: `spellfix1.so`).
Native extensions execute code: only install binaries you trust, matching the
platform/architecture. The old geo-data compiler script can produce the Windows
DLL; pin its source revision and headers before building release binaries.
Linux requires a Linux build, not the Windows DLL.

## Storage and rebuilding

`indexes/entities/manifest.json` selects an immutable generation SQLite file.
The dictionary contains distinct names, article counts, a spellfix virtual table,
and one mapping per name/article with its earliest mention offset and content hash.
Indexing streams batches from one read snapshot, writes a new generation, and
publishes the manifest only after success. It does not modify the corpus database.
Indexing is a full rebuild, not a resumable incremental process.

Finish NER first to avoid holding a long read snapshot while it writes. Rebuild
after more extraction; restart the gateway/stdio reader to pick up the new
generation. Existing readers retain their generation and remain usable. Old
generation files are deliberately not removed automatically: they consume disk
space and can be removed manually once no reader uses them. Invalidation in the
live corpus prevents changed/deleted articles from using stale mention offsets;
name counts remain snapshot counts until rebuild.

## Search API

`SearchEngine.search_entities(query)` returns at most 100 candidates, with name,
article_count, weighted distance and generation. Only a trailing `*` is accepted;
the non-wildcard query must have 3–200 characters. Spellfix uses weighted spelling
costs and phonetic candidate pruning, not exhaustive unit-cost k-distance search.
The 100 candidates selected by spellfix are displayed in distance order, then
article count and normalized name. `limited=true` means the cap was reached, not
that an exact total count is known.

`SearchEngine.search_entity_articles(entity, limit=20, after=None, ...)` performs
exact normalized-name lookup, optional newspaper/date filtering, and stable
article-ID keyset paging. Pass `next_after` and `generation` back for continuation.
Results include original source positions and short mention-centered snippets.
Articles lacking original-source offsets are excluded. Names are not verified
real-world identities. Both methods are available through stdio:

```json
{"id":1,"op":"search_entities","query":"Marija*"}
{"id":2,"op":"search_entity_articles","entity":"Marija Abram","limit":20}
```

Nora caches the returned 100 discovery candidates and supplies opaque cursors
for pages of at most 20; it does not repeat the fuzzy query for each page.

## Tests

Set `SEMORA_TEST_SPELLFIX` to a trusted DLL/SO and run
`pytest tests/test_entity_search.py`. Tests use temporary databases only and
cover label-independent grouping, candidate limits, article paging/filters,
stale-offset protection, checksum rejection and safe generation replacement.

The Windows implementation was verified with the existing geo-data DLL
(`53153b9fca1c9d65ca829912a13ae6b4795b3c44315c47df1b082deeeb775f8c`).
An actual CLI smoke build indexed 977 extracted mentions into 790 distinct names.
The selected Semora regression suite passed 43 tests; the full Nora agent/gateway
suite passed 105 tests, including five cached pages from one fuzzy query.
No production archive indexing or service startup was performed during these tests.


## Reviewed name groups

`semora.curation` stores preferred names and normalized variant membership in an
independent, persistent SQLite database. The source dictionary and corpus remain
read-only. Initialize it explicitly with the published corpus identity:

```powershell
python -m semora.curation.cli --store indexes/curation/curation.sqlite --corpus-id <corpus-id> init
python -m semora.curation.cli --store indexes/curation/curation.sqlite --corpus-id <corpus-id> list
python -m semora.curation.cli --store indexes/curation/curation.sqlite --corpus-id <corpus-id> mutate --request request.json
```

A mutation envelope contains `action`, `payload`, `operation_id` (UUID), `actor`
(trusted operator identity), and `expected_revision`. Actions are `create_group`,
`update_group`, `delete_group`, `reject_pair`, `reopen_pair`, and `undo_change`.
Group creation/update takes `preferred_name`, 1-100 `members`, and optional
`description`; updates also take `group_id`. Pair operations take `first`,
`second`, `reason`. Undo takes the original `operation_id`; dissolution takes
`group_id`. This CLI is for trusted local operators. AIDHCH's native curation
worker supplies authenticated admin identities for remote operations.

Mutations, scoped audit snapshots and revision advancement commit together.
Retried operation IDs replay their saved results; different requests cannot reuse
an ID. Undo checks only affected records and current membership/rejection
conflicts. One normalized name belongs to at most one group. Group UUIDs and name
keys survive dictionary rebuilds; missing names remain unresolved.

Pass `curation_path=...` to `SearchEngine` to opt into curated discovery and group
lookup. A published corpus is required and must match the store binding. Indexed
preferred exact/prefix matches take precedence; otherwise raw spellfix candidates
map to their approved groups. Exact `entity` lookup remains unchanged. Group
article lookup takes `group_id` instead of `entity` and returns deduplicated source
hits with original `matched_name`, unresolved members and curation revision.
Continuations require the same publication generation and curation revision.

`CuratedRetrieval.count_articles()` computes a separate exact distinct-article
count with live source/hash eligibility, within a 15-minute SQL budget. Save the
result through `CurationStore.save_count()` only after computation succeeds and
the revision is still current. Cached counts are tied to publication and group
membership version. Failed counts remain pending; no approximate sum is returned.
NORA/AIDHCH schedules this work separately after group edits. Admin suggestions
are bounded to five spellfix anchors; evidence validates only the first 40 raw
memberships and reports its nonrepresentative sampling scope.

Back up curation using SQLite's backup API or a consistent stopped-writer copy;
never delete it when rebuilding derived indexes. Focused tests are
`test_curation_store.py`, `test_curation_catalogue.py`, and
`test_curated_retrieval.py`; the last uses the existing trusted spellfix fixture.
