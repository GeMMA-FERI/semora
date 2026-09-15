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
