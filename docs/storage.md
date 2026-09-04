# Storage and migrations

`Database.initialize()` applies the packaged SQLite migrations. It is safe to
call when opening either a new or existing Semora database.

```python
from semora import Article, Database, Newspaper, Run

database = Database("documents.sqlite")
try:
    database.initialize()
    database.runs.add(Run(run_id="import-1", run_type="import"))
    database.documents.add_newspaper(
        Newspaper(
            newspaper_id="issue-1",
            run_id="import-1",
            content="Full source document",
        )
    )
    database.documents.add_article(
        Article(
            article_id="article-1",
            run_id="import-1",
            newspaper_id="issue-1",
            title="Example",
            content="Article text",
        )
    )
finally:
    database.close()
```

The domain repositories cover common operations. Advanced batch and analytical
queries remain available directly on `Database`.

Newspaper records may include a POSIX-style path relative to `corpus/`.
Articles and chunks may include `char_start`, `char_end`, `line_start`, and
`line_end` positions in the original Markdown issue. Character offsets are
zero-based and end-exclusive; line numbers are one-based and inclusive. These
fields let search clients retrieve nearby source context without reconstructing
positions from cleaned text.
