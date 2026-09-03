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
