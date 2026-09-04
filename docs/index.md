# Semora Documentation

Semora is organized around six focused domains:

- `semora.corpus`: source-aware newspaper ingestion and token chunking
- `semora.storage`: SQLite records, migrations, repositories, and queries
- `semora.text`: Markdown cleanup, chunk identifiers, and chunking strategies
- `semora.embeddings`: backend interfaces, batching, I/O, and serialization
- `semora.projection`: dimensionality reduction for stored embeddings
- `semora.retrieval`: BM25, regular-expression, and dense search

## Guides

- [Installation](./installation.md)
- [Corpus indexing and search](./corpus-search.md)
- [Storage and migrations](./storage.md)
- [Text chunking](./text.md)
- [Embedding backends](./embeddings.md)
- [Command-line tools](./cli.md)
