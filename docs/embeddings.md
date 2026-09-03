# Embedding backends

`get_embedder()` selects a backend without loading its model. Call `load()` once
and reuse the resulting object for queries and documents.

```python
from semora.embeddings import get_embedder

embedder = get_embedder("intfloat/multilingual-e5-large").load()
query = embedder.embed_query("example query")
documents = embedder.embed_documents(["first document", "second document"])
```

The asynchronous OpenAI Batch API workflow is implemented separately in
`semora.embeddings.openai_batch` and persists its resumable state in SQLite.
