ALTER TABLE embeddings
ADD COLUMN is_valid INTEGER NOT NULL DEFAULT 0 CHECK (is_valid IN (0, 1));

-- Replace single-column indexes with composites matching the actual read paths.
DROP INDEX IF EXISTS idx_chunks_chunking_run_id;
CREATE INDEX idx_chunks_chunking_run_article
ON chunks(chunking_run_id, article_id, chunk_index);

DROP INDEX IF EXISTS idx_embeddings_embedding_run_id;
CREATE INDEX idx_embeddings_run_chunk
ON embeddings(embedding_run_id, chunk_id);

-- Keep the hot valid-only indexes small. SQLite can use these only for queries
-- which explicitly contain is_valid = 1.
CREATE INDEX idx_embeddings_valid_run_embedding
ON embeddings(embedding_run_id, embedding_id)
WHERE is_valid = 1;

CREATE INDEX idx_embeddings_valid_chunk
ON embeddings(chunk_id)
WHERE is_valid = 1;

CREATE INDEX idx_articles_valid_article
ON articles(article_id)
WHERE is_valid = 1;

CREATE INDEX idx_embedding_runs_model
ON embedding_runs(model_id, embedding_run_id);

CREATE INDEX idx_embedding_runs_model_chunking
ON embedding_runs(
    model_id,
    json_extract(config_json, '$.chunking_run_id'),
    embedding_run_id
);
