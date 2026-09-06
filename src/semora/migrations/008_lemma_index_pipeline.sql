DROP INDEX IF EXISTS idx_chunks_article_id;

CREATE INDEX idx_chunks_article_chunk
ON chunks(article_id, chunk_index);
