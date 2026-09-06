ALTER TABLE chunk_fts_map
ADD COLUMN article_id TEXT;

UPDATE chunk_fts_map
SET article_id = (
    SELECT chunks.article_id
    FROM chunks
    WHERE chunks.chunk_id = chunk_fts_map.chunk_id
);

CREATE INDEX idx_chunk_fts_map_article_id
ON chunk_fts_map(article_id);

INSERT OR IGNORE INTO lemma_index_articles (article_id)
SELECT DISTINCT article_id
FROM chunk_fts_map
WHERE article_id IS NOT NULL;
