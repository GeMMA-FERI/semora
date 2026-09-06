CREATE TABLE article_lemmas (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    pipeline_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
);

-- Existing contentless FTS rows cannot be used to reconstruct lemma text.
-- Reset only the lemma index so the materialized rows and FTS stay aligned.
INSERT INTO article_lemma_fts(article_lemma_fts) VALUES('delete-all');
DELETE FROM article_lemma_index_state;
