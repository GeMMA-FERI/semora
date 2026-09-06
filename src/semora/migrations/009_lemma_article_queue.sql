CREATE TABLE lemma_index_articles (
    article_id TEXT PRIMARY KEY,
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
) WITHOUT ROWID;
