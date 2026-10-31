-- Missing batch policy means legacy, unfiltered derived text.
CREATE TABLE pipeline_batch_table_policies (
    batch_id TEXT PRIMARY KEY REFERENCES pipeline_batches(batch_id),
    policy TEXT NOT NULL CHECK (policy = 'exclude_html_tables_v1'),
    state TEXT NOT NULL CHECK (state IN ('repairing', 'active')),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A repair commits chunks and lexical changes for each article atomically.
CREATE TABLE pipeline_article_table_repairs (
    article_id TEXT PRIMARY KEY REFERENCES articles(article_id),
    batch_id TEXT NOT NULL REFERENCES pipeline_batches(batch_id),
    policy TEXT NOT NULL CHECK (policy = 'exclude_html_tables_v1'),
    previous_chunks INTEGER NOT NULL,
    replacement_chunks INTEGER NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX pipeline_article_table_repairs_batch ON pipeline_article_table_repairs(batch_id);
