-- There is one compatible configuration per lexical stage. Keep the lookup
-- bounded even after adopting millions of historical completion records.
CREATE TABLE IF NOT EXISTS pipeline_lexical_configs (
    stage TEXT PRIMARY KEY CHECK (stage IN ('surface', 'lemma')),
    config_json TEXT NOT NULL,
    UNIQUE (stage, config_json)
);

CREATE TABLE IF NOT EXISTS pipeline_lexical_state (
    article_id TEXT NOT NULL REFERENCES articles(article_id),
    stage TEXT NOT NULL CHECK (stage IN ('surface', 'lemma')),
    input_sha256 TEXT NOT NULL,
    config_json TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stage, article_id),
    FOREIGN KEY (stage, config_json) REFERENCES pipeline_lexical_configs(stage, config_json)
);

CREATE TABLE IF NOT EXISTS pipeline_baselines (
    baseline_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state = 'articles_adopted'),
    adopted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
