-- A historical NER run enters additive processing only after a full read-only
-- coverage audit. Keep the evidence and stable model identity separate from
-- runtime tuning in ner_runs.config_json.
CREATE TABLE pipeline_ner_adoptions (
    ner_run_id TEXT PRIMARY KEY REFERENCES ner_runs(ner_run_id),
    evidence_json TEXT NOT NULL,
    model_identity_json TEXT NOT NULL,
    adopted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
