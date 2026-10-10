-- 0002: the approval store — analyses and underwriter decisions.
-- The payload is the full record (result + submission + decision
-- fields) as JSONB; the columns exist for identity, ordering, and the
-- decided filter. `seq` preserves insertion order (a re-save of a
-- decided record updates in place and keeps its position), matching
-- the ordering the history/feedback reads rely on.
CREATE TABLE IF NOT EXISTS approvals (
    submission_id TEXT PRIMARY KEY,
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    payload JSONB NOT NULL,
    decided BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
