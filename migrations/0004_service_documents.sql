-- 0004: service documents — small JSON documents the service keeps
-- beside the approval records: the calibration proposals and the
-- accepted diff-trigger threshold settings (see proposals.py).
-- Keyed documents, not records: they carry no submission identity,
-- and their lifecycle lives in the payload itself.
CREATE TABLE IF NOT EXISTS service_documents (
    key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
