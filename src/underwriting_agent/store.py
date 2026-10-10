"""Approval store: where analyses and underwriter decisions live.

PostgreSQL is the system of record: with ``DATABASE_URL`` set, records
live in the ``approvals`` table (migration ``0002``) via
:class:`PostgresStore`, so an underwriter's decision — and the
diagnosis memory built from earlier submissions — survives restarts
and is shared across service replicas. The SQLite and in-memory
implementations below remain as **test doubles only**: the hermetic
test suite injects them explicitly, and a run without ``DATABASE_URL``
falls back to the in-memory double (nothing persists — that mode is
for tests and offline smoke runs, not a runtime story).

The store is also the memory source: ``prior_submissions`` feeds the
diagnosis evidence (same insured, same broker) and the feedback loop
(decisions with their reasons teach later runs).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
    submission: dict = field(default_factory=dict)
    approved: bool = False
    approver: str = ""
    approve_reason: str = ""
    rejected_by: str = ""
    reject_reason: str = ""
    dispatch_status: str | None = None
    # The webhook delivery ledger: one dict per dispatch attempt
    # (see schemas.DispatchAttempt). Older stored payloads predate
    # it and load with an empty ledger.
    dispatch_attempts: list[dict] = field(default_factory=list)
    # The drafted queue status note for an SLA-breached case (see
    # service.draft_status_note): {text, drafted_at, sla, validation,
    # outstanding} or None when none has been drafted — plus, when a
    # wording-only failure was steered into a passing redraft,
    # {repair_attempted, repaired, original_validation,
    # repair_guidance} recording the repair. ``outstanding``
    # snapshots what the case was waiting on; a later resubmission
    # from the same insured adds a ``follow_through`` marking to the
    # note (see service._record_note_follow_through) without ever
    # rewriting the note itself. When the note's loop escalates, an
    # underwriter's worked touch is appended to the note as an
    # ``escalation_touches`` entry (see
    # service.mark_escalated_note_worked): {worked_by, worked_at,
    # note} — the latest touch quiets the escalation for one
    # window. An internal draft — it is stored on the record and
    # never sent. Older payloads predate it.
    status_note: dict | None = None
    # The SLA-breach webhook notice (see service.notify_sla_breaches):
    # set the first time the queue projection observes this record
    # past its SLA budget AND a breach dispatch was attempted —
    # {first_observed_at, insured_name, broker_name, outcome,
    # risk_tier, sla, dispatch_status}. Its presence is the dedupe:
    # a submission is breach-notified at most once, however often
    # the queue is re-checked. The attempts live in their own ledger
    # below (the same shape as dispatch_attempts, a different event).
    # Older payloads predate both fields.
    sla_breach_notice: dict | None = None
    sla_breach_attempts: list[dict] = field(default_factory=list)
    # The operator acknowledgement of an exhausted breach notice
    # (see service.acknowledge_exhausted_breach_notice):
    # {acknowledged_by, acknowledged_at, note} recorded when a
    # human attests they picked up a case whose breach notice
    # never landed — the call the exhaustion worklist exists to
    # prompt. It marks the notice worked; it does not pretend the
    # webhook delivered, and it never touches the attempts
    # ledger. Older payloads predate it.
    breach_notice_acknowledgement: dict | None = None
    # The acknowledged-escalation webhook notice (see
    # service.notify_acknowledged_escalations): set the first
    # time an acknowledged-but-still-waiting case is observed
    # past the second rung (twice the follow-through window)
    # AND an escalation dispatch was attempted —
    # {first_observed_at, insured_name, broker_name, risk_tier,
    # acknowledgement, acknowledged_age_days,
    # escalation_window_days, dispatch_status}. Its presence is
    # the dedupe: one firing per acknowledgement episode (an
    # acknowledgement is recorded once, so one episode per
    # record). The attempts live in their own ledger below.
    # Older payloads predate both fields.
    escalation_notice: dict | None = None
    escalation_attempts: list[dict] = field(default_factory=list)


def history_entry(record: ApprovalRecord) -> dict:
    """The compact history projection of one analysed submission.

    Insured/broker/line/class come from the stored submission payload
    (the result object itself carries only insured + line)."""
    result = record.result
    sub = record.submission or {}
    return {
        "submission_id": result.submission_id,
        "insured_name": result.insured_name or sub.get("insured_name", ""),
        "broker_name": sub.get("broker_name", ""),
        "line": result.line_of_business or sub.get("line_of_business", ""),
        "class_of_business": sub.get("class_of_business", ""),
        "outcome": result.triage.outcome,
        "risk_tier": result.triage.risk_tier,
    }


def feedback_entry(record: ApprovalRecord) -> dict | None:
    """A decided record as a feedback entry, or None when undecided
    (or decided without a reason — feedback without a reason teaches
    nothing)."""
    result = record.result
    if result.status not in ("approved", "rejected") or not result.decision_reason:
        return None
    sub = record.submission or {}
    return {
        "submission_id": result.submission_id,
        "insured_name": result.insured_name or sub.get("insured_name", ""),
        "broker_name": sub.get("broker_name", ""),
        "decision": result.status,
        "reason": result.decision_reason,
        "decided_at": result.decided_at,
    }


def broker_summary(priors: list[dict], broker: str) -> dict:
    """Outcome counts for one broker across prior submissions."""
    hits = [e for e in priors if broker and e.get("broker_name") == broker]
    counts: dict[str, int] = {}
    for entry in hits:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    return {
        "broker": broker,
        "broker_count": len(hits),
        "broker_outcome_counts": counts,
    }


def format_outcome_counts(counts: dict[str, int]) -> str:
    """{'in_appetite': 2, 'borderline': 1} -> '2 in_appetite, 1 borderline'."""
    return ", ".join(f"{n} {outcome}" for outcome, n in sorted(counts.items())) or "none"


def _record_to_dict(record: ApprovalRecord) -> dict:
    return {
        "result": record.result.model_dump(mode="json"),
        "submission": record.submission,
        "approved": record.approved,
        "approver": record.approver,
        "approve_reason": record.approve_reason,
        "rejected_by": record.rejected_by,
        "reject_reason": record.reject_reason,
        "dispatch_status": record.dispatch_status,
        "dispatch_attempts": record.dispatch_attempts,
        "status_note": record.status_note,
        "sla_breach_notice": record.sla_breach_notice,
        "sla_breach_attempts": record.sla_breach_attempts,
        "breach_notice_acknowledgement": record.breach_notice_acknowledgement,
        "escalation_notice": record.escalation_notice,
        "escalation_attempts": record.escalation_attempts,
    }


def _dump_record(record: ApprovalRecord) -> str:
    return json.dumps(_record_to_dict(record))


def _record_from_dict(data: dict) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult.model_validate(data["result"]),
        submission=data.get("submission", {}),
        approved=data.get("approved", False),
        approver=data.get("approver", ""),
        approve_reason=data.get("approve_reason", ""),
        rejected_by=data.get("rejected_by", ""),
        reject_reason=data.get("reject_reason", ""),
        dispatch_status=data.get("dispatch_status"),
        dispatch_attempts=data.get("dispatch_attempts") or [],
        status_note=data.get("status_note"),
        sla_breach_notice=data.get("sla_breach_notice"),
        sla_breach_attempts=data.get("sla_breach_attempts") or [],
        breach_notice_acknowledgement=data.get("breach_notice_acknowledgement"),
        escalation_notice=data.get("escalation_notice"),
        escalation_attempts=data.get("escalation_attempts") or [],
    )


def _load_record(payload: str) -> ApprovalRecord:
    return _record_from_dict(json.loads(payload))


class InMemoryStore:
    """Process-local store (tests, ephemeral runs). Thread-safe."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._documents: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[record.result.submission_id] = record

    def get_document(self, key: str) -> dict | None:
        with self._lock:
            return self._documents.get(key)

    def save_document(self, key: str, payload: dict) -> None:
        with self._lock:
            self._documents[key] = payload

    def get(self, submission_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(submission_id)

    def _all(self) -> list[ApprovalRecord]:
        with self._lock:
            return list(self._records.values())

    def all_records(self) -> list[ApprovalRecord]:
        return self._all()

    def prior_submissions(self, exclude_submission_id: str) -> list[dict]:
        return [
            history_entry(r)
            for r in self._all()
            if r.result.submission_id != exclude_submission_id
        ]

    def decision_feedback(self, exclude_submission_id: str) -> list[dict]:
        entries = []
        for record in reversed(self._all()):
            if record.result.submission_id == exclude_submission_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    submission_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    decided INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS documents (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


class PostgresStore:
    """PostgreSQL-backed store — the production system of record.

    One short-lived psycopg connection per call (the same shape as
    the SQLite double), payload as JSONB, ordering by the identity
    ``seq`` column so history reads match insertion order. Schema is
    owned by the migrations (``db.ensure_migrated`` runs them before
    the first query); this class never issues DDL.
    """

    def __init__(self, url: str) -> None:
        from .db import ensure_migrated

        self._url = url
        ensure_migrated(url)

    def _connect(self):
        from .db import connect

        return connect(self._url)

    def save(self, record: ApprovalRecord) -> None:
        from psycopg.types.json import Jsonb

        decided = record.result.status in ("approved", "rejected")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (submission_id, payload, decided) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (submission_id) DO UPDATE SET "
                "payload = EXCLUDED.payload, decided = EXCLUDED.decided, "
                "updated_at = now()",
                (
                    record.result.submission_id,
                    Jsonb(_record_to_dict(record)),
                    decided,
                ),
            )
            conn.commit()

    def get(self, submission_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM approvals WHERE submission_id = %s",
                (submission_id,),
            ).fetchone()
        return _record_from_dict(row[0]) if row else None

    def get_document(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM service_documents WHERE key = %s",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def save_document(self, key: str, payload: dict) -> None:
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO service_documents (key, payload) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET "
                "payload = EXCLUDED.payload, updated_at = now()",
                (key, Jsonb(payload)),
            )
            conn.commit()

    def _all_records(self) -> list[ApprovalRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM approvals ORDER BY seq"
            ).fetchall()
        return [_record_from_dict(row[0]) for row in rows]

    def all_records(self) -> list[ApprovalRecord]:
        return self._all_records()

    def prior_submissions(self, exclude_submission_id: str) -> list[dict]:
        return [
            history_entry(record)
            for record in self._all_records()
            if record.result.submission_id != exclude_submission_id
        ]

    def decision_feedback(self, exclude_submission_id: str) -> list[dict]:
        entries = []
        for record in reversed(self._all_records()):
            if record.result.submission_id == exclude_submission_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


class SQLiteStore:
    """SQLite-backed store — a **test double** (the persistence tests
    exercise the store contract against it; production runs use
    :class:`PostgresStore`). One short-lived connection per call, WAL
    journaling, so concurrent batch writers serialise cleanly."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, record: ApprovalRecord) -> None:
        decided = 1 if record.result.status in ("approved", "rejected") else 0
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (submission_id, payload, decided) VALUES (?, ?, ?) "
                "ON CONFLICT(submission_id) DO UPDATE SET payload=excluded.payload, "
                "decided=excluded.decided",
                (record.result.submission_id, _dump_record(record), decided),
            )
            conn.commit()

    def get(self, submission_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM approvals WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        return _load_record(row[0]) if row else None

    def get_document(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM documents WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_document(self, key: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (key, payload) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                (key, json.dumps(payload)),
            )
            conn.commit()

    def _all_payloads(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM approvals ORDER BY rowid"
            ).fetchall()
        return [row[0] for row in rows]

    def all_records(self) -> list[ApprovalRecord]:
        return [_load_record(payload) for payload in self._all_payloads()]

    def prior_submissions(self, exclude_submission_id: str) -> list[dict]:
        return [
            history_entry(_load_record(payload))
            for payload in self._all_payloads()
            if _load_record(payload).result.submission_id != exclude_submission_id
        ]

    def decision_feedback(self, exclude_submission_id: str) -> list[dict]:
        entries = []
        for payload in reversed(self._all_payloads()):
            record = _load_record(payload)
            if record.result.submission_id == exclude_submission_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries


def default_store() -> PostgresStore | InMemoryStore:
    """The store from the environment.

    ``DATABASE_URL`` set → :class:`PostgresStore` (the production
    system of record). Unset → :class:`InMemoryStore`, the test
    double: analyses and decisions live for the process only. There
    is deliberately no file-backed default any more — local files
    are not a runtime story (see ``docs/architecture.md``).
    """
    from .db import database_url

    url = database_url()
    if url:
        return PostgresStore(url)
    return InMemoryStore()
