"""Approval store: where analyses and underwriter decisions live.

SQLite on disk by default (``STATE_DB_PATH``, defaulting to a
git-ignored ``.data/`` file next to the repo), so an underwriter's
decision — and the diagnosis memory built from earlier submissions —
survives a restart. ``InMemoryStore`` remains for tests and ephemeral
runs. Both implement the same ``Store`` protocol (``ports.py``).

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

from .config import load_dotenv, env_str
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


def _dump_record(record: ApprovalRecord) -> str:
    return json.dumps(
        {
            "result": record.result.model_dump(mode="json"),
            "submission": record.submission,
            "approved": record.approved,
            "approver": record.approver,
            "approve_reason": record.approve_reason,
            "rejected_by": record.rejected_by,
            "reject_reason": record.reject_reason,
            "dispatch_status": record.dispatch_status,
        }
    )


def _load_record(payload: str) -> ApprovalRecord:
    data = json.loads(payload)
    return ApprovalRecord(
        result=AgentResult.model_validate(data["result"]),
        submission=data.get("submission", {}),
        approved=data.get("approved", False),
        approver=data.get("approver", ""),
        approve_reason=data.get("approve_reason", ""),
        rejected_by=data.get("rejected_by", ""),
        reject_reason=data.get("reject_reason", ""),
        dispatch_status=data.get("dispatch_status"),
    )


class InMemoryStore:
    """Process-local store (tests, ephemeral runs). Thread-safe."""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[record.result.submission_id] = record

    def get(self, submission_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(submission_id)

    def _all(self) -> list[ApprovalRecord]:
        with self._lock:
            return list(self._records.values())

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
"""


class SQLiteStore:
    """SQLite-backed store: one short-lived connection per call, WAL
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

    def _all_payloads(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM approvals ORDER BY rowid"
            ).fetchall()
        return [row[0] for row in rows]

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


def default_store_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".data" / "underwriting-agent.sqlite3"


def default_store() -> InMemoryStore | SQLiteStore:
    """The store from the environment: ``STATE_DB_PATH`` when set,
    else the default on-disk location; ``STATE_DB_PATH=:memory:``
    selects the in-memory store."""
    load_dotenv()
    configured = env_str("STATE_DB_PATH")
    if configured == ":memory:":
        return InMemoryStore()
    return SQLiteStore(configured if configured else default_store_path())
