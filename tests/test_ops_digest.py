"""Round 10: the ops digest.

The surfaces that rot quietly each had their own screen: stuck
webhook dispatches on the dispatch view, breached cases on the
queue view, unanswered status notes in the inbox digest, expired
calibration proposals in the proposals list. The ops digest is
the one composed read an operator starts from — computed, never
sent, and reading it changes nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import ops_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _attempt(at: datetime) -> dict:
    return {
        "attempt": 1,
        "attempted_at": at.isoformat(timespec="seconds"),
        "status": "failed",
        "http_status": None,
        "signature_id": None,
        "error": "connection refused",
        "backoff_seconds": 0,
    }


def _stuck_record() -> ApprovalRecord:
    # First failure 3 days ago, attempts left under the budget: stuck.
    return ApprovalRecord(
        result=AgentResult(
            submission_id="STUCK-1",
            status="approved",
            insured_name="Stuck Co",
            triage=Triage(outcome="in_appetite", risk_tier="low"),
        ),
        submission={"insured_name": "Stuck Co", "broker_name": "Broker Q"},
        dispatch_status="failed",
        dispatch_attempts=[_attempt(NOW - timedelta(days=3))],
    )


def _breached_record() -> ApprovalRecord:
    # Moderate tier: 5-day budget; submitted long before NOW (and
    # before any plausible wall-clock "today" the service reads).
    return ApprovalRecord(
        result=AgentResult(
            submission_id="LATE-1",
            status="awaiting_underwriter",
            insured_name="Late Co",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": "Late Co",
            "broker_name": "Broker Q",
            "submitted_at": "2026-08-01",
        },
    )


def _note_record() -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id="NOTE-1",
            status="awaiting_underwriter",
            insured_name="Note Co",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": "Note Co",
            "broker_name": "Broker Q",
            "submitted_at": "2026-10-19",
        },
        status_note={
            "submission_id": "NOTE-1",
            "drafted_at": "2026-09-25T09:00:00+00:00",
            "text": "INTERNAL DRAFT …",
            "outstanding": {
                "missing_elements": ["loss runs"],
                "gap_sources": [],
                "blocking_lines": [],
            },
        },
    )


def _proposal(pid: str, *, expired: bool, days_left: int) -> dict:
    expires = NOW + timedelta(days=days_left)
    return {
        "proposal_id": pid,
        "kind": "diff_trigger_threshold",
        "dimension": "limit_jump_pct",
        "current_value": 50.0,
        "proposed_value": 40.0,
        "status": "draft",
        "expires_at": expires.isoformat(timespec="seconds"),
        "expired": expired,
    }


def test_empty_store_digest_says_nothing_needs_attention():
    digest = ops_digest([], [], now=NOW)
    assert digest["counts"]["total_attention"] == 0
    assert "nothing needs attention" in digest["headline"]
    assert digest["stuck_dispatches"] == []
    assert digest["breached_queue"] == []


def test_digest_composes_all_four_surfaces():
    records = [_stuck_record(), _breached_record(), _note_record()]
    proposals = [
        _proposal("PROP-SOON", expired=False, days_left=2),
        _proposal("PROP-FAR", expired=False, days_left=10),
        _proposal("PROP-DEAD", expired=True, days_left=-4),
        # Decided proposals are history, never digest material.
        {**_proposal("PROP-DONE", expired=False, days_left=1), "status": "accepted"},
    ]
    digest = ops_digest(records, proposals, now=NOW)
    counts = digest["counts"]
    assert counts["stuck_dispatches"] == 1
    assert counts["breached_queue"] == 1
    assert counts["follow_through_outstanding"] == 1
    # The note is 25 days old — past the 14-day escalation age.
    assert counts["follow_through_escalated"] == 1
    assert counts["proposals_expiring_soon"] == 1
    assert counts["proposals_expired"] == 1
    assert [e["submission_id"] for e in digest["stuck_dispatches"]] == ["STUCK-1"]
    assert [e["submission_id"] for e in digest["breached_queue"]] == ["LATE-1"]
    assert digest["follow_through_escalated"][0]["submission_id"] == "NOTE-1"
    assert digest["proposals_expiring_soon"][0]["proposal_id"] == "PROP-SOON"
    assert digest["proposals_expiring_soon"][0]["days_until_expiry"] == 2
    assert digest["proposals_expired"][0]["proposal_id"] == "PROP-DEAD"
    assert "1 stuck dispatch" in digest["headline"]
    assert "1 breached queue case" in digest["headline"]
    assert "1 escalated follow-through" in digest["headline"]


def test_digest_warning_window_is_configurable():
    proposals = [_proposal("PROP-FAR", expired=False, days_left=10)]
    digest = ops_digest([], proposals, now=NOW)
    assert digest["counts"]["proposals_expiring_soon"] == 0
    digest = ops_digest([], proposals, now=NOW, proposal_warning_days=14)
    assert digest["counts"]["proposals_expiring_soon"] == 1
    assert digest["proposal_warning_days"] == 14


def test_service_digest_measures_breaches_against_budgets_in_force():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    digest = service.ops_digest()
    assert digest["counts"]["total_attention"] == 0
    store = service._get_store()
    store.save(_breached_record())
    digest = service.ops_digest()
    assert digest["counts"]["breached_queue"] == 1
