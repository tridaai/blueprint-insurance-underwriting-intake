"""Round 14: acknowledged but still waiting.

Round 13 gave the exhaustion worklist a worked state: an operator
acknowledges the notice, the case leaves the "nobody was told"
count. But an acknowledgement closes the telling gap, not the
case — the operator made the call, and the file went back to
waiting for a decision, where the worked state could become a
quieter place to rot. Now a notice acknowledged a
follow-through-escalation window ago (the sibling knob, default
14 days) whose case is still undecided resurfaces on the digest
as its own section and headline line: the call happened; the
file has not moved. A case decided in the meantime drops off —
the decision is the movement the line was asking for.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import ops_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _attempt(at: datetime) -> dict:
    return {
        "attempt": 1,
        "attempted_at": at.isoformat(timespec="seconds"),
        "status": "failed",
        "http_status": 500,
        "signature_id": None,
        "error": "connection refused",
        "backoff_seconds": 0,
    }


def _breach_record(
    sid: str, *, status: str = "awaiting_underwriter"
) -> ApprovalRecord:
    observed = NOW - timedelta(days=30)
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-09-20",
        },
        sla_breach_notice={
            "first_observed_at": observed.isoformat(timespec="seconds"),
            "submission_id": sid,
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "outcome": "borderline",
            "risk_tier": "moderate",
            "sla": {"budget_days": 5, "age_days": 30, "days_over": 25},
            "dispatch_status": "failed",
        },
        sla_breach_attempts=[
            _attempt(observed + timedelta(hours=i)) for i in range(8)
        ],
    )


def _acknowledge(record: ApprovalRecord, *, days_ago: int, by: str = "Ops Lead"):
    record.breach_notice_acknowledgement = {
        "acknowledged_by": by,
        "acknowledged_at": (NOW - timedelta(days=days_ago)).isoformat(
            timespec="seconds"
        ),
        "note": "Called the broker",
    }


def test_an_old_acknowledgement_on_an_undecided_case_resurfaces():
    record = _breach_record("BR-OLD")
    _acknowledge(record, days_ago=20)
    digest = ops_digest([record], [], now=NOW)

    assert digest["counts"]["exhausted_breach_notices"] == 0
    assert digest["counts"]["acknowledged_still_waiting"] == 1
    entry = digest["acknowledged_still_waiting"][0]
    assert entry["submission_id"] == "BR-OLD"
    assert entry["acknowledged_age_days"] == 20
    assert entry["acknowledgement"]["acknowledged_by"] == "Ops Lead"
    assert "the call happened; the file has not moved" in digest["headline"]
    # The case is counted on purpose twice over: the breached
    # queue asks who picks it up, this surface asks why the
    # decision still has not landed.
    assert digest["counts"]["total_attention"] == (
        digest["counts"]["breached_queue"]
        + digest["counts"]["acknowledged_still_waiting"]
    )


def test_a_recent_acknowledgement_stays_quiet():
    record = _breach_record("BR-NEW")
    _acknowledge(record, days_ago=3)
    digest = ops_digest([record], [], now=NOW)
    assert digest["counts"]["acknowledged_still_waiting"] == 0
    assert digest["acknowledged_still_waiting"] == []
    # Still readable on the acknowledged list — quiet is not gone.
    assert digest["acknowledged_breach_notices"][0]["submission_id"] == "BR-NEW"


def test_the_window_is_the_follow_through_escalation_window():
    record = _breach_record("BR-1")
    _acknowledge(record, days_ago=10)
    # Default window (14d): not yet.
    assert ops_digest([record], [], now=NOW)["acknowledged_wait_days"] == 14
    assert (
        ops_digest([record], [], now=NOW)["counts"]["acknowledged_still_waiting"]
        == 0
    )
    # The same window the follow-through escalation reads: pass it
    # through and the notice resurfaces at 10 days.
    digest = ops_digest([record], [], now=NOW, follow_escalation_days=7)
    assert digest["acknowledged_wait_days"] == 7
    assert digest["counts"]["acknowledged_still_waiting"] == 1
    # Or name the window directly.
    digest = ops_digest([record], [], now=NOW, acknowledged_wait_days=30)
    assert digest["counts"]["acknowledged_still_waiting"] == 0


def test_a_decided_case_drops_off_however_old_the_acknowledgement():
    record = _breach_record("BR-DONE", status="approved")
    _acknowledge(record, days_ago=40)
    digest = ops_digest([record], [], now=NOW)
    assert digest["counts"]["acknowledged_still_waiting"] == 0
    assert digest["acknowledged_breach_notices"] == []


def test_oldest_acknowledgement_first():
    older = _breach_record("BR-B")
    _acknowledge(older, days_ago=30)
    newer = _breach_record("BR-A")
    _acknowledge(newer, days_ago=15)
    digest = ops_digest([newer, older], [], now=NOW)
    assert [
        e["submission_id"] for e in digest["acknowledged_still_waiting"]
    ] == ["BR-B", "BR-A"]


def test_the_snapshot_keeps_the_new_count_and_section():
    record = _breach_record("BR-OLD")
    # The service reads the wall clock, so this acknowledgement is
    # dated against it, not the fixtures' fixed NOW.
    record.breach_notice_acknowledgement = {
        "acknowledged_by": "Ops Lead",
        "acknowledged_at": (
            datetime.now(timezone.utc) - timedelta(days=20)
        ).isoformat(timespec="seconds"),
        "note": "Called the broker",
    }
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service._get_store().save(record)
    snapshot = service.snapshot_ops_digest("Ops Lead")
    assert snapshot["counts"]["acknowledged_still_waiting"] == 1
    assert snapshot["sections"]["acknowledged_still_waiting"] == ["BR-OLD"]
    # And the next digest reads movement against it by name.
    digest = service.ops_digest()
    assert digest["movement"]["counts"]["acknowledged_still_waiting"][
        "direction"
    ] == "flat"
