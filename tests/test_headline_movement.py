"""Round 14: the digest headline speaks membership, not just counts.

Round 12 made the trend-aware headline speak the counts ("breached
queue cases up 1"), but the membership behind the movement — which
cases entered, which cleared — still read as data on the console.
An operator works ids, not arithmetic, and the headline is the one
surface every digest reader sees (including any future sent
digest). Now the same line names them: "…— LATE-1 entered breached
queue cases; LATE-2 cleared stuck dispatches". A count that sits
flat while its membership churns says so, instead of reporting no
movement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import build_digest_snapshot, ops_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _breached_record(sid: str) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
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


def _exhausted_record(sid: str) -> ApprovalRecord:
    record = _breached_record(sid)
    observed = NOW - timedelta(days=8)
    record.sla_breach_notice = {
        "first_observed_at": observed.isoformat(timespec="seconds"),
        "submission_id": sid,
        "insured_name": "Late Co",
        "broker_name": "Broker Q",
        "outcome": "borderline",
        "risk_tier": "moderate",
        "sla": {"budget_days": 5, "age_days": 12, "days_over": 7},
        "dispatch_status": "failed",
    }
    record.sla_breach_attempts = [
        {
            "attempt": i + 1,
            "attempted_at": (observed + timedelta(hours=i)).isoformat(
                timespec="seconds"
            ),
            "status": "failed",
            "http_status": 500,
            "signature_id": None,
            "error": "connection refused",
            "backoff_seconds": 0,
        }
        for i in range(8)
    ]
    return record


def _snapshot_of(records) -> dict:
    return build_digest_snapshot(ops_digest(records, [], now=NOW), "snap-0001", "Ops Lead")


def test_an_entered_case_is_named_in_the_headline():
    snapshot = _snapshot_of([])
    digest = ops_digest(
        [_breached_record("LATE-1")], [], now=NOW, latest_snapshot=snapshot
    )
    assert "breached queue cases up 1" in digest["headline"]
    assert "LATE-1 entered breached queue cases" in digest["headline"]


def test_a_cleared_case_is_named_in_the_headline():
    snapshot = _snapshot_of([_breached_record("LATE-1")])
    digest = ops_digest([], [], now=NOW, latest_snapshot=snapshot)
    assert "breached queue cases down 1" in digest["headline"]
    assert "LATE-1 cleared breached queue cases" in digest["headline"]


def test_flat_counts_with_churned_membership_still_speak():
    snapshot = _snapshot_of([_breached_record("LATE-1")])
    digest = ops_digest(
        [_breached_record("LATE-2")], [], now=NOW, latest_snapshot=snapshot
    )
    # The count never moved; the case behind it changed entirely.
    assert digest["movement"]["counts"]["breached_queue"]["change"] == 0
    assert "no movement in the headline counts" in digest["headline"]
    assert "LATE-2 entered breached queue cases" in digest["headline"]
    assert "LATE-1 cleared breached queue cases" in digest["headline"]


def test_membership_across_sections_is_named_section_by_section():
    snapshot = _snapshot_of([])
    digest = ops_digest(
        [_exhausted_record("BR-9")], [], now=NOW, latest_snapshot=snapshot
    )
    assert "BR-9 entered exhausted breach notices" in digest["headline"]
    assert "BR-9 entered breached queue cases" in digest["headline"]


def test_no_movement_anywhere_keeps_the_plain_statement():
    records = [_breached_record("LATE-1")]
    snapshot = _snapshot_of(records)
    digest = ops_digest(records, [], now=NOW, latest_snapshot=snapshot)
    assert digest["headline"].endswith(
        "Since snap-0001: no movement in the headline counts."
    )
