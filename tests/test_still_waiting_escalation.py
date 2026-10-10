"""Round 15: still-waiting escalation — the second rung.

Round 14 resurfaced an acknowledged breach notice whose case
still had not moved a follow-through window after the call: the
digest reports it. But reporting is not a nudge, and the
underwriter who finally picked the case up saw none of it — the
queue, their worklist, carried no sign that the broker had
already been called once. Now, at twice the window, the case
joins the queue's escalated surfaces: the item itself carries
the acknowledgement (who called, when, their note), the cases
are listed with it attached, and the digest flags the subset
``escalated`` and counts it — a subset of still-waiting, never
a second helping of total_attention: the same cases, louder.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    ops_digest,
    queue_aging,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

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


# ---------------------------------------------------------------------------
# The digest: the subset is flagged and counted, never double-counted
# ---------------------------------------------------------------------------


def test_past_the_second_window_the_still_waiting_entry_is_escalated():
    record = _breach_record("BR-OLD")
    _acknowledge(record, days_ago=30)  # window 14 → second rung at 28
    digest = ops_digest([record], [], now=NOW)
    assert digest["counts"]["acknowledged_still_waiting"] == 1
    assert digest["counts"]["acknowledged_escalated"] == 1
    entry = digest["acknowledged_still_waiting"][0]
    assert entry["escalated"] is True
    assert "past the second window" in digest["headline"]
    # The escalated subset is the same case, louder — total
    # attention counts it once, under still-waiting.
    assert digest["counts"]["total_attention"] == (
        digest["counts"]["breached_queue"]
        + digest["counts"]["acknowledged_still_waiting"]
    )


def test_between_the_windows_the_entry_waits_unescalated():
    record = _breach_record("BR-MID")
    _acknowledge(record, days_ago=20)  # past 14, short of 28
    digest = ops_digest([record], [], now=NOW)
    assert digest["counts"]["acknowledged_still_waiting"] == 1
    assert digest["counts"]["acknowledged_escalated"] == 0
    assert digest["acknowledged_still_waiting"][0]["escalated"] is False
    assert "second window" not in digest["headline"]


def test_the_second_rung_scales_with_the_window():
    record = _breach_record("BR-FAST")
    _acknowledge(record, days_ago=15)
    digest = ops_digest([record], [], now=NOW, follow_escalation_days=7)
    assert digest["counts"]["acknowledged_escalated"] == 1


def test_the_snapshot_keeps_the_escalated_count():
    record = _breach_record("BR-OLD")
    _acknowledge(record, days_ago=30)
    digest = ops_digest([record], [], now=NOW)
    snapshot = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert snapshot["counts"]["acknowledged_escalated"] == 1


# ---------------------------------------------------------------------------
# The queue: the case joins the escalated surfaces, acknowledgement attached
# ---------------------------------------------------------------------------


def _queue_item(view: dict, sid: str) -> dict:
    for bucket in view["buckets"]:
        for item in bucket["submissions"]:
            if item["submission_id"] == sid:
                return item
    raise AssertionError(f"{sid} not in the queue")


def test_the_queue_item_carries_the_acknowledgement_past_the_second_rung():
    record = _breach_record("BR-OLD")
    _acknowledge(record, days_ago=30)
    view = queue_aging([record], NOW)
    item = _queue_item(view, "BR-OLD")
    escalation = item["acknowledged_escalation"]
    assert escalation["acknowledged_by"] == "Ops Lead"
    assert escalation["note"] == "Called the broker"
    assert escalation["acknowledged_age_days"] == 30
    assert view["items_with_escalated_acknowledgements"] == 1
    listed = view["escalated_acknowledgements"]
    assert [e["submission_id"] for e in listed] == ["BR-OLD"]
    # The acknowledgement itself rides the list entry — the
    # second touch reads the first one's record.
    assert listed[0]["acknowledgement"]["acknowledged_by"] == "Ops Lead"
    assert listed[0]["escalation_days"] == 28


def test_between_the_windows_the_queue_stays_unflagged():
    record = _breach_record("BR-MID")
    _acknowledge(record, days_ago=20)
    view = queue_aging([record], NOW)
    assert "acknowledged_escalation" not in _queue_item(view, "BR-MID")
    assert view["escalated_acknowledgements"] == []
    assert view["items_with_escalated_acknowledgements"] == 0


def test_a_decided_case_never_escalates_however_old_the_acknowledgement():
    record = _breach_record("BR-DONE", status="approved")
    _acknowledge(record, days_ago=60)
    view = queue_aging([record], NOW)
    assert view["escalated_acknowledgements"] == []
    digest = ops_digest([record], [], now=NOW)
    assert digest["counts"]["acknowledged_escalated"] == 0


def test_the_queue_rung_scales_with_the_follow_through_window():
    record = _breach_record("BR-FAST")
    _acknowledge(record, days_ago=15)
    view = queue_aging([record], NOW, follow_escalation_days=7)
    assert view["items_with_escalated_acknowledgements"] == 1
    assert view["escalated_acknowledgements"][0]["escalation_days"] == 14


def test_oldest_acknowledgement_escalates_first_in_the_list():
    older = _breach_record("BR-B")
    _acknowledge(older, days_ago=40)
    newer = _breach_record("BR-A")
    _acknowledge(newer, days_ago=29)
    view = queue_aging([newer, older], NOW)
    assert [
        e["submission_id"] for e in view["escalated_acknowledgements"]
    ] == ["BR-B", "BR-A"]
