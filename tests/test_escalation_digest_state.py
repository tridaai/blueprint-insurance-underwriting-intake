"""Round 18: the escalation telling's state in the ops digest.

Round 16 gave the second rung a telling of its own — a signed
``acknowledged_escalation`` webhook with its own notice and
ledger — but the ops digest, the morning read, named the
escalated cases without saying whether that telling ever fired,
failed, or was never run. The digest now carries the notice
state on each still-waiting entry (delivered / announced /
exhausted / unannounced / not_escalated), counts the exhausted
escalation notices as their own event — distinct from exhausted
breach notices, where nobody was told at all — and counts the
escalated cases no check has ever announced, so a sweep nobody
ran is itself visible. Both counts are subsets of cases already
counted as still waiting: neither enters ``total_attention``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    digest_trend,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
WAIT_DAYS = 14  # the follow-through window the digest defaults to


def _attempt(at: datetime, status: str = "failed") -> dict:
    return {
        "attempt": 1,
        "attempted_at": at.isoformat(timespec="seconds"),
        "status": status,
        "http_status": 500 if status == "failed" else 204,
        "signature_id": None,
        "error": "connection refused" if status == "failed" else None,
        "backoff_seconds": 0,
    }


def _rung_record(
    sid: str,
    *,
    acked_days_ago: int = 35,
    escalation_notice: dict | None = None,
    escalation_attempts: list[dict] | None = None,
) -> ApprovalRecord:
    """An exhausted breach notice acknowledged ``acked_days_ago``
    before NOW (default: past the second rung at 2 × 14 days),
    still undecided — a still-waiting, escalated case."""
    observed = NOW - timedelta(days=60)
    acked_at = (NOW - timedelta(days=acked_days_ago)).isoformat(
        timespec="seconds"
    )
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-08-01",
        },
        sla_breach_notice={
            "first_observed_at": observed.isoformat(timespec="seconds"),
            "submission_id": sid,
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "outcome": "borderline",
            "risk_tier": "moderate",
            "sla": {"budget_days": 5, "age_days": 40, "days_over": 35},
            "dispatch_status": "failed",
        },
        sla_breach_attempts=[
            _attempt(observed + timedelta(hours=i)) for i in range(8)
        ],
        breach_notice_acknowledgement={
            "acknowledged_by": "Ops Lead",
            "acknowledged_at": acked_at,
            "note": "Called the broker",
        },
        escalation_notice=escalation_notice,
        escalation_attempts=escalation_attempts or [],
    )


def _notice(sid: str, acked_days_ago: int = 35) -> dict:
    fired = NOW - timedelta(days=acked_days_ago - 2 * WAIT_DAYS)
    return {
        "first_observed_at": fired.isoformat(timespec="seconds"),
        "submission_id": sid,
        "insured_name": f"Insured {sid}",
        "broker_name": "Broker Q",
        "risk_tier": "moderate",
        "acknowledgement": {
            "acknowledged_by": "Ops Lead",
            "acknowledged_at": (
                NOW - timedelta(days=acked_days_ago)
            ).isoformat(timespec="seconds"),
            "note": "Called the broker",
        },
        "acknowledged_age_days": 2 * WAIT_DAYS,
        "escalation_window_days": WAIT_DAYS,
        "dispatch_status": "failed",
    }


def _entry(digest: dict, sid: str) -> dict:
    return next(
        e
        for e in digest["acknowledged_still_waiting"]
        if e["submission_id"] == sid
    )


# --- the per-entry state -------------------------------------------------------


def test_an_escalated_case_with_no_notice_reads_unannounced():
    digest = ops_digest([_rung_record("ESC-U")], now=NOW)
    entry = _entry(digest, "ESC-U")
    assert entry["escalated"] is True
    assert entry["escalation_notice_state"]["state"] == "unannounced"
    assert entry["escalation_notice_state"]["attempts"] == 0
    assert digest["counts"]["unannounced_escalations"] == 1
    assert "never announced" in digest["headline"]
    assert "the escalation check has not run" in digest["headline"]


def test_a_delivered_escalation_reads_delivered():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "ESC-D",
        escalation_notice=_notice("ESC-D"),
        escalation_attempts=[_attempt(fired, status="sent")],
    )
    digest = ops_digest([record], now=NOW)
    state = _entry(digest, "ESC-D")["escalation_notice_state"]
    assert state["state"] == "delivered"
    assert state["attempts"] == 1
    assert digest["counts"]["unannounced_escalations"] == 0
    assert digest["counts"]["exhausted_escalation_notices"] == 0


def test_a_failed_escalation_with_attempts_left_reads_announced():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "ESC-A",
        escalation_notice=_notice("ESC-A"),
        escalation_attempts=[
            _attempt(fired),
            _attempt(fired + timedelta(hours=1)),
        ],
    )
    digest = ops_digest([record], now=NOW)
    state = _entry(digest, "ESC-A")["escalation_notice_state"]
    assert state["state"] == "announced"
    assert state["attempts"] == 2
    assert state["last_error"] == "connection refused"
    assert digest["counts"]["exhausted_escalation_notices"] == 0


def test_a_case_still_in_its_first_window_reads_not_escalated():
    # Acknowledged 20 days ago: past the 14-day waiting window,
    # short of the 28-day second rung — no telling is owed yet.
    digest = ops_digest([_rung_record("ESC-N", acked_days_ago=20)], now=NOW)
    entry = _entry(digest, "ESC-N")
    assert entry["escalated"] is False
    assert entry["escalation_notice_state"] == {"state": "not_escalated"}
    assert digest["counts"]["unannounced_escalations"] == 0


# --- the exhausted telling, counted as its own event ---------------------------


def test_an_exhausted_escalation_is_counted_and_listed_on_its_own():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "ESC-X",
        escalation_notice=_notice("ESC-X"),
        escalation_attempts=[
            _attempt(fired + timedelta(hours=i)) for i in range(8)
        ],
    )
    digest = ops_digest([record], now=NOW)
    assert _entry(digest, "ESC-X")["escalation_notice_state"]["state"] == (
        "exhausted"
    )
    counts = digest["counts"]
    assert counts["exhausted_escalation_notices"] == 1
    # The distinct event: this case's breach notice was
    # acknowledged, so it is NOT an exhausted breach notice —
    # somebody was told; the escalation is what never landed.
    assert counts["exhausted_breach_notices"] == 0
    (listed,) = digest["exhausted_escalation_notices"]
    assert listed["submission_id"] == "ESC-X"
    assert listed["notice_kind"] == "acknowledged_escalation"
    assert "exhausted escalation notice" in digest["headline"]
    assert "the escalation saying so never landed" in digest["headline"]


def test_the_telling_counts_never_enter_total_attention():
    fired = NOW - timedelta(days=7)
    records = [
        _rung_record("ESC-U"),
        _rung_record(
            "ESC-X",
            escalation_notice=_notice("ESC-X"),
            escalation_attempts=[
                _attempt(fired + timedelta(hours=i)) for i in range(8)
            ],
        ),
    ]
    digest = ops_digest(records, now=NOW)
    counts = digest["counts"]
    assert counts["unannounced_escalations"] == 1
    assert counts["exhausted_escalation_notices"] == 1
    # Both cases are already counted once, as still waiting;
    # the telling counts read their state, not new work.
    worklist_keys = (
        "stuck_dispatches",
        "exhausted_breach_notices",
        "acknowledged_still_waiting",
        "breached_queue",
        "follow_through_escalated",
        "proposals_expired",
    )
    assert counts["total_attention"] == sum(
        counts[key] for key in worklist_keys
    )


def test_the_snapshot_keeps_the_telling_counts_and_section():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "ESC-X",
        escalation_notice=_notice("ESC-X"),
        escalation_attempts=[
            _attempt(fired + timedelta(hours=i)) for i in range(8)
        ],
    )
    digest = ops_digest([record], now=NOW)
    snap = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert snap["counts"]["exhausted_escalation_notices"] == 1
    assert snap["counts"]["unannounced_escalations"] == 0
    assert snap["sections"]["exhausted_escalation_notices"] == ["ESC-X"]
    trend = digest_trend([snap])
    assert (
        trend["series"][0]["counts"]["exhausted_escalation_notices"] == 1
    )


def test_movement_names_an_entered_exhausted_escalation():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "ESC-X",
        escalation_notice=_notice("ESC-X"),
        escalation_attempts=[
            _attempt(fired + timedelta(hours=i)) for i in range(8)
        ],
    )
    before = build_digest_snapshot(
        ops_digest([], now=NOW), "snap-0001", "Ops Lead"
    )
    digest = ops_digest([record], now=NOW, latest_snapshot=before)
    assert "exhausted escalation notices up 1" in digest["headline"]
    assert (
        "ESC-X entered exhausted escalation notices" in digest["headline"]
    )
