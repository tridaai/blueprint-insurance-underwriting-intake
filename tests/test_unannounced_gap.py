"""Round 19: the sweep gap, aged.

Round 18's digest counted the escalated cases no escalation
check had ever announced — a sweep nobody ran, made visible.
But a count says nothing about how long the case has sat
unannounced: the escalation check that has not run yet today
and the one that stopped a month ago were the same number.
The still-waiting entry now carries ``unannounced_days`` —
the days since the case crossed the second rung with no
telling — the headline names the oldest, and the cases
unannounced past the gap threshold (one further window by
default, ``OPS_DIGEST_UNANNOUNCED_GAP_DAYS`` to move it) are
counted on their own as ``stale_unannounced_escalations``:
the gap itself becomes an attention count. It stays a subset
of still-waiting — the cases are already counted there, so
it never enters ``total_attention``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import build_digest_snapshot, ops_digest
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


def _rung_record(sid: str, *, acked_days_ago: int) -> ApprovalRecord:
    """An exhausted breach notice acknowledged ``acked_days_ago``
    before NOW, still undecided, never escalation-announced —
    a still-waiting, escalated, unannounced case whose
    unannounced age is ``acked_days_ago - 2 * WAIT_DAYS``."""
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
    )


def _entry(digest: dict, sid: str) -> dict:
    return next(
        e
        for e in digest["acknowledged_still_waiting"]
        if e["submission_id"] == sid
    )


# --- the age on the entry ---------------------------------------------------------


def test_an_unannounced_entry_carries_its_age_since_the_rung():
    digest = ops_digest([_rung_record("ESC-1", acked_days_ago=35)], now=NOW)
    entry = _entry(digest, "ESC-1")
    assert entry["escalation_notice_state"]["state"] == "unannounced"
    # Crossed the rung at 2 × 14 days of acknowledgement age;
    # seven more days have passed with no telling.
    assert entry["unannounced_days"] == 35 - 2 * WAIT_DAYS


def test_a_just_escalated_unannounced_case_is_young_not_stale():
    digest = ops_digest([_rung_record("ESC-1", acked_days_ago=29)], now=NOW)
    entry = _entry(digest, "ESC-1")
    assert entry["unannounced_days"] == 1
    assert digest["counts"]["unannounced_escalations"] == 1
    assert digest["counts"]["stale_unannounced_escalations"] == 0


# --- the stale count -----------------------------------------------------------------


def test_past_the_gap_threshold_the_gap_itself_is_counted():
    digest = ops_digest([_rung_record("ESC-1", acked_days_ago=50)], now=NOW)
    # 50 - 28 = 22 unannounced days; the default threshold is
    # one further window (14).
    assert digest["counts"]["stale_unannounced_escalations"] == 1
    assert digest["unannounced_gap_days"] == WAIT_DAYS
    assert "past the 14-day gap threshold" in digest["headline"]
    assert "the sweep gap itself needs attention" in digest["headline"]


def test_the_headline_names_the_oldest_unannounced_age():
    records = [
        _rung_record("ESC-1", acked_days_ago=35),  # 7 days unannounced
        _rung_record("ESC-2", acked_days_ago=50),  # 22 days unannounced
    ]
    digest = ops_digest(records, now=NOW)
    assert digest["counts"]["unannounced_escalations"] == 2
    assert "the oldest has sat unannounced 22 day(s)" in digest["headline"]


def test_the_gap_threshold_is_a_parameter_not_a_constant():
    records = [_rung_record("ESC-1", acked_days_ago=50)]  # 22 days
    wide = ops_digest(records, now=NOW, unannounced_gap_days=30)
    assert wide["counts"]["stale_unannounced_escalations"] == 0
    narrow = ops_digest(records, now=NOW, unannounced_gap_days=5)
    assert narrow["counts"]["stale_unannounced_escalations"] == 1
    assert narrow["unannounced_gap_days"] == 5


def test_the_gap_threshold_reads_the_environment(monkeypatch):
    monkeypatch.setenv("OPS_DIGEST_UNANNOUNCED_GAP_DAYS", "3")
    digest = ops_digest([_rung_record("ESC-1", acked_days_ago=35)], now=NOW)
    assert digest["unannounced_gap_days"] == 3
    assert digest["counts"]["stale_unannounced_escalations"] == 1


def test_the_stale_count_never_enters_total_attention():
    young = ops_digest([_rung_record("ESC-1", acked_days_ago=35)], now=NOW)
    stale = ops_digest([_rung_record("ESC-1", acked_days_ago=50)], now=NOW)
    assert young["counts"]["stale_unannounced_escalations"] == 0
    assert stale["counts"]["stale_unannounced_escalations"] == 1
    # The same case, read young and read stale: the stale
    # reading of its telling adds nothing to the total — the
    # case is already counted as still waiting (and breached).
    assert (
        stale["counts"]["total_attention"]
        == young["counts"]["total_attention"]
    )


# --- the kept history -------------------------------------------------------------------


def test_the_stale_count_rides_the_digest_snapshots():
    fresh = ops_digest([_rung_record("ESC-1", acked_days_ago=29)], now=NOW)
    snapshot = build_digest_snapshot(fresh, "snap-0001", "Ops Lead")
    assert snapshot["counts"]["stale_unannounced_escalations"] == 0
    aged = ops_digest(
        [_rung_record("ESC-1", acked_days_ago=50)],
        now=NOW,
        latest_snapshot=snapshot,
    )
    assert aged["counts"]["stale_unannounced_escalations"] == 1
    assert "stale unannounced escalations up 1" in aged["headline"]
