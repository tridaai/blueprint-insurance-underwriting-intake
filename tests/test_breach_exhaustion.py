"""Round 12: breach-notice exhaustion — the case nobody was told about.

An SLA-breach notice has its own delivery ledger, separate from
the approval ledger the dispatch view reads. When that ledger
spends its whole attempt budget and the last attempt still
failed, the notice never landed: the case breached, it is still
undecided, and no automated nudge is coming. The dispatch view
and the ops digest now carry those notices as their own list —
aged from first observation, bucketed, and distinguished from
approval failures, because the next step is a human call, not
another retry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    digest_trend,
    dispatch_overview,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord, InMemoryStore
from underwriting_agent.service import SubmissionService

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


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


def _breach_record(
    sid: str,
    *,
    observed_days_ago: int = 8,
    attempts: int = 8,
    last_status: str = "failed",
    record_status: str = "awaiting_underwriter",
) -> ApprovalRecord:
    observed = NOW - timedelta(days=observed_days_ago)
    ledger = [
        _attempt(observed + timedelta(hours=i)) for i in range(attempts - 1)
    ]
    ledger.append(_attempt(observed + timedelta(hours=attempts), last_status))
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=record_status,
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
            "sla": {"budget_days": 5, "age_days": 12, "days_over": 7},
            "dispatch_status": last_status,
        },
        sla_breach_attempts=ledger,
    )


# ---------------------------------------------------------------------------
# The dispatch view: exhausted, distinguished, aged
# ---------------------------------------------------------------------------


def test_an_exhausted_notice_is_listed_aged_and_bucketed():
    overview = dispatch_overview([_breach_record("BR-1")], now=NOW)
    assert overview["exhausted_breach_count"] == 1
    entry = overview["exhausted_breach_notices"][0]
    assert entry["submission_id"] == "BR-1"
    assert entry["attempts"] == 8
    assert entry["attempts_remaining"] == 0
    assert entry["last_http_status"] == 500
    # Age runs from the first observation — the breach is 8 days
    # old, not as old as the last failed attempt.
    assert entry["notice_age_days"] == 8
    assert entry["notice_age_bucket"] == "8-14 days"
    assert entry["sla_budget_days"] == 5
    buckets = {b["bucket"]: b for b in overview["exhausted_breach_age_buckets"]}
    assert buckets["8-14 days"]["submission_ids"] == ["BR-1"]


def test_exhausted_notices_are_not_approval_failures():
    record = _breach_record("BR-1")
    # The same record also has a failed approval dispatch: the
    # two ledgers stay in their own lists.
    record.dispatch_status = "failed"
    record.dispatch_attempts = [_attempt(NOW - timedelta(days=1))]
    overview = dispatch_overview([record], now=NOW)
    assert [f["submission_id"] for f in overview["failures"]] == ["BR-1"]
    assert overview["failed_records"] == 1
    assert overview["exhausted_breach_count"] == 1
    assert overview["exhausted_breach_notices"][0] is not overview["failures"][0]


def test_a_notice_with_attempts_left_is_not_exhausted():
    overview = dispatch_overview(
        [_breach_record("BR-1", attempts=7)], now=NOW
    )
    assert overview["exhausted_breach_count"] == 0
    # …but a smaller configured budget exhausts it sooner.
    overview = dispatch_overview(
        [_breach_record("BR-1", attempts=7)], now=NOW, attempt_budget=7
    )
    assert overview["exhausted_breach_count"] == 1


def test_a_delivered_or_decided_notice_is_not_exhausted():
    delivered = _breach_record("BR-SENT", last_status="sent")
    decided = _breach_record("BR-DONE", record_status="approved")
    no_notice = ApprovalRecord(
        result=AgentResult(submission_id="BR-NONE", status="awaiting_underwriter"),
        submission={"submitted_at": "2026-09-01"},
    )
    overview = dispatch_overview([delivered, decided, no_notice], now=NOW)
    assert overview["exhausted_breach_count"] == 0
    assert overview["exhausted_breach_notices"] == []


def test_exhausted_notices_order_oldest_observation_first():
    newer = _breach_record("BR-NEW", observed_days_ago=2)
    older = _breach_record("BR-OLD", observed_days_ago=12)
    overview = dispatch_overview([newer, older], now=NOW)
    assert [
        e["submission_id"] for e in overview["exhausted_breach_notices"]
    ] == ["BR-OLD", "BR-NEW"]


# ---------------------------------------------------------------------------
# The ops digest: beside the stuck dispatches, in the headline
# ---------------------------------------------------------------------------


def test_digest_carries_exhausted_notices_in_count_headline_and_total():
    digest = ops_digest([_breach_record("BR-1")], [], now=NOW)
    assert digest["counts"]["exhausted_breach_notices"] == 1
    assert [e["submission_id"] for e in digest["exhausted_breach_notices"]] == [
        "BR-1"
    ]
    assert "1 exhausted breach notice" in digest["headline"]
    assert "nobody was told" in digest["headline"]
    # The case itself also counts as a breached queue case — the
    # two surfaces answer different questions (the queue asks who
    # picks it up; the notice asks who was told), so the total
    # counts it on both.
    assert digest["counts"]["breached_queue"] == 1
    assert digest["counts"]["total_attention"] == 2


def test_service_digest_and_dispatch_views_agree():
    store = InMemoryStore()
    store.save(_breach_record("BR-1"))
    service = SubmissionService(store=store, checkpointer=False)
    overview = service.dispatch_ops()
    assert overview["exhausted_breach_count"] == 1
    digest = service.ops_digest()
    assert digest["counts"]["exhausted_breach_notices"] == 1


def test_snapshots_keep_the_exhausted_section_for_the_trend():
    digest = ops_digest([_breach_record("BR-1")], [], now=NOW)
    first = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert first["counts"]["exhausted_breach_notices"] == 1
    assert first["sections"]["exhausted_breach_notices"] == ["BR-1"]
    second = build_digest_snapshot(
        ops_digest([], [], now=NOW), "snap-0002", "Ops Lead"
    )
    trend = digest_trend([first, second])
    assert trend["deltas"]["exhausted_breach_notices"]["since_previous"] == -1
    assert trend["movement"]["sections"]["exhausted_breach_notices"] == {
        "entered": [], "cleared": ["BR-1"],
    }
