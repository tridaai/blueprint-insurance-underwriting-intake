"""Round 24: the queue counts the telling's bands.

Round 23 ordered the queue's escalated list by telling band
— unannounced, exhausted, announced, delivered — but the
worklist's shape still read only by scanning the list: how
many cases nobody has heard about, against how many merely
wait on a decision, was a count an operator made by eye. The
queue read now carries ``telling_band_counts`` beside
``items_with_escalated_acknowledgements``, composed from the
same entries the ordering reads: the list shows the work,
the counts state its shape, and the two can never disagree.
The bands are named in band order — the unannounced first —
zero-filled, because a band with no cases in it is a fact
about the queue, stated, not omitted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import queue_aging
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
WAIT_DAYS = 14  # the follow-through window; the rung is 2 × it


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
    now: datetime = NOW,
    acked_days_ago: int = 35,
    escalation_notice: dict | None = None,
    escalation_attempts: list[dict] | None = None,
) -> ApprovalRecord:
    """An exhausted breach notice acknowledged ``acked_days_ago``
    before ``now``, still undecided — a case on the queue's
    escalated surfaces (the round-22 builder)."""
    observed = now - timedelta(days=60)
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
            "acknowledged_at": (
                now - timedelta(days=acked_days_ago)
            ).isoformat(timespec="seconds"),
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


def _mixed_queue() -> list[ApprovalRecord]:
    """One case in each telling band, and a second silent one."""
    fired = NOW - timedelta(days=7)
    return [
        _rung_record("Q-U1"),
        _rung_record("Q-U2", acked_days_ago=40),
        _rung_record(
            "Q-X",
            escalation_notice=_notice("Q-X"),
            escalation_attempts=[
                _attempt(fired + timedelta(hours=i)) for i in range(8)
            ],
        ),
        _rung_record(
            "Q-A",
            escalation_notice=_notice("Q-A"),
            escalation_attempts=[_attempt(fired)],
        ),
        _rung_record(
            "Q-D",
            escalation_notice=_notice("Q-D"),
            escalation_attempts=[_attempt(fired, status="sent")],
        ),
    ]


# --- the counts -----------------------------------------------------------------


def test_the_bands_are_counted_in_band_order():
    view = queue_aging(_mixed_queue(), NOW)
    counts = view["telling_band_counts"]
    assert counts == {
        "unannounced": 2,
        "exhausted": 1,
        "announced": 1,
        "delivered": 1,
    }
    # Band order is the read order: the unannounced first, as
    # the console header names them.
    assert list(counts) == [
        "unannounced",
        "exhausted",
        "announced",
        "delivered",
    ]


def test_the_counts_are_composed_from_the_list_they_beside():
    view = queue_aging(_mixed_queue(), NOW)
    entries = view["escalated_acknowledgements"]
    assert sum(view["telling_band_counts"].values()) == len(entries)
    for band, count in view["telling_band_counts"].items():
        assert count == sum(
            1
            for e in entries
            if e["escalation_notice_state"]["state"] == band
        )
    # And the ordering still leads with the counted-first band.
    assert entries[0]["escalation_notice_state"]["state"] == "unannounced"


def test_an_empty_escalated_list_counts_zero_in_every_band():
    view = queue_aging([], NOW)
    assert view["telling_band_counts"] == {
        "unannounced": 0,
        "exhausted": 0,
        "announced": 0,
        "delivered": 0,
    }


def test_a_case_before_the_second_rung_is_in_no_band():
    view = queue_aging([_rung_record("Q-MID", acked_days_ago=20)], NOW)
    assert view["escalated_acknowledgements"] == []
    assert sum(view["telling_band_counts"].values()) == 0


def test_the_service_queue_carries_the_counts():
    # The service reads the real clock, so these records are
    # anchored to it rather than the frozen NOW above.
    real_now = datetime.now(timezone.utc)
    store = InMemoryStore()
    store.save(_rung_record("Q-U1", now=real_now))
    store.save(
        _rung_record(
            "Q-D",
            now=real_now,
            escalation_notice=_notice("Q-D"),
            escalation_attempts=[
                _attempt(real_now - timedelta(days=7), status="sent")
            ],
        )
    )
    service = SubmissionService(store=store, checkpointer=False)
    view = service.queue()
    assert view["telling_band_counts"]["unannounced"] == 1
    assert view["telling_band_counts"]["delivered"] == 1
