"""Round 23: the queue's escalated list, ordered by the telling.

Round 22 put the telling's state on every
``escalated_acknowledgements`` entry, but the list still
sorted by acknowledgement age alone — so the case whose
escalation landed a month ago could sit above the case no
sweep has ever announced. The list now orders by telling
band (``insights._escalated_telling_order``): the
**unannounced** lead, stalest silence first — a sweep owes
those cases a voice; then the **exhausted**, whose telling
spent its attempts; then the **announced**, still failing
with retries left; and last the **delivered**, whose cases
wait on a decision, not a telling. Inside a band the oldest
acknowledgement still leads, the order the list has always
read — round 16's pin (two unannounced cases, the older
acknowledgement first) survives, because inside the
unannounced band the stalest silence *is* the oldest
acknowledgement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import queue_aging
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

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


def _notice(sid: str, acked_days_ago: int) -> dict:
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


def _rung_record(
    sid: str,
    *,
    acked_days_ago: int,
    escalation_notice: dict | None = None,
    escalation_attempts: list[dict] | None = None,
) -> ApprovalRecord:
    """An exhausted breach notice acknowledged
    ``acked_days_ago`` before NOW, still undecided — a case
    on the queue's escalated surfaces, its telling shaped by
    the notice/attempts it is handed."""
    observed = NOW - timedelta(days=60)
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
                NOW - timedelta(days=acked_days_ago)
            ).isoformat(timespec="seconds"),
            "note": "Called the broker",
        },
        escalation_notice=escalation_notice,
        escalation_attempts=escalation_attempts or [],
    )


def _unannounced(sid: str, acked_days_ago: int) -> ApprovalRecord:
    return _rung_record(sid, acked_days_ago=acked_days_ago)


def _delivered(sid: str, acked_days_ago: int) -> ApprovalRecord:
    fired = NOW - timedelta(days=7)
    return _rung_record(
        sid,
        acked_days_ago=acked_days_ago,
        escalation_notice=_notice(sid, acked_days_ago),
        escalation_attempts=[_attempt(fired, status="sent")],
    )


def _announced(sid: str, acked_days_ago: int) -> ApprovalRecord:
    fired = NOW - timedelta(days=7)
    return _rung_record(
        sid,
        acked_days_ago=acked_days_ago,
        escalation_notice=_notice(sid, acked_days_ago),
        escalation_attempts=[
            _attempt(fired),
            _attempt(fired + timedelta(hours=1)),
        ],
    )


def _exhausted(sid: str, acked_days_ago: int) -> ApprovalRecord:
    fired = NOW - timedelta(days=7)
    return _rung_record(
        sid,
        acked_days_ago=acked_days_ago,
        escalation_notice=_notice(sid, acked_days_ago),
        escalation_attempts=[
            _attempt(fired + timedelta(hours=i)) for i in range(8)
        ],
    )


def _order(view: dict) -> list[str]:
    return [e["submission_id"] for e in view["escalated_acknowledgements"]]


def _states(view: dict) -> list[str]:
    return [
        e["escalation_notice_state"]["state"]
        for e in view["escalated_acknowledgements"]
    ]


# --- the bands -----------------------------------------------------------------


def test_the_bands_lead_in_the_order_the_work_is_owed():
    # Acknowledgement ages run exactly against the bands:
    # the delivered case carries the oldest acknowledgement
    # and the unannounced the freshest-but-one — under the
    # old age-only order the list would read D, A, X, U.
    cases = [
        _delivered("ORD-D", acked_days_ago=60),
        _announced("ORD-A", acked_days_ago=50),
        _exhausted("ORD-X", acked_days_ago=45),
        _unannounced("ORD-U", acked_days_ago=40),
    ]
    view = queue_aging(cases, NOW)
    assert _states(view) == [
        "unannounced",
        "exhausted",
        "announced",
        "delivered",
    ]
    assert _order(view) == ["ORD-U", "ORD-X", "ORD-A", "ORD-D"]


def test_the_unannounced_lead_stalest_silence_first():
    cases = [
        _unannounced("ORD-U-FRESH", acked_days_ago=30),
        _unannounced("ORD-U-STALE", acked_days_ago=44),
        _unannounced("ORD-U-MID", acked_days_ago=37),
    ]
    view = queue_aging(cases, NOW)
    assert _order(view) == ["ORD-U-STALE", "ORD-U-MID", "ORD-U-FRESH"]
    ages = [
        e["unannounced_days"] for e in view["escalated_acknowledgements"]
    ]
    assert ages == sorted(ages, reverse=True)


def test_inside_a_band_the_oldest_acknowledgement_still_leads():
    cases = [
        _exhausted("ORD-X-NEW", acked_days_ago=32),
        _exhausted("ORD-X-OLD", acked_days_ago=52),
        _delivered("ORD-D-NEW", acked_days_ago=31),
        _delivered("ORD-D-OLD", acked_days_ago=47),
    ]
    view = queue_aging(cases, NOW)
    assert _order(view) == [
        "ORD-X-OLD",
        "ORD-X-NEW",
        "ORD-D-OLD",
        "ORD-D-NEW",
    ]


def test_the_item_blocks_follow_their_cases_not_the_order():
    # Ordering is a property of the list; each queue item
    # still carries its own case's acknowledgement block.
    cases = [
        _delivered("ORD-D", acked_days_ago=60),
        _unannounced("ORD-U", acked_days_ago=40),
    ]
    view = queue_aging(cases, NOW)
    items = {
        item["submission_id"]: item
        for bucket in view["buckets"]
        for item in bucket["submissions"]
    }
    assert (
        items["ORD-U"]["acknowledged_escalation"][
            "escalation_notice_state"
        ]["state"]
        == "unannounced"
    )
    assert (
        items["ORD-D"]["acknowledged_escalation"][
            "escalation_notice_state"
        ]["state"]
        == "delivered"
    )
    assert view["items_with_escalated_acknowledgements"] == 2
