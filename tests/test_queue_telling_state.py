"""Round 22: the queue reads the telling's state.

Round 18 taught the ops digest to name the second rung's
telling — delivered / announced / exhausted / unannounced —
but the queue, where the underwriter who finally picks the
case up works, still carried only the raw escalation notice:
a dict to be interpreted on the spot, or "not fired yet" in
the console's paraphrase. The queue's escalated surfaces now
carry the computed ``escalation_notice_state`` beside the raw
notice — the digest's own computation, not a second reading —
and age the unannounced the digest's way
(``unannounced_days``), on both the
``escalated_acknowledgements`` list and the item's own
``acknowledged_escalation`` block. One vocabulary, two
surfaces: the morning read and the worklist cannot disagree
about whether an escalation landed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import ops_digest, queue_aging
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


def _rung_record(
    sid: str,
    *,
    acked_days_ago: int = 35,
    escalation_notice: dict | None = None,
    escalation_attempts: list[dict] | None = None,
) -> ApprovalRecord:
    """An exhausted breach notice acknowledged ``acked_days_ago``
    before NOW (default 35: past the second rung at 28 days),
    still undecided — a case on the queue's escalated surfaces."""
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


def _listed(view: dict, sid: str) -> dict:
    return next(
        e
        for e in view["escalated_acknowledgements"]
        if e["submission_id"] == sid
    )


def _queue_item(view: dict, sid: str) -> dict:
    for bucket in view["buckets"]:
        for item in bucket["submissions"]:
            if item["submission_id"] == sid:
                return item
    raise AssertionError(f"{sid} not in the queue")


# --- the list entries read the telling -----------------------------------------


def test_an_unannounced_case_reads_unannounced_and_aged_on_the_list():
    view = queue_aging([_rung_record("Q-U")], NOW)
    entry = _listed(view, "Q-U")
    # The raw notice rides as it always has — null here, never fired.
    assert entry["escalation_notice"] is None
    state = entry["escalation_notice_state"]
    assert state["state"] == "unannounced"
    assert state["attempts"] == 0
    # Acknowledged 35 days ago; the rung is at 28 — seven days
    # of silence since the telling fell due.
    assert entry["unannounced_days"] == 35 - 2 * WAIT_DAYS


def test_a_delivered_telling_reads_delivered():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "Q-D",
        escalation_notice=_notice("Q-D"),
        escalation_attempts=[_attempt(fired, status="sent")],
    )
    entry = _listed(queue_aging([record], NOW), "Q-D")
    state = entry["escalation_notice_state"]
    assert state["state"] == "delivered"
    assert state["attempts"] == 1
    assert state["last_http_status"] == 204
    assert "unannounced_days" not in entry


def test_a_telling_in_trouble_reads_announced():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "Q-A",
        escalation_notice=_notice("Q-A"),
        escalation_attempts=[
            _attempt(fired),
            _attempt(fired + timedelta(hours=1)),
        ],
    )
    entry = _listed(queue_aging([record], NOW), "Q-A")
    state = entry["escalation_notice_state"]
    assert state["state"] == "announced"
    assert state["attempts"] == 2
    assert "unannounced_days" not in entry


def test_a_spent_telling_reads_exhausted():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "Q-X",
        escalation_notice=_notice("Q-X"),
        escalation_attempts=[
            _attempt(fired + timedelta(hours=i)) for i in range(8)
        ],
    )
    entry = _listed(queue_aging([record], NOW), "Q-X")
    state = entry["escalation_notice_state"]
    assert state["state"] == "exhausted"
    assert state["attempts"] == 8
    assert "unannounced_days" not in entry


# --- the item block reads the same words ----------------------------------------


def test_the_item_block_carries_the_state_and_the_age():
    view = queue_aging([_rung_record("Q-U")], NOW)
    block = _queue_item(view, "Q-U")["acknowledged_escalation"]
    assert block["acknowledged_by"] == "Ops Lead"
    assert block["escalation_notice_state"]["state"] == "unannounced"
    assert block["unannounced_days"] == 35 - 2 * WAIT_DAYS


def test_the_item_block_omits_the_age_once_the_telling_fired():
    fired = NOW - timedelta(days=7)
    record = _rung_record(
        "Q-D",
        escalation_notice=_notice("Q-D"),
        escalation_attempts=[_attempt(fired, status="sent")],
    )
    block = _queue_item(queue_aging([record], NOW), "Q-D")[
        "acknowledged_escalation"
    ]
    assert block["escalation_notice_state"]["state"] == "delivered"
    assert "unannounced_days" not in block


# --- one vocabulary, two surfaces -----------------------------------------------


def test_the_queue_and_the_digest_name_the_same_telling_alike():
    fired = NOW - timedelta(days=7)
    cases = [
        _rung_record("Q-U"),
        _rung_record(
            "Q-D",
            escalation_notice=_notice("Q-D"),
            escalation_attempts=[_attempt(fired, status="sent")],
        ),
        _rung_record(
            "Q-X",
            escalation_notice=_notice("Q-X"),
            escalation_attempts=[
                _attempt(fired + timedelta(hours=i)) for i in range(8)
            ],
        ),
    ]
    view = queue_aging(cases, NOW)
    digest = ops_digest(cases, [], now=NOW)
    digest_states = {
        e["submission_id"]: e["escalation_notice_state"]["state"]
        for e in digest["acknowledged_still_waiting"]
    }
    for sid in ("Q-U", "Q-D", "Q-X"):
        queue_state = _listed(view, sid)["escalation_notice_state"][
            "state"
        ]
        assert queue_state == digest_states[sid]
    # And the unannounced age is the digest's number, not a
    # queue-local one.
    digest_entry = next(
        e
        for e in digest["acknowledged_still_waiting"]
        if e["submission_id"] == "Q-U"
    )
    assert _listed(view, "Q-U")["unannounced_days"] == digest_entry[
        "unannounced_days"
    ]


def test_before_the_second_rung_there_is_no_telling_to_read():
    record = _rung_record("Q-MID", acked_days_ago=20)
    view = queue_aging([record], NOW)
    assert view["escalated_acknowledgements"] == []
    assert "acknowledged_escalation" not in _queue_item(view, "Q-MID")
