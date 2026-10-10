"""Round 17: worked counts in the digest snapshots.

Round 16 gave escalated follow-through notes a worked state,
but the state was a live read with no history: the digest
counts carried the problem (escalated notes) and not the
response (notes touched, escalations quieted), so a kept
snapshot could never say whether touches keep pace with
escalations. The counts now carry ``follow_through_worked``
and ``follow_through_quieted`` on the same discipline as the
advisory counts — they ride ``DIGEST_COUNT_KEYS``, so the
snapshots keep them and the trend reads the response beside
the problem — and they never enter ``total_attention``:
attention counts work outstanding, not work done.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    digest_trend,
    follow_through_digest,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _note(drafted_at: str, touches: list[dict] | None = None) -> dict:
    note = {
        "submission_id": "OLD-1",
        "drafted_at": drafted_at,
        "text": "INTERNAL DRAFT …",
        "outstanding": {
            "missing_elements": ["loss runs"],
            "gap_sources": [],
            "blocking_lines": [],
        },
        "follow_through": {
            "recorded": True,
            "state": "still_outstanding",
            "answered": [],
            "still_outstanding": [{"line": "", "item": "loss runs"}],
            "by_submission_id": "NEW-0",
            "checked_at": "2026-10-09T12:00:00+00:00",
        },
    }
    if touches is not None:
        note["escalation_touches"] = touches
    return note


def _record(sid: str, note: dict) -> ApprovalRecord:
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
            "submitted_at": "2026-10-15",
        },
        status_note=note,
    )


def _touch(by: str, at: str) -> dict:
    return {"worked_by": by, "worked_at": at, "note": "chased the broker"}


# Note drafted 30 days before NOW, touched 2 days before NOW:
# the touch is inside its window, so the escalation is quieted.
QUIETED = _record(
    "OLD-Q",
    _note(
        "2026-09-20T09:00:00+00:00",
        touches=[_touch("Uma", "2026-10-18T09:00:00+00:00")],
    ),
)
# Note drafted 30 days before NOW, never touched: escalated.
ESCALATED = _record("OLD-E", _note("2026-09-20T09:00:00+00:00"))
# Note drafted 45 days before NOW, touched 25 days before NOW:
# the touch aged past its window, so the note re-escalated —
# neither worked-standing nor quieted.
LAPSED = _record(
    "OLD-L",
    _note(
        "2026-09-05T09:00:00+00:00",
        touches=[_touch("Uma", "2026-09-25T09:00:00+00:00")],
    ),
)


# --- the follow-through computation ----------------------------------------


def test_the_follow_digest_counts_the_quieted():
    digest = follow_through_digest([QUIETED, ESCALATED, LAPSED], now=NOW)
    assert digest["worked_count"] == 1
    assert digest["quieted_count"] == 1
    assert [e["submission_id"] for e in digest["quieted"]] == ["OLD-Q"]
    assert digest["escalated_count"] == 2


# --- the ops digest counts ---------------------------------------------------


def test_the_digest_counts_carry_the_worked_response():
    digest = ops_digest([QUIETED, ESCALATED, LAPSED], now=NOW)
    counts = digest["counts"]
    assert counts["follow_through_worked"] == 1
    assert counts["follow_through_quieted"] == 1
    assert counts["follow_through_escalated"] == 2
    # Effort is not attention: the total is exactly the sum of
    # the six worklist counts, worked counts nowhere in it.
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
    # And the plain headline speaks the problem, not the effort.
    assert "worked" not in digest["headline"]


def test_a_store_with_no_worked_notes_counts_zero():
    digest = ops_digest([ESCALATED], now=NOW)
    assert digest["counts"]["follow_through_worked"] == 0
    assert digest["counts"]["follow_through_quieted"] == 0


# --- the snapshots ------------------------------------------------------------


def test_a_snapshot_keeps_the_worked_counts():
    digest = ops_digest([QUIETED, ESCALATED], now=NOW)
    snapshot = build_digest_snapshot(digest, "snap-0001", "Uma")
    assert snapshot["counts"]["follow_through_worked"] == 1
    assert snapshot["counts"]["follow_through_quieted"] == 1
    assert snapshot["counts"]["follow_through_escalated"] == 1


def test_the_trend_reads_the_response_beside_the_problem():
    first = build_digest_snapshot(
        ops_digest([ESCALATED], now=NOW), "snap-0001", "Uma"
    )
    second = build_digest_snapshot(
        ops_digest([QUIETED, ESCALATED, LAPSED], now=NOW),
        "snap-0002",
        "Uma",
    )
    trend = digest_trend([first, second])
    assert trend["deltas"]["follow_through_worked"] == {
        "first": 0,
        "latest": 1,
        "change": 1,
        "since_previous": 1,
    }
    assert trend["deltas"]["follow_through_quieted"]["latest"] == 1
    series = {row["snapshot_id"]: row["counts"] for row in trend["series"]}
    assert series["snap-0001"]["follow_through_worked"] == 0
    assert series["snap-0002"]["follow_through_quieted"] == 1


def test_a_snapshot_predating_the_counts_reads_them_as_zero():
    # A kept snapshot from before round 17 has no worked keys;
    # the movement read treats the absence as the snapshot's
    # fact (zero), and the headline speaks the response moving.
    old = {
        "snapshot_id": "snap-0001",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Uma",
        "headline": "Ops digest: …",
        "counts": {"follow_through_escalated": 1},
        "sections": {},
    }
    digest = ops_digest([QUIETED], now=NOW, latest_snapshot=old)
    movement = digest["movement"]
    assert movement["counts"]["follow_through_worked"] == {
        "snapshot": 0,
        "current": 1,
        "change": 1,
        "direction": "up",
    }
    assert movement["counts"]["follow_through_quieted"]["current"] == 1
    assert "worked follow-through notes up 1" in digest["headline"]
