"""Round 13: the trend-aware queue — new fire vs old fire.

The digest has read itself against its snapshots since round 12,
but the queue — the screen the work actually happens on — showed
breaches as a flat state: a case that crossed its budget this
morning looked exactly like one that has been burning for a
week. ``queue_aging`` now takes the kept digest snapshots and
flags each breached item against the latest one's breached ids:
``new`` (not breached when the picture was kept) or
``continuing`` (already burning), with the movement — new,
continuing, and cleared ids — on the view. No snapshot, no
flags: direction is absent, never invented.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import queue_aging
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def _awaiting(sid: str, submitted_at: str, tier: str = "moderate") -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _decided(sid: str) -> ApprovalRecord:
    record = _awaiting(sid, "2026-09-01")
    record.result = record.result.model_copy(
        update={"status": "approved", "decided_at": "2026-10-09"}
    )
    return record


def _snapshot(breached_ids: list[str]) -> dict:
    return {
        "snapshot_id": "snap-0001",
        "taken_at": "2026-10-08T09:00:00+00:00",
        "taken_by": "Ops Lead",
        "headline": "Ops digest: ...",
        "counts": {},
        "sections": {"breached_queue": breached_ids},
    }


def _records() -> list[ApprovalRecord]:
    # Moderate budget is 5 days; at NOW (Oct 10) a Sep 20
    # submission is 20 days in — breached. A Oct 8 submission is
    # within budget and never enters the comparison.
    return [
        _awaiting("OLD-FIRE", "2026-09-20"),
        _awaiting("NEW-FIRE", "2026-09-20"),
        _awaiting("WITHIN", "2026-10-08"),
        _decided("CLEARED-ONE"),
    ]


# ---------------------------------------------------------------------------
# The flags and the movement block
# ---------------------------------------------------------------------------


def test_breaches_are_flagged_new_or_continuing_against_the_snapshot():
    view = queue_aging(
        _records(), NOW, digest_snapshots=[_snapshot(["OLD-FIRE", "CLEARED-ONE"])]
    )
    flags = {
        item["submission_id"]: item["breach_since_snapshot"]
        for item in view["breaches"]
    }
    assert flags == {"OLD-FIRE": "continuing", "NEW-FIRE": "new"}
    movement = view["breach_movement"]
    assert movement["since_snapshot"] == "snap-0001"
    assert movement["snapshot_taken_by"] == "Ops Lead"
    assert movement["new_breaches"] == ["NEW-FIRE"]
    assert movement["continuing_breaches"] == ["OLD-FIRE"]
    # CLEARED-ONE was breached in the snapshot; it has since been
    # decided, so it is cleared — not continuing, not deleted
    # from the story.
    assert movement["cleared_breaches"] == ["CLEARED-ONE"]


def test_the_flag_rides_every_projection_of_the_item():
    view = queue_aging(
        _records(), NOW, digest_snapshots=[_snapshot(["OLD-FIRE"])]
    )
    bucketed = {
        item["submission_id"]: item
        for bucket in view["buckets"]
        for item in bucket["submissions"]
    }
    assert bucketed["NEW-FIRE"]["breach_since_snapshot"] == "new"
    assert bucketed["WITHIN"]["breach_since_snapshot"] is None


def test_without_a_snapshot_there_is_no_movement_and_no_flag():
    view = queue_aging(_records(), NOW)
    assert view["breach_movement"] is None
    assert all(
        item["breach_since_snapshot"] is None for item in view["breaches"]
    )
    empty = queue_aging(_records(), NOW, digest_snapshots=[])
    assert empty["breach_movement"] is None


def test_the_latest_snapshot_is_the_baseline():
    older = _snapshot(["OLD-FIRE", "NEW-FIRE"])
    older["snapshot_id"] = "snap-0001"
    latest = _snapshot(["OLD-FIRE"])
    latest["snapshot_id"] = "snap-0002"
    view = queue_aging(_records(), NOW, digest_snapshots=[older, latest])
    assert view["breach_movement"]["since_snapshot"] == "snap-0002"
    assert view["breach_movement"]["new_breaches"] == ["NEW-FIRE"]


# ---------------------------------------------------------------------------
# Through the service: the digest snapshot is the baseline
# ---------------------------------------------------------------------------


def test_service_queue_reads_against_the_kept_digest_snapshot():
    store = InMemoryStore()
    # Submitted long ago, so the case is breached under any real
    # clock the suite runs on.
    store.save(_awaiting("FIRST-FIRE", "2026-01-05"))
    service = SubmissionService(store=store, checkpointer=False)
    service.snapshot_ops_digest("Ops Lead")

    store.save(_awaiting("SECOND-FIRE", "2026-01-06"))
    view = service.queue()
    movement = view["breach_movement"]
    assert movement is not None
    assert movement["new_breaches"] == ["SECOND-FIRE"]
    assert movement["continuing_breaches"] == ["FIRST-FIRE"]
    assert movement["cleared_breaches"] == []
    flags = {
        item["submission_id"]: item["breach_since_snapshot"]
        for item in view["breaches"]
    }
    assert flags == {"FIRST-FIRE": "continuing", "SECOND-FIRE": "new"}
