"""Round 12: the trend-aware digest.

Snapshots (round 11) made movement visible between two kept
digests, but the screen an operator actually opens — the live
digest — still read as if it had no past. Now, when a snapshot
exists, the digest reads itself against the latest one: a
``movement`` block with per-count change and direction and
per-section entered/cleared ids, and the headline speaks the
movement in one sentence. No snapshot, no movement — the block
is None rather than a fabricated baseline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import build_digest_snapshot, ops_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _breached_record(sid: str = "LATE-1") -> ApprovalRecord:
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


def _snapshot_of(records, snapshot_id: str = "snap-0001") -> dict:
    digest = ops_digest(records, [], now=NOW)
    return build_digest_snapshot(digest, snapshot_id, "Ops Lead")


def test_without_a_snapshot_the_digest_has_no_movement():
    digest = ops_digest([_breached_record()], [], now=NOW)
    assert digest["movement"] is None
    assert "Since snap-" not in digest["headline"]


def test_movement_reads_counts_and_membership_against_the_snapshot():
    snapshot = _snapshot_of([])  # a quiet day, kept
    digest = ops_digest(
        [_breached_record()], [], now=NOW, latest_snapshot=snapshot
    )
    movement = digest["movement"]
    assert movement["since_snapshot"] == "snap-0001"
    assert movement["snapshot_taken_by"] == "Ops Lead"
    row = movement["counts"]["breached_queue"]
    assert row == {"snapshot": 0, "current": 1, "change": 1, "direction": "up"}
    assert movement["counts"]["total_attention"]["direction"] == "up"
    assert movement["counts"]["stuck_dispatches"]["direction"] == "flat"
    assert movement["sections"]["breached_queue"] == {
        "entered": ["LATE-1"], "cleared": [],
    }
    # The headline speaks it, in the digest's own voice.
    assert "Since snap-0001: breached queue cases up 1" in digest["headline"]
    assert "total attention up 1" in digest["headline"]


def test_movement_reports_clearing_as_down_with_the_ids():
    snapshot = _snapshot_of([_breached_record()])
    digest = ops_digest([], [], now=NOW, latest_snapshot=snapshot)
    movement = digest["movement"]
    row = movement["counts"]["breached_queue"]
    assert row == {"snapshot": 1, "current": 0, "change": -1, "direction": "down"}
    assert movement["sections"]["breached_queue"] == {
        "entered": [], "cleared": ["LATE-1"],
    }
    assert "breached queue cases down 1" in digest["headline"]


def test_no_movement_is_stated_not_silenced():
    records = [_breached_record()]
    snapshot = _snapshot_of(records)
    digest = ops_digest(records, [], now=NOW, latest_snapshot=snapshot)
    assert all(
        row["direction"] == "flat"
        for row in digest["movement"]["counts"].values()
    )
    assert "no movement in the headline counts" in digest["headline"]


def test_a_snapshot_from_before_a_section_existed_reads_as_zero():
    # A pre-round-12 snapshot: no exhausted-breach-notices count
    # or section anywhere in it.
    legacy = {
        "snapshot_id": "snap-0007",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Ops Lead",
        "headline": "Ops digest: …",
        "counts": {"breached_queue": 1, "total_attention": 1},
        "sections": {"breached_queue": ["LATE-1"]},
    }
    digest = ops_digest(
        [_breached_record()], [], now=NOW, latest_snapshot=legacy
    )
    movement = digest["movement"]
    assert movement["counts"]["exhausted_breach_notices"] == {
        "snapshot": 0, "current": 0, "change": 0, "direction": "flat",
    }
    assert movement["counts"]["breached_queue"]["change"] == 0
    assert movement["sections"]["exhausted_breach_notices"] == {
        "entered": [], "cleared": [],
    }


def test_service_digest_turns_trend_aware_once_a_snapshot_exists():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    store = service._get_store()
    # No snapshot yet: the digest has no past to read against.
    assert service.ops_digest()["movement"] is None

    service.snapshot_ops_digest("Ops Lead")
    digest = service.ops_digest()
    assert digest["movement"]["since_snapshot"] == "snap-0001"
    assert "no movement in the headline counts" in digest["headline"]

    store.save(_breached_record())
    digest = service.ops_digest()
    assert digest["movement"]["counts"]["breached_queue"]["change"] == 1
    assert digest["movement"]["sections"]["breached_queue"]["entered"] == [
        "LATE-1"
    ]
    # The next snapshot is taken against the moved digest; its own
    # headline is the digest as read (movement sentence included).
    second = service.snapshot_ops_digest("Ops Lead")
    assert second["counts"]["breached_queue"] == 1
    digest = service.ops_digest()
    assert digest["movement"]["since_snapshot"] == "snap-0002"
    assert digest["movement"]["counts"]["breached_queue"]["change"] == 0
