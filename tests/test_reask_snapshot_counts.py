"""Round 23: re-ask counts ride the digest snapshots.

Round 22 counted the proposal re-asks in /metrics — a
point-in-time read of the proposals document. The governance
question the family raises (does the prune-then-re-ask loop
converge, or circle?) is a trend question, and trends live in
the kept digest snapshots: the drafting families have ridden
``DIGEST_COUNT_KEYS`` since rounds 16 and 20 on the advisory
discipline, and the re-ask family's headline counts
(decided / reversed / repeated) now ride the same keys, the
full aggregate along in the digest body as
``proposal_reasks``. Never in ``total_attention`` — a re-ask
decided is governance history, not work outstanding.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    digest_trend,
    ops_digest,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def _marker() -> dict:
    return {
        "status": "rejected",
        "proposal_id": "PROP-0001",
        "decided_at": "2026-09-20T09:00:00+00:00",
        "decided_by": "First Refuser",
        "pruned_at": "2026-10-01T09:00:00+00:00",
        "pruned_by": "Uma Pruner",
    }


def _proposal(status: str, *, reask: bool, kind: str = "sla_budget") -> dict:
    proposal = {"kind": kind, "status": status, "dimension": "moderate"}
    if reask:
        proposal["previously_decided"] = _marker()
    return proposal


def _proposals() -> list[dict]:
    return [
        _proposal("draft", reask=True),
        _proposal("accepted", reask=True),
        _proposal("accepted", reask=True, kind="diff_trigger_threshold"),
        _proposal("rejected", reask=True),
        _proposal("withdrawn", reask=True),
        _proposal("accepted", reask=False),
    ]


# --- the digest counts -------------------------------------------------------


def test_the_digest_counts_carry_the_reask_outcomes():
    digest = ops_digest([], _proposals(), now=NOW)
    counts = digest["counts"]
    # Four re-asks decided (withdrawal counts as decided),
    # two second acceptances reversed the pruned refusal, one
    # second rejection repeated it. The first-ask acceptance
    # is not a re-ask and counts nowhere here.
    assert counts["proposal_reasks_decided"] == 4
    assert counts["proposal_reasks_reversed"] == 2
    assert counts["proposal_reasks_repeated"] == 1


def test_the_full_family_rides_the_digest_body():
    digest = ops_digest([], _proposals(), now=NOW)
    family = digest["proposal_reasks"]
    assert family["reasks"] == 5
    assert family["open"] == 1
    assert family["withdrawn"] == 1
    assert family["by_kind"] == {
        "diff_trigger_threshold": 1,
        "sla_budget": 4,
    }


def test_no_proposals_reads_as_zero_reask_counts():
    digest = ops_digest([], [], now=NOW)
    counts = digest["counts"]
    assert counts["proposal_reasks_decided"] == 0
    assert counts["proposal_reasks_reversed"] == 0
    assert counts["proposal_reasks_repeated"] == 0
    assert digest["proposal_reasks"]["reasks"] == 0


def test_reasks_never_enter_total_attention():
    bare = ops_digest([], [], now=NOW)
    with_reasks = ops_digest([], _proposals(), now=NOW)
    assert (
        with_reasks["counts"]["total_attention"]
        == bare["counts"]["total_attention"]
        == 0
    )


# --- the snapshots keep them, the trend reads them ----------------------------


def test_a_kept_snapshot_carries_the_reask_counts():
    digest = ops_digest([], _proposals(), now=NOW)
    snapshot = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert snapshot["counts"]["proposal_reasks_decided"] == 4
    assert snapshot["counts"]["proposal_reasks_reversed"] == 2
    assert snapshot["counts"]["proposal_reasks_repeated"] == 1


def test_the_trend_moves_on_the_reask_counts():
    before = build_digest_snapshot(
        ops_digest([], [], now=NOW), "snap-0001", "Ops Lead"
    )
    after = build_digest_snapshot(
        ops_digest([], _proposals(), now=NOW), "snap-0002", "Ops Lead"
    )
    trend = digest_trend([before, after])
    assert trend["deltas"]["proposal_reasks_decided"] == {
        "first": 0,
        "latest": 4,
        "change": 4,
        "since_previous": 4,
    }
    assert trend["deltas"]["proposal_reasks_reversed"]["latest"] == 2
    assert trend["deltas"]["proposal_reasks_repeated"]["latest"] == 1


def test_a_snapshot_kept_before_the_counts_existed_reads_zero():
    # Snapshots kept before this round carry no re-ask keys;
    # the trend reads their absence as zero, never an error —
    # the same honesty the movement block applies.
    legacy = {
        "snapshot_id": "snap-0000",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Ops Lead",
        "counts": {"stuck_dispatches": 1},
        "sections": {},
    }
    after = build_digest_snapshot(
        ops_digest([], _proposals(), now=NOW), "snap-0002", "Ops Lead"
    )
    trend = digest_trend([legacy, after])
    assert trend["series"][0]["counts"]["proposal_reasks_decided"] == 0
    assert trend["deltas"]["proposal_reasks_decided"]["change"] == 4


def test_the_live_digest_reads_against_a_legacy_snapshot():
    legacy = {
        "snapshot_id": "snap-0000",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Ops Lead",
        "counts": {"stuck_dispatches": 0},
        "sections": {},
    }
    digest = ops_digest([], _proposals(), now=NOW, latest_snapshot=legacy)
    row = digest["movement"]["counts"]["proposal_reasks_reversed"]
    assert row == {
        "snapshot": 0,
        "current": 2,
        "change": 2,
        "direction": "up",
    }
    assert "re-asks reversing the first decision up 2" in digest["headline"]
