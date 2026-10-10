"""Round 16: advisory counts in the digest snapshots.

The advisory trend was a point-in-time aggregate in /metrics —
today's drafting-quality rate, with no history. The digest
snapshots already trend the operations counts across kept
snapshots; the digest's counts now carry the advisory
aggregates too (runs that drafted advisories, runs the steered
redraft repaired), so a kept snapshot keeps them and the trend
reads drafting quality across the kept mornings — drift
between two kept snapshots, not just today's rate. The counts
are a quality signal: they never enter total_attention and
never reach the headline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    compute_metrics,
    digest_trend,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _record(
    sid: str, advisories: list[dict] | None = None, repaired: bool = False
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            letter_advisories=advisories or [],
            advisory_repaired=repaired,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-10-19",
        },
    )


_ADV = [{"family": "forward_motion", "sentence": "we will proceed"}]


def test_the_digest_counts_carry_the_advisory_aggregates():
    records = [
        _record("A-1", advisories=_ADV, repaired=True),
        _record("A-2", advisories=_ADV + _ADV),
        _record("A-3"),
    ]
    digest = ops_digest(records, now=NOW)
    assert digest["counts"]["advisory_runs_with"] == 2
    assert digest["counts"]["advisory_repaired_runs"] == 1
    # The full family rides along — the same computation /metrics
    # reports, so the two surfaces cannot disagree.
    assert digest["letter_advisories"] == compute_metrics(records)[
        "letter_advisories"
    ]
    assert digest["letter_advisories"]["total"] == 3
    # A quality signal, not a worklist: attention is untouched.
    assert "advisory" not in digest["headline"]


def test_a_snapshot_keeps_the_advisory_counts():
    digest = ops_digest(
        [_record("A-1", advisories=_ADV, repaired=True)], now=NOW
    )
    snapshot = build_digest_snapshot(digest, "snap-0001", "Uma")
    assert snapshot["counts"]["advisory_runs_with"] == 1
    assert snapshot["counts"]["advisory_repaired_runs"] == 1


def test_the_trend_reads_drafting_quality_across_kept_snapshots():
    first = build_digest_snapshot(
        ops_digest([_record("A-1", advisories=_ADV)], now=NOW),
        "snap-0001",
        "Uma",
    )
    second = build_digest_snapshot(
        ops_digest(
            [
                _record("A-1", advisories=_ADV, repaired=True),
                _record("A-2", advisories=_ADV),
                _record("A-3", advisories=_ADV),
            ],
            now=NOW,
        ),
        "snap-0002",
        "Uma",
    )
    trend = digest_trend([first, second])
    runs = trend["deltas"]["advisory_runs_with"]
    assert runs == {
        "first": 1,
        "latest": 3,
        "change": 2,
        "since_previous": 2,
    }
    repaired = trend["deltas"]["advisory_repaired_runs"]
    assert repaired["first"] == 0
    assert repaired["latest"] == 1
    assert repaired["change"] == 1
    series = {row["snapshot_id"]: row["counts"] for row in trend["series"]}
    assert series["snap-0001"]["advisory_runs_with"] == 1
    assert series["snap-0002"]["advisory_repaired_runs"] == 1


def test_a_snapshot_predating_the_counts_reads_them_as_zero():
    # A kept snapshot from before round 16 has no advisory keys;
    # the movement read treats the absence as the snapshot's
    # fact (zero), never as an error or an invention.
    old = {
        "snapshot_id": "snap-0001",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Uma",
        "headline": "Ops digest: …",
        "counts": {"follow_through_escalated": 0},
        "sections": {},
    }
    digest = ops_digest(
        [_record("A-1", advisories=_ADV)], now=NOW, latest_snapshot=old
    )
    movement = digest["movement"]
    assert movement["counts"]["advisory_runs_with"] == {
        "snapshot": 0,
        "current": 1,
        "change": 1,
        "direction": "up",
    }
    assert "runs drafting letter advisories up 1" in digest["headline"]


def test_metrics_advisory_family_is_unchanged_by_the_extraction():
    records = [
        _record("A-1", advisories=_ADV, repaired=True),
        _record("A-2", advisories=[{"family": "soft_time", "sentence": "shortly"}]),
        _record("A-3"),
    ]
    family = compute_metrics(records)["letter_advisories"]
    assert family == {
        "runs_with_advisories": 2,
        "total": 2,
        "per_run": 0.667,
        "by_family": {"forward_motion": 1, "soft_time": 1},
        "repaired_runs": 1,
        "clear_rate": 0.5,
    }
