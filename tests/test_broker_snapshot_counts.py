"""Round 20: broker drafting counts in the digest snapshots.

The drafting-repair family had three surfaces, but only two
rode the digest's counts: the letter's advisories and the
status note's repairs and refusals trended across the kept
snapshots, while the broker quality report's drafting outcomes
— drafted on read, recorded as events — lived only in the
point-in-time /metrics aggregate. The digest counts now carry
the broker source's drafted / repaired / refused too, computed
from the drafting events the service passes in, so a kept
snapshot trends all three drafting surfaces on the same
advisory discipline: a quality signal, never in
total_attention, never in the headline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    compute_metrics,
    digest_trend,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, SubmissionFacts, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
BROKER = "Harborline Brokers"


def _event(outcome: str, **overrides) -> dict:
    event = {
        "broker": BROKER,
        "drafted_at": "2026-10-19T09:00:00+00:00",
        "outcome": outcome,
        "repair_families": [],
        "failed_checks": [],
        "steered": False,
    }
    event.update(overrides)
    return event


_EVENTS = [
    _event("clean"),
    _event("repaired", repair_families=["forward_motion"]),
    _event(
        "refused", failed_checks=["no_firm_premium"], steered=False
    ),
]


def _record(sid: str, broker: str = BROKER, **facts) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            facts=SubmissionFacts(**facts),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": broker,
            "submitted_at": "2026-09-01",
        },
    )


# --- the digest counts -----------------------------------------------------------------


def test_the_digest_counts_carry_the_broker_drafting_aggregates():
    digest = ops_digest([], now=NOW, broker_drafting_events=_EVENTS)
    counts = digest["counts"]
    assert counts["broker_quality_drafted"] == 3
    assert counts["broker_quality_repaired"] == 1
    assert counts["broker_quality_refusals"] == 1
    # The full family rides along — the same computation /metrics
    # reports nested in its repair family, so the two surfaces
    # cannot disagree.
    assert digest["broker_quality_repairs"] == compute_metrics(
        [], broker_drafting_events=_EVENTS
    )["status_note_repairs"]["broker_quality"]
    assert digest["broker_quality_repairs"]["by_family"] == {
        "forward_motion": 1
    }
    # A quality signal, not a worklist: attention is untouched
    # and the headline does not speak it.
    assert counts["total_attention"] == 0
    assert "broker" not in digest["headline"]


def test_no_drafting_events_read_as_zero_not_absent():
    digest = ops_digest([], now=NOW)
    counts = digest["counts"]
    assert counts["broker_quality_drafted"] == 0
    assert counts["broker_quality_repaired"] == 0
    assert counts["broker_quality_refusals"] == 0
    assert digest["broker_quality_repairs"]["reports_drafted"] == 0
    assert digest["broker_quality_repairs"]["steer_success_rate"] is None


def test_a_snapshot_keeps_the_broker_counts_and_the_trend_reads_them():
    first = build_digest_snapshot(
        ops_digest([], now=NOW, broker_drafting_events=_EVENTS[:1]),
        "snap-0001",
        "Uma",
    )
    second = build_digest_snapshot(
        ops_digest([], now=NOW, broker_drafting_events=_EVENTS),
        "snap-0002",
        "Uma",
    )
    assert first["counts"]["broker_quality_drafted"] == 1
    assert second["counts"]["broker_quality_drafted"] == 3
    assert second["counts"]["broker_quality_refusals"] == 1
    trend = digest_trend([first, second])
    drafted = trend["deltas"]["broker_quality_drafted"]
    assert drafted == {
        "first": 1,
        "latest": 3,
        "change": 2,
        "since_previous": 2,
    }
    refused = trend["deltas"]["broker_quality_refusals"]
    assert refused["change"] == 1


def test_a_snapshot_kept_before_the_counts_reads_them_as_zero():
    legacy = {
        "snapshot_id": "snap-0001",
        "taken_at": "2026-10-01T09:00:00+00:00",
        "taken_by": "Uma",
        "headline": "Ops digest: nothing needs attention.",
        "counts": {"stuck_dispatches": 0, "total_attention": 0},
        "sections": {},
    }
    current = build_digest_snapshot(
        ops_digest([], now=NOW, broker_drafting_events=_EVENTS),
        "snap-0002",
        "Uma",
    )
    trend = digest_trend([legacy, current])
    # Absence in the older snapshot is a fact about the snapshot:
    # the count reads as zero there, never invented.
    assert trend["series"][0]["counts"]["broker_quality_drafted"] == 0
    assert trend["deltas"]["broker_quality_drafted"]["first"] == 0
    assert trend["deltas"]["broker_quality_drafted"]["latest"] == 3


# --- through the service -----------------------------------------------------------------


def _service_with(*records: ApprovalRecord) -> SubmissionService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


def test_the_service_digest_and_its_snapshots_carry_the_counts():
    service = _service_with(
        _record(
            "Q-1",
            broker="Wording Brokers",
            completeness_score=0.8,
            missing_elements=[
                "we will proceed with the file once audits land"
            ],
        ),
        _record("Q-2", broker="Wording Brokers", completeness_score=0.9),
        _record(
            "Q-3",
            broker="Premium Brokers",
            completeness_score=0.8,
            missing_elements=["premium at $12,000 for the term"],
        ),
        _record("Q-4", broker="Premium Brokers", completeness_score=0.9),
        _record("Q-5", completeness_score=0.7, discrepancy_count=1),
        _record(
            "Q-6", completeness_score=0.9, missing_elements=["loss runs"]
        ),
    )
    service.broker_quality_report("Wording Brokers")
    service.broker_quality_report("Premium Brokers")
    digest = service.ops_digest()
    assert digest["counts"]["broker_quality_drafted"] == 2
    assert digest["counts"]["broker_quality_repaired"] == 1
    assert digest["counts"]["broker_quality_refusals"] == 1
    # The digest's family and /metrics' family are one computation
    # over the same events at the same moment.
    assert digest["broker_quality_repairs"] == service.metrics()[
        "status_note_repairs"
    ]["broker_quality"]

    snapshot = service.snapshot_ops_digest("Uma")
    assert snapshot["counts"]["broker_quality_drafted"] == 2

    service.broker_quality_report(BROKER)
    service.snapshot_ops_digest("Uma")
    trend = service.ops_digest_trend()
    assert trend["deltas"]["broker_quality_drafted"]["latest"] == 3
    assert trend["deltas"]["broker_quality_drafted"]["change"] == 1
