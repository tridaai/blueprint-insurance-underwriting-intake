"""Round 21: the digest trend explains a broker-count step.

The broker drafting counts are computed from the drafting
events as they stand now, while a kept digest snapshot froze
the counts of its moment. When retention prunes the events
document between two snapshots, the later snapshot's broker
counts step *down* — and an unannotated trend reads that as a
drafting collapse. The trend now consults the drafting
document's prune record and annotates each broker-count step
a prune inside the step's window explains: the step is
retention, named with the prune's own facts, not a fall in
drafting quality. A step down no prune explains is left
unannotated — real movement is not softened.
"""

from __future__ import annotations

from underwriting_agent.insights import (
    BROKER_QUALITY_DRAFTING_KEY,
    DIGEST_SNAPSHOTS_KEY,
    digest_trend,
)
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _snapshot(sid: str, taken_at: str, **counts) -> dict:
    base = {"broker_quality_drafted": 0, "total_attention": 0}
    base.update(counts)
    return {
        "snapshot_id": sid,
        "taken_at": taken_at,
        "taken_by": "Ops Lead",
        "headline": "Ops digest: …",
        "counts": base,
        "sections": {},
    }


def _prune(pruned_at: str, pruned_by: str = "Uma", count: int = 9) -> dict:
    return {
        "pruned_at": pruned_at,
        "pruned_by": pruned_by,
        "requested_keep": 50,
        "pruned_count": count,
        "pruned_oldest": "2026-09-01T09:00:00+00:00",
        "pruned_newest": "2026-09-20T09:00:00+00:00",
        "surviving_first": "2026-09-21T09:00:00+00:00",
        "surviving_last": "2026-10-19T09:00:00+00:00",
    }


FIRST = _snapshot(
    "snap-0001", "2026-10-05T09:00:00+00:00", broker_quality_drafted=20
)
SECOND = _snapshot(
    "snap-0002", "2026-10-12T09:00:00+00:00", broker_quality_drafted=12
)
IN_WINDOW = _prune("2026-10-08T09:00:00+00:00")


# --- the explained step ------------------------------------------------------------


def test_a_prune_inside_the_window_annotates_the_step_down():
    trend = digest_trend([FIRST, SECOND], drafting_prunes=[IN_WINDOW])
    assert len(trend["annotations"]) == 1
    note = trend["annotations"][0]
    assert note["count"] == "broker_quality_drafted"
    assert note["from_snapshot"] == "snap-0001"
    assert note["to_snapshot"] == "snap-0002"
    assert note["change"] == -8
    assert note["pruned_events"] == 9
    assert note["prunes"] == [
        {
            "pruned_at": "2026-10-08T09:00:00+00:00",
            "pruned_by": "Uma",
            "pruned_count": 9,
        }
    ]
    # The prose names the prune and refuses the collapse reading.
    assert "Uma" in note["note"]
    assert "2026-10-08" in note["note"]
    assert "retention, not a drafting collapse" in note["note"]
    # The deltas themselves are untouched — annotation is beside
    # the arithmetic, never a rewrite of it.
    assert trend["deltas"]["broker_quality_drafted"]["change"] == -8


def test_each_broker_count_stepping_down_is_annotated_apart():
    first = _snapshot(
        "snap-0001",
        "2026-10-05T09:00:00+00:00",
        broker_quality_drafted=20,
        broker_quality_repaired=6,
        broker_quality_refusals=4,
    )
    second = _snapshot(
        "snap-0002",
        "2026-10-12T09:00:00+00:00",
        broker_quality_drafted=12,
        broker_quality_repaired=6,
        broker_quality_refusals=1,
    )
    trend = digest_trend([first, second], drafting_prunes=[IN_WINDOW])
    # Repaired held flat: no annotation for it.
    assert {a["count"] for a in trend["annotations"]} == {
        "broker_quality_drafted",
        "broker_quality_refusals",
    }


def test_two_prunes_in_one_window_are_both_named():
    prunes = [
        _prune("2026-10-07T09:00:00+00:00", count=4),
        _prune("2026-10-10T09:00:00+00:00", pruned_by="Ravi", count=5),
    ]
    trend = digest_trend([FIRST, SECOND], drafting_prunes=prunes)
    note = trend["annotations"][0]
    assert note["pruned_events"] == 9
    assert "Uma" in note["note"] and "Ravi" in note["note"]


# --- the unexplained step stays unexplained ------------------------------------------


def test_a_step_down_with_no_prune_in_its_window_is_not_annotated():
    before = _prune("2026-10-01T09:00:00+00:00")  # before snap-0001
    after = _prune("2026-10-19T09:00:00+00:00")  # after snap-0002
    trend = digest_trend([FIRST, SECOND], drafting_prunes=[before, after])
    assert trend["annotations"] == []


def test_a_step_up_is_never_explained_by_a_prune():
    trend = digest_trend([SECOND, FIRST], drafting_prunes=[IN_WINDOW])
    assert trend["annotations"] == []


def test_only_broker_counts_are_prune_explainable():
    first = _snapshot(
        "snap-0001",
        "2026-10-05T09:00:00+00:00",
        advisory_runs_with=10,
        total_attention=7,
    )
    second = _snapshot(
        "snap-0002",
        "2026-10-12T09:00:00+00:00",
        advisory_runs_with=3,
        total_attention=2,
    )
    trend = digest_trend([first, second], drafting_prunes=[IN_WINDOW])
    assert trend["annotations"] == []


def test_no_prunes_passed_means_no_annotations():
    assert digest_trend([FIRST, SECOND])["annotations"] == []
    assert digest_trend([])["annotations"] == []


def test_each_step_is_judged_in_its_own_window():
    third = _snapshot(
        "snap-0003", "2026-10-19T09:00:00+00:00",
        broker_quality_drafted=4,
    )
    # The prune sits between snap-0002 and snap-0003 only.
    late = _prune("2026-10-15T09:00:00+00:00", count=8)
    trend = digest_trend(
        [FIRST, SECOND, third], drafting_prunes=[late]
    )
    assert len(trend["annotations"]) == 1
    assert trend["annotations"][0]["from_snapshot"] == "snap-0002"
    assert trend["annotations"][0]["to_snapshot"] == "snap-0003"


# --- the service wires the drafting document's record in ------------------------------


def test_the_service_trend_reads_the_drafting_documents_prunes():
    store = InMemoryStore()
    store.save_document(
        DIGEST_SNAPSHOTS_KEY, {"snapshots": [FIRST, SECOND]}
    )
    store.save_document(
        BROKER_QUALITY_DRAFTING_KEY,
        {"events": [], "prunes": [IN_WINDOW]},
    )
    service = SubmissionService(store=store, checkpointer=False)
    trend = service.ops_digest_trend()
    assert len(trend["annotations"]) == 1
    assert trend["annotations"][0]["count"] == "broker_quality_drafted"
