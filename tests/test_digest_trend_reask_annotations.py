"""Round 24: the digest trend explains a re-ask-count step.

Round 23 put the re-ask counts on the digest snapshots;
they are computed from the proposals document as it stands
now, while a kept snapshot froze the counts of its moment.
When retention prunes decided proposals between two
snapshots — and some of what left carried a
``previously_decided`` marker — the later snapshot's re-ask
counts step *down*, and an unannotated trend reads that as
the prune-then-re-ask loop converging. The trend now
consults the proposals document's own prune record (whose
entries count the markers they removed, ``pruned_reasks``)
and annotates each re-ask step a prune inside the step's
window explains — the round-21 broker annotation's honesty,
one document over. A prune that removed no markers explains
nothing; a step down no prune explains is left unannotated.
"""

from __future__ import annotations

from underwriting_agent.insights import DIGEST_SNAPSHOTS_KEY, digest_trend
from underwriting_agent.proposals import (
    PROPOSALS_KEY,
    proposal_prune_entry,
)
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _snapshot(sid: str, taken_at: str, **counts) -> dict:
    base = {"proposal_reasks_decided": 0, "total_attention": 0}
    base.update(counts)
    return {
        "snapshot_id": sid,
        "taken_at": taken_at,
        "taken_by": "Ops Lead",
        "headline": "Ops digest: …",
        "counts": base,
        "sections": {},
    }


def _prune(
    pruned_at: str,
    pruned_by: str = "Uma",
    count: int = 5,
    reasks: int = 3,
) -> dict:
    return {
        "pruned_at": pruned_at,
        "pruned_by": pruned_by,
        "requested_keep": 10,
        "pruned_count": count,
        "pruned_ids": [f"PROP-{i:04d}" for i in range(1, count + 1)],
        "pruned_by_status": {"rejected": count},
        "pruned_rejections": [],
        "pruned_reasks": reasks,
        "surviving_first": "PROP-0100",
        "surviving_last": "PROP-0110",
    }


FIRST = _snapshot(
    "snap-0001", "2026-10-05T09:00:00+00:00",
    proposal_reasks_decided=6,
    proposal_reasks_reversed=4,
    proposal_reasks_repeated=2,
)
SECOND = _snapshot(
    "snap-0002", "2026-10-12T09:00:00+00:00",
    proposal_reasks_decided=3,
    proposal_reasks_reversed=2,
    proposal_reasks_repeated=1,
)
IN_WINDOW = _prune("2026-10-08T09:00:00+00:00")


# --- the prune entry counts the markers it removes -------------------------------


def test_the_proposals_prune_entry_counts_pruned_reasks():
    marker = {
        "proposal_id": "PROP-0001",
        "direction": "raise",
        "decided_by": "Uma",
        "decided_at": "2026-09-20T09:00:00+00:00",
        "pruned_at": "2026-10-01T09:00:00+00:00",
        "pruned_by": "Ravi",
    }
    pruned = [
        {
            "proposal_id": "PROP-0001",
            "status": "rejected",
            "previously_decided": marker,
        },
        {"proposal_id": "PROP-0002", "status": "accepted"},
        {
            "proposal_id": "PROP-0003",
            "status": "rejected",
            "previously_decided": marker,
        },
    ]
    entry = proposal_prune_entry(
        actor="Uma",
        keep=1,
        pruned=pruned,
        kept=[],
        pruned_at="2026-10-06T09:00:00+00:00",
    )
    assert entry["pruned_reasks"] == 2
    assert entry["pruned_count"] == 3


# --- the explained step ------------------------------------------------------------


def test_a_prune_inside_the_window_annotates_the_reask_step_down():
    trend = digest_trend([FIRST, SECOND], proposal_prunes=[IN_WINDOW])
    assert len(trend["annotations"]) == 3
    by_count = {a["count"]: a for a in trend["annotations"]}
    note = by_count["proposal_reasks_decided"]
    assert note["from_snapshot"] == "snap-0001"
    assert note["to_snapshot"] == "snap-0002"
    assert note["change"] == -3
    assert note["pruned_reasks"] == 3
    assert note["prunes"] == [
        {
            "pruned_at": "2026-10-08T09:00:00+00:00",
            "pruned_by": "Uma",
            "pruned_count": 5,
            "pruned_reasks": 3,
        }
    ]
    # The prose names the prune and refuses the convergence
    # reading.
    assert "Uma" in note["note"]
    assert "2026-10-08" in note["note"]
    assert "retention, not the loop converging" in note["note"]
    # The deltas themselves are untouched.
    assert trend["deltas"]["proposal_reasks_decided"]["change"] == -3


def test_each_reask_count_stepping_down_is_annotated_apart():
    second = _snapshot(
        "snap-0002", "2026-10-12T09:00:00+00:00",
        proposal_reasks_decided=3,
        proposal_reasks_reversed=4,  # held: no step, no note
        proposal_reasks_repeated=1,
    )
    trend = digest_trend([FIRST, second], proposal_prunes=[IN_WINDOW])
    assert {a["count"] for a in trend["annotations"]} == {
        "proposal_reasks_decided",
        "proposal_reasks_repeated",
    }


def test_a_prune_that_removed_no_markers_explains_nothing():
    plain = _prune("2026-10-08T09:00:00+00:00", count=5, reasks=0)
    trend = digest_trend([FIRST, SECOND], proposal_prunes=[plain])
    assert trend["annotations"] == []


def test_a_legacy_prune_entry_without_the_count_explains_nothing():
    # Entries written before pruned_reasks was recorded read
    # as zero markers: the record did not keep the number,
    # and the annotation does not reconstruct it.
    legacy = _prune("2026-10-08T09:00:00+00:00")
    del legacy["pruned_reasks"]
    trend = digest_trend([FIRST, SECOND], proposal_prunes=[legacy])
    assert trend["annotations"] == []


def test_a_step_down_with_no_prune_in_its_window_is_not_annotated():
    before = _prune("2026-10-01T09:00:00+00:00")
    after = _prune("2026-10-19T09:00:00+00:00")
    trend = digest_trend([FIRST, SECOND], proposal_prunes=[before, after])
    assert trend["annotations"] == []


def test_a_step_up_is_never_explained_by_a_prune():
    trend = digest_trend([SECOND, FIRST], proposal_prunes=[IN_WINDOW])
    assert trend["annotations"] == []


def test_broker_counts_are_not_explained_by_a_proposals_prune():
    first = _snapshot(
        "snap-0001", "2026-10-05T09:00:00+00:00",
        broker_quality_drafted=20,
    )
    second = _snapshot(
        "snap-0002", "2026-10-12T09:00:00+00:00",
        broker_quality_drafted=12,
    )
    trend = digest_trend([first, second], proposal_prunes=[IN_WINDOW])
    assert trend["annotations"] == []


def test_no_prunes_passed_means_no_annotations():
    assert digest_trend([FIRST, SECOND])["annotations"] == []


# --- the service wires the proposals document's record in --------------------------


def test_the_service_trend_reads_the_proposals_documents_prunes():
    store = InMemoryStore()
    store.save_document(
        DIGEST_SNAPSHOTS_KEY, {"snapshots": [FIRST, SECOND]}
    )
    store.save_document(
        PROPOSALS_KEY, {"proposals": [], "prunes": [IN_WINDOW]}
    )
    service = SubmissionService(store=store, checkpointer=False)
    trend = service.ops_digest_trend()
    assert {a["count"] for a in trend["annotations"]} == {
        "proposal_reasks_decided",
        "proposal_reasks_reversed",
        "proposal_reasks_repeated",
    }
