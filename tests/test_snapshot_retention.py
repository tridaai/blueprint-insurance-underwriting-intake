"""Round 14: snapshot retention — one discipline, both documents.

Digest snapshots and calibration snapshots both accumulate one
document per operator ask, unboundedly: a year of weekly
snapshots becomes an archive nobody trends. A keep-last-N prune
— operator-called, never a timer — is now the only way a kept
snapshot leaves either document, and the prune is recorded on
the document it pruned (who, when, what was asked, what left,
the surviving range), so a gap in a trend explains itself. Ids
run one past the highest number still kept, so a citation names
exactly one kept report forever, pruned or not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import (
    DIGEST_SNAPSHOTS_KEY,
    next_snapshot_id,
    prune_snapshot_list,
)
from underwriting_agent.proposals import CALIBRATION_SNAPSHOTS_KEY
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def _take_digests(service: SubmissionService, n: int) -> list[dict]:
    return [service.snapshot_ops_digest("Ops Lead") for _ in range(n)]


# ---------------------------------------------------------------------------
# The shared computation
# ---------------------------------------------------------------------------


def test_prune_list_keeps_the_newest_in_order():
    snaps = [{"snapshot_id": f"snap-{i:04d}"} for i in range(1, 8)]
    kept, pruned = prune_snapshot_list(snaps, 3)
    assert [s["snapshot_id"] for s in kept] == [
        "snap-0005",
        "snap-0006",
        "snap-0007",
    ]
    assert [s["snapshot_id"] for s in pruned] == [
        "snap-0001",
        "snap-0002",
        "snap-0003",
        "snap-0004",
    ]


def test_prune_list_refuses_to_empty_the_document():
    with pytest.raises(ValueError, match="at least 1"):
        prune_snapshot_list([{"snapshot_id": "snap-0001"}], 0)


def test_next_id_runs_past_the_highest_kept_number_not_the_length():
    snaps = [{"snapshot_id": "snap-0005"}, {"snapshot_id": "snap-0007"}]
    assert next_snapshot_id(snaps, "snap-") == "snap-0008"
    assert next_snapshot_id([], "cal-") == "cal-0001"


# ---------------------------------------------------------------------------
# Digest snapshots
# ---------------------------------------------------------------------------


def test_pruning_digest_snapshots_records_itself_on_the_document():
    service = _service()
    _take_digests(service, 5)
    report = service.prune_ops_digest_snapshots("Ops Lead", 2)

    assert report["kept"] == 2
    assert report["pruned_count"] == 3
    assert report["pruned_ids"] == ["snap-0001", "snap-0002", "snap-0003"]
    assert report["surviving_first"] == "snap-0004"
    assert report["surviving_last"] == "snap-0005"
    assert [s["snapshot_id"] for s in service.ops_digest_snapshots()] == [
        "snap-0004",
        "snap-0005",
    ]

    prunes = service.ops_digest_snapshot_prunes()
    assert len(prunes) == 1
    entry = prunes[0]
    assert entry["pruned_by"] == "Ops Lead"
    assert entry["requested_keep"] == 2
    assert entry["pruned_ids"] == report["pruned_ids"]
    assert entry["surviving_first"] == "snap-0004"
    assert entry["pruned_at"]

    # The trend now reads only the survivors — and says so by
    # their ids, against a baseline that still exists.
    trend = service.ops_digest_trend()
    assert trend["snapshot_count"] == 2
    assert trend["series"][0]["snapshot_id"] == "snap-0004"


def test_a_noop_prune_writes_and_records_nothing():
    service = _service()
    _take_digests(service, 2)
    report = service.prune_ops_digest_snapshots("Ops Lead", 12)
    assert report["pruned_count"] == 0
    assert report["prune"] is None
    assert service.ops_digest_snapshot_prunes() == []
    document = service._get_store().get_document(DIGEST_SNAPSHOTS_KEY)
    assert "prunes" not in document


def test_ids_are_never_reissued_after_a_prune():
    service = _service()
    _take_digests(service, 5)
    service.prune_ops_digest_snapshots("Ops Lead", 2)
    # Taking the next snapshot must not reissue snap-0003 (or
    # collide with a survivor): it runs past the highest kept.
    snapshot = service.snapshot_ops_digest("Ops Lead")
    assert snapshot["snapshot_id"] == "snap-0006"
    # And the prune record survived the take.
    assert len(service.ops_digest_snapshot_prunes()) == 1

    service.prune_ops_digest_snapshots("Ops Lead", 1)
    prunes = service.ops_digest_snapshot_prunes()
    assert len(prunes) == 2  # each prune appends its own record
    assert prunes[1]["pruned_ids"] == ["snap-0004", "snap-0005"]
    assert prunes[1]["surviving_first"] == "snap-0006"


# ---------------------------------------------------------------------------
# Calibration snapshots — the same discipline, the cited document
# ---------------------------------------------------------------------------


def test_pruning_calibration_snapshots_uses_the_same_discipline():
    service = _service()
    for _ in range(3):
        service.snapshot_sla_calibration("UW Manager")
    report = service.prune_sla_calibration_snapshots("UW Manager", 1)

    assert report["pruned_ids"] == ["cal-0001", "cal-0002"]
    assert report["surviving_last"] == "cal-0003"
    assert [
        s["snapshot_id"] for s in service.sla_calibration_snapshots()
    ] == ["cal-0003"]
    prunes = service.sla_calibration_snapshot_prunes()
    assert len(prunes) == 1
    assert prunes[0]["pruned_by"] == "UW Manager"

    # With fewer than two kept, the diff reports itself absent
    # rather than inventing movement — and the next kept report
    # continues the numbering, cal-0004, never a reissue.
    assert service.sla_calibration_snapshot_diff()["diff"] is None
    snapshot = service.snapshot_sla_calibration("UW Manager")
    assert snapshot["snapshot_id"] == "cal-0004"
    document = service._get_store().get_document(CALIBRATION_SNAPSHOTS_KEY)
    assert len(document["prunes"]) == 1


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_api_prunes_both_documents_and_lists_the_record(client):
    for _ in range(3):
        client.post("/ops/digest/snapshots", json={"actor": "Ops Lead"})
        client.post(
            "/queue/sla/calibration/snapshots", json={"actor": "UW Manager"}
        )

    res = client.post(
        "/ops/digest/snapshots/prune", json={"actor": "Ops Lead", "keep": 1}
    )
    assert res.status_code == 200
    assert res.json()["pruned_ids"] == ["snap-0001", "snap-0002"]

    listing = client.get("/ops/digest/snapshots").json()
    assert [s["snapshot_id"] for s in listing["snapshots"]] == ["snap-0003"]
    assert listing["prunes"][0]["pruned_by"] == "Ops Lead"

    res = client.post(
        "/queue/sla/calibration/snapshots/prune",
        json={"actor": "UW Manager", "keep": 2},
    )
    assert res.status_code == 200
    assert res.json()["surviving_first"] == "cal-0002"
    listing = client.get("/queue/sla/calibration/snapshots").json()
    assert len(listing["prunes"]) == 1


def test_api_prune_validates_keep_and_actor(client):
    client.post("/ops/digest/snapshots", json={"actor": "Ops Lead"})
    refused = client.post(
        "/ops/digest/snapshots/prune", json={"actor": "Ops Lead", "keep": 0}
    )
    assert refused.status_code == 422
    assert "at least 1" in refused.json()["detail"]
    anonymous = client.post("/ops/digest/snapshots/prune", json={"keep": 1})
    assert anonymous.status_code == 422
