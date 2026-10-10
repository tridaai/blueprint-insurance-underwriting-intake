"""Round 15: prune previews for both kept-snapshot documents.

Round 14's retention prune is exact about what left — after it
left. An operator deciding whether the oldest kept reports may
go committed blind: no way to see which ids a keep-N would
remove, or what range would survive, before the write. The
preview (the restore/refresh idiom) runs the prune's own split
and reports the plan — would-remove ids, surviving range —
writing nothing and recording nothing. And because preview and
prune share the one computation, the prune that follows on the
same document produces exactly the previewed removals.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import (
    DIGEST_SNAPSHOTS_KEY,
    snapshot_prune_preview,
)
from underwriting_agent.proposals import CALIBRATION_SNAPSHOTS_KEY
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _snapshots(prefix: str, count: int) -> list[dict]:
    return [
        {
            "snapshot_id": f"{prefix}{i:04d}",
            "taken_at": f"2026-10-{i:02d}T09:00:00+00:00",
            "taken_by": "Ops Lead",
        }
        for i in range(1, count + 1)
    ]


def _service_with_documents() -> tuple[SubmissionService, InMemoryStore]:
    store = InMemoryStore()
    store.save_document(
        DIGEST_SNAPSHOTS_KEY, {"snapshots": _snapshots("snap-", 5)}
    )
    store.save_document(
        CALIBRATION_SNAPSHOTS_KEY, {"snapshots": _snapshots("cal-", 4)}
    )
    return SubmissionService(store=store, checkpointer=False), store


# ---------------------------------------------------------------------------
# The pure plan
# ---------------------------------------------------------------------------


def test_the_preview_names_what_would_leave_and_what_would_survive():
    plan = snapshot_prune_preview(_snapshots("snap-", 5), 2)
    assert plan == {
        "snapshot_count": 5,
        "keep": 2,
        "would_change": True,
        "would_prune_count": 3,
        "would_prune_ids": ["snap-0001", "snap-0002", "snap-0003"],
        # Round 23 shape growth, deliberate: the plan also
        # names the renames that would leave the kept reports
        # (empty here — these snapshots were never renamed),
        # so the prune's kept-whole record is previewed too.
        "would_prune_renames": [],
        "surviving_count": 2,
        "surviving_first": "snap-0004",
        "surviving_last": "snap-0005",
    }


def test_a_preview_that_would_remove_nothing_says_so():
    plan = snapshot_prune_preview(_snapshots("snap-", 2), 12)
    assert plan["would_change"] is False
    assert plan["would_prune_ids"] == []
    assert plan["surviving_first"] == "snap-0001"
    assert plan["surviving_last"] == "snap-0002"


def test_the_preview_keeps_the_prunes_refusal():
    with pytest.raises(ValueError, match="at least 1"):
        snapshot_prune_preview(_snapshots("snap-", 2), 0)


# ---------------------------------------------------------------------------
# The service: a dry run over either document, then the real prune
# ---------------------------------------------------------------------------


def test_the_digest_preview_writes_nothing_and_records_nothing():
    service, store = _service_with_documents()
    before = copy.deepcopy(store.get_document(DIGEST_SNAPSHOTS_KEY))
    plan = service.preview_prune_ops_digest_snapshots(2)
    assert plan["would_prune_ids"] == ["snap-0001", "snap-0002", "snap-0003"]
    assert store.get_document(DIGEST_SNAPSHOTS_KEY) == before


def test_the_calibration_preview_reads_its_own_document():
    service, store = _service_with_documents()
    before = copy.deepcopy(store.get_document(CALIBRATION_SNAPSHOTS_KEY))
    plan = service.preview_prune_sla_calibration_snapshots(3)
    assert plan["would_prune_ids"] == ["cal-0001"]
    assert plan["surviving_first"] == "cal-0002"
    assert store.get_document(CALIBRATION_SNAPSHOTS_KEY) == before


def test_the_prune_produces_exactly_the_previewed_removals():
    service, _store = _service_with_documents()
    plan = service.preview_prune_ops_digest_snapshots(2)
    result = service.prune_ops_digest_snapshots("Ops Lead", 2)
    assert result["pruned_ids"] == plan["would_prune_ids"]
    assert result["surviving_first"] == plan["surviving_first"]
    assert result["surviving_last"] == plan["surviving_last"]
    assert result["kept"] == plan["surviving_count"]


def test_the_service_preview_keeps_the_refusal():
    service, _store = _service_with_documents()
    with pytest.raises(ValueError, match="at least 1"):
        service.preview_prune_ops_digest_snapshots(0)
    with pytest.raises(ValueError, match="at least 1"):
        service.preview_prune_sla_calibration_snapshots(0)


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    service, _store = _service_with_documents()
    api_module.service = service
    return TestClient(api_module.app)


def test_preview_endpoints_for_both_documents(client):
    res = client.post(
        "/ops/digest/snapshots/prune/preview",
        json={"actor": "Ops Lead", "keep": 2},
    )
    assert res.status_code == 200
    assert res.json()["would_prune_ids"] == [
        "snap-0001",
        "snap-0002",
        "snap-0003",
    ]
    res = client.post(
        "/queue/sla/calibration/snapshots/prune/preview",
        json={"actor": "Ops Lead", "keep": 3},
    )
    assert res.status_code == 200
    assert res.json()["would_prune_ids"] == ["cal-0001"]
    # And nothing was pruned by previewing: both lists still full.
    assert len(client.get("/ops/digest/snapshots").json()["snapshots"]) == 5
    assert (
        len(client.get("/queue/sla/calibration/snapshots").json()["snapshots"])
        == 4
    )


def test_preview_endpoints_refuse_keep_below_one(client):
    for url in (
        "/ops/digest/snapshots/prune/preview",
        "/queue/sla/calibration/snapshots/prune/preview",
    ):
        res = client.post(url, json={"actor": "Ops Lead", "keep": 0})
        assert res.status_code == 422
