"""Round 20: retention for the proposals document.

The two snapshot documents and the drafting-events document
carry the retention discipline; the proposals document —
every drafted threshold and SLA proposal with its decision —
grew unbounded beside them. It now has the same governed
treatment, with its own unit: keep-last-N over the **decided
history**. A live draft is never pruned — an open question is
not history — and the prune is recorded on the document
itself, because the drafting scans read the decided history
(a rejected direction is never re-asked): a prune thins what
the scans remember, and the document says when, under whom,
and what left.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import PROPOSALS_KEY, load_proposals
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _proposal(pid: str, status: str) -> dict:
    return {
        "proposal_id": pid,
        "kind": "diff_trigger_threshold",
        "dimension": "limit_jump_pct",
        "status": status,
        "drafted_at": "2026-09-01T09:00:00+00:00",
    }


def _document() -> list[dict]:
    return [
        _proposal("p-001", "accepted"),
        _proposal("p-002", "draft"),
        _proposal("p-003", "rejected"),
        _proposal("p-004", "withdrawn"),
        _proposal("p-005", "draft"),
        _proposal("p-006", "accepted"),
        _proposal("p-007", "rejected"),
    ]


def _service_with(*proposals: dict) -> SubmissionService:
    store = InMemoryStore()
    store.save_document(PROPOSALS_KEY, {"proposals": list(proposals)})
    return SubmissionService(store=store, checkpointer=False)


# --- the preview -------------------------------------------------------------------------


def test_the_preview_counts_history_and_drafts_apart():
    service = _service_with(*_document())
    preview = service.preview_prune_calibration_proposals(2)
    assert preview == {
        "proposal_count": 7,
        "history_count": 5,
        "draft_count": 2,
        "keep": 2,
        "would_change": True,
        "would_prune_count": 3,
        "would_prune_ids": ["p-001", "p-003", "p-004"],
        "surviving_count": 4,
        "surviving_history_count": 2,
        "surviving_first": "p-006",
        "surviving_last": "p-007",
    }
    # A preview pruned nothing and recorded nothing.
    assert len(service.calibration_proposals()) == 7
    assert service.calibration_proposal_prunes() == []


# --- the prune ---------------------------------------------------------------------------


def test_the_prune_keeps_the_newest_history_and_every_draft():
    service = _service_with(*_document())
    report = service.prune_calibration_proposals("Uma", 2)
    assert report["kept"] == 4
    assert report["kept_history"] == 2
    assert report["kept_drafts"] == 2
    assert report["pruned_ids"] == ["p-001", "p-003", "p-004"]
    assert report["surviving_first"] == "p-006"
    assert report["surviving_last"] == "p-007"
    survivors = load_proposals(service._get_store())
    # Document order preserved; the drafts stand where they were
    # drafted, interleaved with the surviving history.
    assert [p["proposal_id"] for p in survivors] == [
        "p-002",
        "p-005",
        "p-006",
        "p-007",
    ]
    prunes = service.calibration_proposal_prunes()
    assert len(prunes) == 1
    entry = prunes[0]
    assert entry["pruned_by"] == "Uma"
    assert entry["requested_keep"] == 2
    assert entry["pruned_ids"] == ["p-001", "p-003", "p-004"]
    assert entry["pruned_by_status"] == {
        "accepted": 1,
        "rejected": 1,
        "withdrawn": 1,
    }
    assert report["prune"] == entry


def test_drafts_survive_even_the_tightest_keep():
    service = _service_with(*_document())
    service.prune_calibration_proposals("Uma", 1)
    survivors = load_proposals(service._get_store())
    assert [p["proposal_id"] for p in survivors] == [
        "p-002",
        "p-005",
        "p-007",
    ]


def test_the_preview_and_the_prune_cannot_disagree():
    service = _service_with(*_document())
    preview = service.preview_prune_calibration_proposals(3)
    report = service.prune_calibration_proposals("Uma", 3)
    assert report["pruned_ids"] == preview["would_prune_ids"]
    assert report["pruned_count"] == preview["would_prune_count"]
    assert report["surviving_first"] == preview["surviving_first"]
    assert report["surviving_last"] == preview["surviving_last"]


def test_a_prune_that_removes_nothing_records_nothing():
    service = _service_with(*_document())
    report = service.prune_calibration_proposals("Uma", 5)
    assert report["pruned_count"] == 0
    assert report["prune"] is None
    assert service.calibration_proposal_prunes() == []
    assert len(service.calibration_proposals()) == 7


def test_keep_below_one_is_refused_by_prune_and_preview():
    service = _service_with(*_document())
    with pytest.raises(ValueError, match="at least 1"):
        service.prune_calibration_proposals("Uma", 0)
    with pytest.raises(ValueError, match="at least 1"):
        service.preview_prune_calibration_proposals(0)
    assert len(service.calibration_proposals()) == 7


def test_the_prune_record_survives_later_proposal_writes():
    document = _document()
    # The surviving draft's evidence window is still live, so it
    # can be decided below.
    document[1]["expires_at"] = "2027-01-01T00:00:00+00:00"
    service = _service_with(*document)
    service.prune_calibration_proposals("Uma", 2)
    # A later write to the document — here a decision, which
    # saves the proposals list back — must not erase the prune
    # record it does not own.
    service.decide_calibration_proposal("p-002", "Uma", accept=False)
    assert len(service.calibration_proposal_prunes()) == 1
    survivors = {
        p["proposal_id"]: p["status"]
        for p in load_proposals(service._get_store())
    }
    assert survivors["p-002"] == "rejected"


def test_a_second_prune_appends_beside_the_first():
    service = _service_with(*_document())
    service.prune_calibration_proposals("Uma", 4)
    service.prune_calibration_proposals("Priya", 2)
    prunes = service.calibration_proposal_prunes()
    assert [p["pruned_by"] for p in prunes] == ["Uma", "Priya"]
    assert prunes[0]["pruned_ids"] == ["p-001"]
    assert prunes[1]["pruned_ids"] == ["p-003", "p-004"]


# --- the API -----------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app

    service = _service_with(*_document())
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(app)


def test_the_proposals_read_carries_the_prunes(client):
    res = client.get("/calibration/proposals")
    assert res.status_code == 200
    assert res.json()["prunes"] == []


def test_the_prune_and_its_preview_over_the_api(client):
    res = client.post(
        "/calibration/proposals/prune/preview",
        json={"actor": "Uma", "keep": 2},
    )
    assert res.status_code == 200
    assert res.json()["would_prune_ids"] == ["p-001", "p-003", "p-004"]
    res = client.post(
        "/calibration/proposals/prune", json={"actor": "Uma", "keep": 2}
    )
    assert res.status_code == 200
    assert res.json()["pruned_count"] == 3
    body = client.get("/calibration/proposals").json()
    assert len(body["proposals"]) == 4
    assert body["prunes"][0]["pruned_by"] == "Uma"


def test_keep_below_one_is_a_422_over_the_api(client):
    res = client.post(
        "/calibration/proposals/prune", json={"actor": "Uma", "keep": 0}
    )
    assert res.status_code == 422
    res = client.post(
        "/calibration/proposals/prune/preview",
        json={"actor": "Uma", "keep": 0},
    )
    assert res.status_code == 422
