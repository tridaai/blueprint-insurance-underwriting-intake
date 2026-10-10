"""Round 20: retention for the drafting-events document.

The broker-quality drafting events grow one per served report —
drafted on read, so the document grows with reading, not with
asking — and nothing ever left it: the aggregate needs history,
not infinity. The events now carry the snapshot documents'
retention discipline: a keep-last-N prune, operator-called,
previewed first, recorded on the document itself. The stakes
are named in the prune's own record: the events are the broker
drafting counts' only source, so a prune is the one act that
legitimately moves those counts backwards — in /metrics and in
the kept digest snapshots alike — and the document says so.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import BROKER_QUALITY_DRAFTING_KEY
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

BROKER = "Harborline Brokers"


def _event(day: int, outcome: str = "clean") -> dict:
    return {
        "broker": BROKER,
        "drafted_at": f"2026-10-{day:02d}T09:00:00+00:00",
        "outcome": outcome,
        "repair_families": [],
        "failed_checks": [],
        "steered": False,
    }


def _service_with_events(*events: dict) -> SubmissionService:
    store = InMemoryStore()
    store.save_document(
        BROKER_QUALITY_DRAFTING_KEY, {"events": list(events)}
    )
    return SubmissionService(store=store, checkpointer=False)


def _five_events() -> list[dict]:
    return [
        _event(1),
        _event(2, "repaired"),
        _event(3, "refused"),
        _event(4),
        _event(5, "repaired"),
    ]


# --- the preview -------------------------------------------------------------------------


def test_the_preview_reports_the_cut_and_writes_nothing():
    service = _service_with_events(*_five_events())
    preview = service.preview_prune_broker_quality_drafting_events(2)
    assert preview == {
        "event_count": 5,
        "keep": 2,
        "would_change": True,
        "would_prune_count": 3,
        "would_prune_oldest": "2026-10-01T09:00:00+00:00",
        "would_prune_newest": "2026-10-03T09:00:00+00:00",
        "surviving_count": 2,
        "surviving_first": "2026-10-04T09:00:00+00:00",
        "surviving_last": "2026-10-05T09:00:00+00:00",
    }
    # A preview pruned nothing: the document stands as it was.
    assert len(service.broker_quality_drafting_events()) == 5
    assert service.broker_quality_drafting_prunes() == []


def test_a_preview_at_or_under_the_count_changes_nothing():
    service = _service_with_events(*_five_events())
    preview = service.preview_prune_broker_quality_drafting_events(5)
    assert preview["would_change"] is False
    assert preview["would_prune_count"] == 0
    assert preview["surviving_count"] == 5


# --- the prune ---------------------------------------------------------------------------


def test_the_prune_keeps_the_newest_and_records_itself():
    service = _service_with_events(*_five_events())
    report = service.prune_broker_quality_drafting_events("Uma", 2)
    assert report["kept"] == 2
    assert report["pruned_count"] == 3
    assert report["pruned_oldest"] == "2026-10-01T09:00:00+00:00"
    assert report["pruned_newest"] == "2026-10-03T09:00:00+00:00"
    assert report["surviving_first"] == "2026-10-04T09:00:00+00:00"
    assert report["surviving_last"] == "2026-10-05T09:00:00+00:00"
    events = service.broker_quality_drafting_events()
    assert [e["drafted_at"] for e in events] == [
        "2026-10-04T09:00:00+00:00",
        "2026-10-05T09:00:00+00:00",
    ]
    prunes = service.broker_quality_drafting_prunes()
    assert len(prunes) == 1
    entry = prunes[0]
    assert entry["pruned_by"] == "Uma"
    assert entry["requested_keep"] == 2
    assert entry["pruned_count"] == 3
    assert entry["pruned_oldest"] == report["pruned_oldest"]
    assert report["prune"] == entry


def test_the_preview_and_the_prune_cannot_disagree():
    service = _service_with_events(*_five_events())
    preview = service.preview_prune_broker_quality_drafting_events(3)
    report = service.prune_broker_quality_drafting_events("Uma", 3)
    assert report["pruned_count"] == preview["would_prune_count"]
    assert report["pruned_oldest"] == preview["would_prune_oldest"]
    assert report["pruned_newest"] == preview["would_prune_newest"]
    assert report["surviving_first"] == preview["surviving_first"]
    assert report["surviving_last"] == preview["surviving_last"]


def test_a_prune_that_removes_nothing_records_nothing():
    service = _service_with_events(*_five_events())
    report = service.prune_broker_quality_drafting_events("Uma", 12)
    assert report["pruned_count"] == 0
    assert report["prune"] is None
    assert service.broker_quality_drafting_prunes() == []
    assert len(service.broker_quality_drafting_events()) == 5


def test_the_counts_after_a_prune_read_the_survivors_and_say_so():
    service = _service_with_events(*_five_events())
    before = service.metrics()["status_note_repairs"]["broker_quality"]
    assert before["reports_drafted"] == 5
    assert before["repaired_reports"] == 2
    assert before["refusals"] == 1
    service.prune_broker_quality_drafting_events("Uma", 2)
    after = service.metrics()["status_note_repairs"]["broker_quality"]
    # Only the survivors count now — the step down is the prune's
    # doing, and the prune's record on the document is the
    # explanation a later reader finds.
    assert after["reports_drafted"] == 2
    assert after["repaired_reports"] == 1
    assert after["refusals"] == 0
    digest = service.ops_digest()
    assert digest["counts"]["broker_quality_drafted"] == 2
    prunes = service.broker_quality_drafting_prunes()
    assert prunes[0]["pruned_count"] == 3


def test_keep_below_one_is_refused_by_prune_and_preview():
    service = _service_with_events(*_five_events())
    with pytest.raises(ValueError, match="at least 1"):
        service.prune_broker_quality_drafting_events("Uma", 0)
    with pytest.raises(ValueError, match="at least 1"):
        service.preview_prune_broker_quality_drafting_events(0)
    assert len(service.broker_quality_drafting_events()) == 5


def test_new_events_append_after_a_prune():
    service = _service_with_events(*_five_events())
    service.prune_broker_quality_drafting_events("Uma", 2)
    service._record_broker_quality_drafting(
        {"broker": BROKER, "draft_validation": {"passed": True}}
    )
    events = service.broker_quality_drafting_events()
    assert len(events) == 3
    assert events[-1]["outcome"] == "clean"
    # The prune record rides along untouched by the append.
    assert len(service.broker_quality_drafting_prunes()) == 1


# --- the API -----------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app

    service = _service_with_events(*_five_events())
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(app)


def test_the_events_read_over_the_api_with_their_prunes(client):
    res = client.get("/reports/broker-quality/drafting-events")
    assert res.status_code == 200
    body = res.json()
    assert len(body["events"]) == 5
    assert body["prunes"] == []


def test_the_prune_and_its_preview_over_the_api(client):
    res = client.post(
        "/reports/broker-quality/drafting-events/prune/preview",
        json={"actor": "Uma", "keep": 2},
    )
    assert res.status_code == 200
    assert res.json()["would_prune_count"] == 3
    res = client.post(
        "/reports/broker-quality/drafting-events/prune",
        json={"actor": "Uma", "keep": 2},
    )
    assert res.status_code == 200
    assert res.json()["pruned_count"] == 3
    body = client.get("/reports/broker-quality/drafting-events").json()
    assert len(body["events"]) == 2
    assert body["prunes"][0]["pruned_by"] == "Uma"


def test_keep_below_one_is_a_422_over_the_api(client):
    res = client.post(
        "/reports/broker-quality/drafting-events/prune",
        json={"actor": "Uma", "keep": 0},
    )
    assert res.status_code == 422
    res = client.post(
        "/reports/broker-quality/drafting-events/prune/preview",
        json={"actor": "Uma", "keep": 0},
    )
    assert res.status_code == 422
