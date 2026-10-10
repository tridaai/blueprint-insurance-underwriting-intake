"""Round 24: stale-citation traffic is counted.

Round 23 repaired a renamed-away citation on read; round 24
answers a pruned one with a tombstone — and reads recorded
nothing, so a debate whose citations keep going stale
(rename churn outrunning its readers) was invisible: the
repairs happened silently, one reader at a time. The
answering read now leaves one note per answer on the
calibration document — the one write a read makes here, and
only the answering kinds make it: a direct hit records
nothing (the citation was fresh) and a 404 records nothing
(the citation named nothing). /metrics counts the notes as a
small family: answers in total, split by kind, per cited
label — a label everyone still cites is a rename that didn't
take, and now it shows as a number.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import compute_metrics
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

BASE = date(2026, 9, 1)


def _decided(sid: str, wait_days: int) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="approved",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            decided_at=(BASE + timedelta(days=wait_days)).isoformat(),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": BASE.isoformat(),
        },
    )


def _service() -> SubmissionService:
    store = InMemoryStore()
    store.save(_decided("M-2", 2))
    return SubmissionService(store=store, checkpointer=False)


def _renamed(service: SubmissionService) -> None:
    """One kept report, pre-change renamed to post-review."""
    service.snapshot_sla_calibration("Ops Lead", label="pre-change")
    service.rename_sla_calibration_snapshot_label(
        "pre-change", "post-review", "Uma"
    )


# --- what is recorded, and what is not -------------------------------------------


def test_a_successor_answer_is_recorded_once():
    service = _service()
    _renamed(service)
    answer = service.answer_sla_calibration_citation("pre-change")
    assert answer["current_label"] == "post-review"
    notes = service.sla_calibration_citation_traffic()
    assert len(notes) == 1
    note = notes[0]
    assert note["kind"] == "successor"
    assert note["cited_label"] == "pre-change"
    assert note["current_label"] == "post-review"
    assert note["snapshot_id"] == "cal-0001"
    assert note["chain_length"] == 1
    assert note["answered_at"]


def test_a_tombstone_answer_is_recorded_with_its_kind():
    service = _service()
    _renamed(service)
    service._get_store().save(_decided("M-9", 4))
    service.snapshot_sla_calibration("Ops Lead", label="later")
    service.prune_sla_calibration_snapshots("Ravi", 1)
    answer = service.answer_sla_calibration_citation("pre-change")
    assert answer["pruned"] is True
    notes = service.sla_calibration_citation_traffic()
    assert len(notes) == 1
    assert notes[0]["kind"] == "tombstone"
    assert notes[0]["snapshot_id"] == "cal-0001"
    assert notes[0]["current_label"] == ""


def test_a_direct_hit_records_nothing():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="may")
    # The plain by-label read is the direct hit's surface; the
    # answering method is only for names no report carries —
    # and a name that never named anything answers nothing
    # and records nothing.
    assert service.sla_calibration_snapshot_by_label("may") is not None
    assert service.answer_sla_calibration_citation("nope") is None
    assert service.sla_calibration_citation_traffic() == []


def test_repeated_stale_citations_accumulate_per_label():
    service = _service()
    _renamed(service)
    for _ in range(3):
        service.answer_sla_calibration_citation("pre-change")
    traffic = service.metrics()["citation_traffic"]
    assert traffic["answers"] == 3
    assert traffic["by_cited_label"] == {"pre-change": 3}


# --- the metrics family -------------------------------------------------------------


def test_the_family_counts_kinds_labels_and_targets():
    service = _service()
    _renamed(service)
    service.answer_sla_calibration_citation("pre-change")
    service.answer_sla_calibration_citation("pre-change")
    service._get_store().save(_decided("M-9", 4))
    service.snapshot_sla_calibration("Ops Lead", label="later")
    service.prune_sla_calibration_snapshots("Ravi", 1)
    service.answer_sla_calibration_citation("pre-change")
    traffic = service.metrics()["citation_traffic"]
    assert traffic["answers"] == 3
    assert traffic["successor_answers"] == 2
    assert traffic["tombstone_answers"] == 1
    assert traffic["by_cited_label"] == {"pre-change": 3}
    # Only successor answers point anywhere.
    assert traffic["successor_targets"] == {"post-review": 2}
    assert traffic["last_answered_at"]


def test_no_answers_is_an_empty_family_not_an_absent_one():
    traffic = compute_metrics([])["citation_traffic"]
    assert traffic == {
        "answers": 0,
        "successor_answers": 0,
        "tombstone_answers": 0,
        "by_cited_label": {},
        "successor_targets": {},
        "last_answered_at": "",
    }


# --- through the API ------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_the_by_label_read_records_its_repairs(client):
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    # A direct hit leaves no note.
    res = client.get("/queue/sla/calibration/snapshots/by-label/pre-change")
    assert res.status_code == 200
    assert client.get("/metrics").json()["citation_traffic"]["answers"] == 0
    client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={
            "actor": "Uma",
            "from_label": "pre-change",
            "to_label": "post-review",
        },
    )
    # The repaired read does.
    res = client.get("/queue/sla/calibration/snapshots/by-label/pre-change")
    assert res.json()["current_label"] == "post-review"
    traffic = client.get("/metrics").json()["citation_traffic"]
    assert traffic["successor_answers"] == 1
    assert traffic["by_cited_label"] == {"pre-change": 1}
    # And a 404 leaves none behind.
    res = client.get("/queue/sla/calibration/snapshots/by-label/nope")
    assert res.status_code == 404
    assert client.get("/metrics").json()["citation_traffic"]["answers"] == 1
