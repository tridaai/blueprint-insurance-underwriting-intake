"""Guideline change-impact re-triage: when the corpus changes, say
which queued submissions would move — and change nothing doing it.
"""

from __future__ import annotations

import pytest

from underwriting_agent.guidelines_data import GUIDELINES_VERSION
from underwriting_agent.retriage import guideline_change_impact
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore
from underwriting_agent.triage import classify_submission

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def test_results_are_stamped_with_the_corpus_version():
    service = make_service()
    result = service.analyze(SAMPLES["SUB-1001"])
    assert result.guidelines_version == GUIDELINES_VERSION


def test_unchanged_rules_move_nothing():
    service = make_service()
    for sid in ("SUB-1001", "SUB-1002", "SUB-1004"):
        service.analyze(SAMPLES[sid])
    report = service.guideline_change_impact()
    assert report["guidelines_version"] == GUIDELINES_VERSION
    assert report["checked"] == 3
    assert report["moved"] == [] and report["moved_count"] == 0
    assert report["unchanged_count"] == 3


def test_decided_submissions_are_out_of_scope():
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1009"])
    service.approve("SUB-1001", approver="U")
    report = service.guideline_change_impact()
    assert report["checked"] == 1  # only SUB-1009 is still awaiting


def _tightened_appetite(sub, facts) -> Triage:
    """A simulated corpus change: cyber is now referred by default."""
    triage = classify_submission(sub, facts)
    if sub.line_of_business == "cyber" and triage.outcome == "in_appetite":
        return triage.model_copy(
            update={
                "outcome": "borderline",
                "rationale": "Simulated change: cyber now referred by default.",
            }
        )
    return triage


def test_changed_rules_report_the_movers_and_mutate_nothing():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])  # cyber, in_appetite
    service.analyze(SAMPLES["SUB-1009"])  # property, in_appetite
    before = {
        r.result.submission_id: (r.result.triage.outcome, r.result.status)
        for r in store.all_records()
    }

    report = guideline_change_impact(store.all_records(), classifier=_tightened_appetite)
    assert report["checked"] == 2
    assert report["moved_count"] == 1
    mover = report["moved"][0]
    assert mover["submission_id"] == "SUB-1001"
    assert mover["from_outcome"] == "in_appetite"
    assert mover["to_outcome"] == "borderline"
    assert mover["triaged_under"] == GUIDELINES_VERSION

    after = {
        r.result.submission_id: (r.result.triage.outcome, r.result.status)
        for r in store.all_records()
    }
    assert before == after  # report only — nothing was rewritten


def test_unversioned_records_are_named_as_such():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    record = store.get("SUB-1001")
    record.result.guidelines_version = ""  # as stored before versioning
    store.save(record)
    report = guideline_change_impact(store.all_records(), classifier=_tightened_appetite)
    assert report["moved"][0]["triaged_under"] == "unversioned"


def test_api_change_impact_endpoint(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    res = client.get("/guidelines/change-impact")
    assert res.status_code == 200
    body = res.json()
    assert body["guidelines_version"] == GUIDELINES_VERSION
    assert body["checked"] == 1
    assert body["moved"] == []
    assert "No stored result was changed" in body["note"]
