"""Round 7: the insured loss-development timeline.

Each renewal diff answers for its own pair; the timeline chains
an insured's stored diffs (per line) into one history — an entry
per submission and one event stream per claim: appeared,
developed, no longer listed — so the third renewal reads as an
arc instead of three separate comparisons.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import insured_loss_timeline
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
BLUEPEAK = "Bluepeak Logistics Centers Inc"
NORTHGATE = "Northgate Legal LLP"


def developed_renewal() -> dict:
    sub = copy.deepcopy(SAMPLES["SUB-1002"])
    sub["submission_id"] = "SUB-2002"
    sub["submitted_at"] = "2026-10-09"
    for loss in sub["losses"]:
        if loss["date"] == "2025-09-30":
            loss["amount_incurred"] = 80000
    return sub


def settled_renewal() -> dict:
    """Third Bluepeak submission: the developed claim is settled
    and drops off the run entirely."""
    sub = developed_renewal()
    sub["submission_id"] = "SUB-2003"
    sub["submitted_at"] = "2026-10-10"
    sub["losses"] = [l for l in sub["losses"] if l["date"] != "2025-09-30"]
    return sub


@pytest.fixture()
def service() -> SubmissionService:
    svc = SubmissionService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["SUB-1002"])
    svc.analyze(developed_renewal())
    svc.analyze(settled_renewal())
    svc.analyze(SAMPLES["SUB-1001"])
    svc.analyze(SAMPLES["SUB-1013"])
    return svc


def test_timeline_chains_the_diffs_per_insured(service):
    timeline = insured_loss_timeline(service._get_store().all_records(), BLUEPEAK)
    assert timeline["insured"] == BLUEPEAK
    assert timeline["submissions"] == 3
    entries = timeline["entries"]
    assert [e["submission_id"] for e in entries] == ["SUB-1002", "SUB-2002", "SUB-2003"]
    assert entries[0]["prior_submission_id"] is None
    assert entries[1]["prior_submission_id"] == "SUB-1002"
    assert entries[1]["net_incurred_change"] == 67250.0
    assert entries[1]["events"] == {"appeared": 0, "developed": 1, "no_longer_listed": 0}
    assert entries[2]["events"] == {"appeared": 0, "developed": 0, "no_longer_listed": 1}
    totals = timeline["totals"]
    assert totals["renewals"] == 2
    assert totals["developed"] == 1
    assert totals["no_longer_listed"] == 1
    # +67,250 of development, then the $80,000 claim leaves the run.
    assert totals["cumulative_net_incurred_change"] == -12750.0


def test_timeline_claim_event_stream(service):
    timeline = insured_loss_timeline(service._get_store().all_records(), BLUEPEAK)
    assert len(timeline["claims"]) == 1
    claim = timeline["claims"][0]
    assert claim["date"] == "2025-09-30"
    assert claim["first_seen_submission_id"] == "SUB-2002"
    assert [e["type"] for e in claim["events"]] == ["developed", "no_longer_listed"]
    developed = claim["events"][0]
    assert developed["prior_incurred"] == 12750
    assert developed["incurred"] == 80000
    assert developed["delta_incurred"] == 67250.0
    assert claim["state"] == "no_longer_listed"
    assert claim["latest_incurred"] == 80000


def test_timeline_claim_appears_on_a_renewal(service):
    timeline = insured_loss_timeline(service._get_store().all_records(), NORTHGATE)
    assert timeline["submissions"] == 2
    assert timeline["totals"]["appeared"] == 1
    claim = timeline["claims"][0]
    assert claim["events"][0]["type"] == "appeared"
    assert claim["events"][0]["submission_id"] == "SUB-1013"
    assert claim["events"][0]["incurred"] == 18500
    assert claim["state"] == "on_latest_run"
    assert claim["latest_status"] == "closed"


def test_timeline_unknown_insured_and_bundle_entries(service):
    records = service._get_store().all_records()
    assert insured_loss_timeline(records, "Nobody Insurance Co") is None
    service.analyze(SAMPLES["SUB-1014"])
    timeline = insured_loss_timeline(
        service._get_store().all_records(), "Willow Creek Hospitality Group"
    )
    assert timeline["submissions"] == 1
    assert {e["line_of_business"] for e in timeline["entries"]} == {"property", "cyber"}
    assert timeline["claims"] == []
    assert timeline["totals"]["renewals"] == 0


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(api_module.app)


def test_loss_timeline_endpoint(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    client.post("/submissions/analyze", json=SAMPLES["SUB-1013"])
    response = client.get(f"/insureds/{NORTHGATE}/loss-timeline")
    assert response.status_code == 200
    body = response.json()
    assert body["insured"] == NORTHGATE
    assert body["totals"]["appeared"] == 1
    missing = client.get("/insureds/Nobody Insurance Co/loss-timeline")
    assert missing.status_code == 404
