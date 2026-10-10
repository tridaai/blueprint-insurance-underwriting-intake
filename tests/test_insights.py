"""Round 2: inbox grouping + history scorecards.

The store is the system of record; the inbox groups it by insured
and by broker, and the scorecards aggregate each party's track
record (outcome mix, approval rate, average completeness, top
referral reasons). A new submission's diagnosis evidence cites the
scorecard when one exists for its insured or broker.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.api import app
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import SubmissionInput
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


def make_submission(submission_id: str, insured: str, broker: str) -> SubmissionInput:
    return SubmissionInput.model_validate(
        {
            "submission_id": submission_id,
            "insured_name": insured,
            "broker_name": broker,
            "line_of_business": "cyber",
            "class_of_business": "consulting",
            "broker_email_text": "Submission attached. MFA is enforced.",
            "documents": [
                {
                    "document_id": "app",
                    "doc_type": "application",
                    "fields": {"insured_name": insured, "revenue": "5000000"},
                    "raw_text": "Application. Multi-factor authentication enforced.",
                },
                {
                    "document_id": "lr",
                    "doc_type": "loss runs",
                    "fields": {"loss_history_present": "yes"},
                    "raw_text": "No losses in the period.",
                },
            ],
            "revenue": 5_000_000,
            "years_in_business": 8,
            "losses": [],
        }
    )


@pytest.fixture()
def service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def test_inbox_groups_by_broker_and_insured(service):
    service.analyze(make_submission("SUB-G1", "Insured A", "Broker One"))
    service.analyze(make_submission("SUB-G2", "Insured B", "Broker One"))
    service.analyze(make_submission("SUB-G3", "Insured A", "Broker Two"))

    inbox = service.inbox()
    brokers = {g["name"]: g for g in inbox["brokers"]}
    assert brokers["Broker One"]["submission_count"] == 2
    assert brokers["Broker One"]["latest_submission_id"] == "SUB-G2"
    assert brokers["Broker One"]["latest_outcome"] == "in_appetite"
    assert brokers["Broker One"]["latest_readiness"] == "ready"
    assert brokers["Broker One"]["awaiting_count"] == 2
    assert brokers["Broker Two"]["submission_count"] == 1

    insureds = {g["name"]: g for g in inbox["insureds"]}
    assert insureds["Insured A"]["submission_count"] == 2
    assert insureds["Insured A"]["latest_submission_id"] == "SUB-G3"
    assert insureds["Insured B"]["submission_count"] == 1


def test_scorecard_aggregates_decisions_and_completeness(service):
    service.analyze(make_submission("SUB-G1", "Insured A", "Broker One"))
    service.analyze(make_submission("SUB-G2", "Insured B", "Broker One"))
    service.approve("SUB-G1", approver="U. Writer", reason="clean")
    service.reject("SUB-G2", reviewer="U. Writer", reason="not this time")

    card = service.scorecard("broker", "Broker One")
    assert card is not None
    assert card["submissions"] == 2
    assert card["outcome_mix"] == {"in_appetite": 2}
    assert card["decisions"] == 2
    assert card["approvals"] == 1
    assert card["rejections"] == 1
    assert card["approval_rate"] == 0.5
    assert card["avg_completeness"] == 1.0

    cards = service.scorecards()
    assert [c["name"] for c in cards["brokers"]] == ["Broker One"]
    assert service.scorecard("broker", "Nobody") is None


def test_scorecard_cites_top_referral_reasons(service):
    # SUB-1012 carries referral triggers (borderline on loss history).
    service.analyze(SubmissionInput.model_validate(SAMPLES["SUB-1012"]))
    card = service.scorecard("broker", "Meridian Wholesale Brokers")
    assert card is not None
    assert card["submissions"] == 1
    assert card["top_referral_reasons"], "expected referral triggers on SUB-1012"


def test_diagnosis_evidence_cites_scorecard(service):
    service.analyze(make_submission("SUB-G1", "Insured A", "Broker One"))
    service.approve("SUB-G1", approver="U. Writer", reason="clean")
    result = service.analyze(make_submission("SUB-G2", "Insured B", "Broker One"))
    lines = [e for e in result.diagnosis.evidence if "scorecard" in e]
    assert any(line.startswith("broker scorecard:") for line in lines)
    assert any("approval rate 1.0" in line for line in lines)


def test_no_scorecard_line_without_history(service):
    result = service.analyze(make_submission("SUB-G1", "Insured A", "Broker One"))
    assert not any("scorecard" in e for e in result.diagnosis.evidence)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_inbox_and_scorecard_endpoints(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1002"])
    client.post("/submissions/analyze", json=SAMPLES["SUB-1005"])

    inbox = client.get("/inbox").json()
    meridian = next(g for g in inbox["brokers"] if g["name"] == "Meridian Wholesale Brokers")
    assert meridian["submission_count"] == 2
    assert meridian["latest_submission_id"] == "SUB-1005"

    brokers_only = client.get("/inbox/brokers").json()
    assert any(g["name"] == "Meridian Wholesale Brokers" for g in brokers_only)
    insureds_only = client.get("/inbox/insureds").json()
    assert {g["name"] for g in insureds_only} == {
        "Bluepeak Logistics Centers Inc",
        "Continental Grocers Cooperative",
    }

    cards = client.get("/scorecards").json()
    card = next(c for c in cards["brokers"] if c["name"] == "Meridian Wholesale Brokers")
    assert card["submissions"] == 2
    assert card["approval_rate"] is None  # no decisions yet

    one = client.get("/scorecards/brokers/Meridian Wholesale Brokers")
    assert one.status_code == 200
    assert one.json()["kind"] == "broker"
    assert client.get("/scorecards/brokers/No Such Broker").status_code == 404
    assert client.get("/scorecards/insureds/No Such Insured").status_code == 404
