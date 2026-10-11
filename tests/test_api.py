"""API surface: analyze → gate → decision, streaming, auth, errors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.api import app
from underwriting_agent.samples import load_sample_submissions

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    # Rebuild the module-level service against the in-memory doubles
    # (no DATABASE_URL in the suite environment → nothing persists).
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_console_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Underwriting Submission Intake Agent" in res.text
    assert "tridaai-logo.png" in res.text


def test_samples_and_guidelines_endpoints(client):
    samples = client.get("/samples").json()
    # 17 since the duplicate-detection round added the designed
    # cross-broker pair (SUB-1015/1016) and its near-miss (SUB-1017).
    assert len(samples) == 17
    guidelines = client.get("/guidelines").json()
    assert any(g["id"] == "UW-GEN-04" for g in guidelines)


def test_analyze_approve_flow(client):
    res = client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "awaiting_underwriter"
    assert body["triage"]["outcome"] == "in_appetite"
    assert body["external_action_taken"] is False

    decided = client.post(
        "/submissions/SUB-1001/approve", json={"actor": "U. Writer", "reason": "clean"}
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"

    fetched = client.get("/submissions/SUB-1001")
    assert fetched.json()["decided_by"] == "U. Writer"


def test_approve_failed_validation_returns_422(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1011"])
    res = client.post("/submissions/SUB-1011/approve", json={"actor": "U. Writer"})
    assert res.status_code == 422
    assert "guardrail" in res.json()["detail"]


def test_decision_requires_an_actor(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    res = client.post("/submissions/SUB-1001/approve", json={})
    assert res.status_code == 422


def test_unknown_submission_returns_404(client):
    assert client.get("/submissions/SUB-9999").status_code == 404
    assert client.post("/submissions/SUB-9999/reject", json={"actor": "X"}).status_code == 404


def test_streaming_endpoint_emits_events_and_final_result(client):
    res = client.post("/submissions/analyze/stream", json=SAMPLES["SUB-1009"])
    assert res.status_code == 200
    assert "run_started" in res.text
    assert "node_completed" in res.text
    assert "run_completed" in res.text
    assert '"submission_id": "SUB-1009"' in res.text


def test_api_key_auth_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    assert client.get("/samples").status_code == 401
    assert client.get("/samples", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/samples", headers={"X-API-Key": "test-secret"}).status_code == 200
    # Health and the console stay open.
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
