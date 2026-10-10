"""Round 2: signed webhook, /metrics, audit export.

The approval webhook gains an HMAC-SHA256 signature
(``X-Trida-Signature``) when ``ACTION_WEBHOOK_SECRET`` is set, so a
receiving system can verify the packet came from this service. The
metrics and audit export are deterministic projections of the store
of record — the same records the underwriter gate writes.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.api import app
from underwriting_agent.insights import audit_csv
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


@pytest.fixture()
def service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture_webhook(monkeypatch, secret: str | None = None) -> dict:
    import underwriting_agent.service as service_module

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse()

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    if secret is not None:
        monkeypatch.setenv("ACTION_WEBHOOK_SECRET", secret)
    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_webhook_is_signed_when_secret_is_set(monkeypatch, service):
    captured = _capture_webhook(monkeypatch, secret="s3cret-key")
    service.analyze(SAMPLES["SUB-1001"])
    decided = service.approve("SUB-1001", approver="U. Writer")
    assert decided.dispatch_status == "sent"

    request = captured["request"]
    signature = request.get_header("X-trida-signature")
    assert signature is not None and signature.startswith("sha256=")
    expected = hmac.new(
        b"s3cret-key", request.data, hashlib.sha256
    ).hexdigest()
    assert signature == f"sha256={expected}"


def test_webhook_is_unsigned_without_a_secret(monkeypatch, service):
    captured = _capture_webhook(monkeypatch)
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U. Writer")
    assert captured["request"].get_header("X-trida-signature") is None


def test_metrics_aggregate_the_store(service):
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1002"])
    service.approve("SUB-1001", approver="U. Writer", reason="clean")

    metrics = service.metrics()
    assert metrics["runs"] == 2
    assert metrics["awaiting_underwriter"] == 1
    assert metrics["approved"] == 1
    assert metrics["approval_rate"] == 1.0
    assert metrics["outcomes"]["in_appetite"] == 1
    assert metrics["model_calls"] >= 2
    assert metrics["latency_seconds"]["total"] > 0
    # The offline fallback reports no tokens/cost — honestly None/0.
    assert metrics["token_runs"] == 0
    assert metrics["estimated_cost_usd"] is None


def test_audit_rows_and_csv(service):
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U. Writer", reason="clean file")

    rows = service.audit_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["submission_id"] == "SUB-1001"
    assert row["insured_name"] == "Northgate Legal LLP"
    assert row["broker_name"] == "Harborline Brokers"
    assert row["status"] == "approved"
    assert row["decided_by"] == "U. Writer"
    assert row["decision_reason"] == "clean file"
    assert row["guardrails_passed"] is True

    text = audit_csv(rows)
    lines = text.strip().splitlines()
    assert lines[0].startswith("submission_id,insured_name,broker_name")
    assert "SUB-1001" in lines[1]
    assert "clean file" in lines[1]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_metrics_and_audit_endpoints(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["runs"] == 1

    as_json = client.get("/audit/export")
    assert as_json.status_code == 200
    body = as_json.json()
    assert body["count"] == 1
    assert body["records"][0]["submission_id"] == "SUB-1001"

    as_csv = client.get("/audit/export", params={"format": "csv"})
    assert as_csv.status_code == 200
    assert as_csv.headers["content-type"].startswith("text/csv")
    assert "SUB-1001" in as_csv.text

    assert client.get("/audit/export", params={"format": "xml"}).status_code == 422
