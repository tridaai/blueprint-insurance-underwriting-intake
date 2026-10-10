"""Webhook delivery ledger: every dispatch attempt is recorded,
and a failed dispatch has a retry path.

Before this, a failed approval webhook was a status string — the
approval stood (correctly), but the delivery left no trace beyond
``failed`` and no way to try again short of re-approving, which the
gate forbids. The ledger records each attempt (timestamp, HTTP
status, signature id, error, its scheduled backoff) on the record,
and ``retry_dispatch`` appends new attempts until one lands.
"""

from __future__ import annotations

import pytest

import underwriting_agent.service as service_module
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, SQLiteStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


@pytest.fixture()
def client(monkeypatch):
    """The API over a fresh in-memory service (same arrangement as
    test_api.py's fixture)."""
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def approve_with_urlopen(monkeypatch, urlopen, secret: str | None = None):
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    if secret:
        monkeypatch.setenv("ACTION_WEBHOOK_SECRET", secret)
    monkeypatch.setattr(service_module.urllib.request, "urlopen", urlopen)
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    return service, service.approve("SUB-1001", approver="U")


def test_successful_dispatch_records_a_ledger_attempt(monkeypatch):
    captured = {}

    def ok(request, timeout=None):
        captured["headers"] = dict(request.headers)
        return FakeResponse(204)

    service, decided = approve_with_urlopen(monkeypatch, ok, secret="s3cret")
    assert decided.dispatch_status == "sent"
    assert len(decided.dispatch_attempts) == 1
    attempt = decided.dispatch_attempts[0]
    assert attempt.attempt == 1
    assert attempt.status == "sent"
    assert attempt.http_status == 204
    assert attempt.error is None
    assert attempt.backoff_seconds == 0
    assert attempt.attempted_at  # a timestamp was recorded
    # Signed, and the ledger keeps a truncated identifier only.
    assert captured["headers"]["X-trida-signature"].startswith("sha256=")
    assert attempt.signature_id is not None
    assert attempt.signature_id.startswith("sha256=")
    assert len(attempt.signature_id) < len(captured["headers"]["X-trida-signature"])
    # The service exposes the same ledger.
    ledger = service.dispatch_ledger("SUB-1001")
    assert ledger["dispatch_status"] == "sent"
    assert len(ledger["attempts"]) == 1


def test_failed_dispatch_records_the_error_and_http_status(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    _service, decided = approve_with_urlopen(monkeypatch, boom)
    assert decided.status == "approved"  # the approval stands
    assert decided.dispatch_status == "failed"
    attempt = decided.dispatch_attempts[0]
    assert attempt.status == "failed"
    assert attempt.http_status is None
    assert "connection refused" in (attempt.error or "")
    assert attempt.signature_id is None  # unsigned run


def test_receiver_error_status_is_recorded_from_http_error(monkeypatch):
    import urllib.error

    def unavailable(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", {}, None
        )

    _service, decided = approve_with_urlopen(monkeypatch, unavailable)
    attempt = decided.dispatch_attempts[0]
    assert attempt.status == "failed"
    assert attempt.http_status == 503
    assert "503" in (attempt.error or "")


def test_retry_dispatch_recovers_a_failed_delivery(monkeypatch):
    state = {"failing": True}

    def flaky(request, timeout=None):
        if state["failing"]:
            raise OSError("connection refused")
        return FakeResponse(200)

    service, decided = approve_with_urlopen(monkeypatch, flaky)
    assert decided.dispatch_status == "failed"
    assert decided.external_action_taken is False

    state["failing"] = False
    retried = service.retry_dispatch("SUB-1001")
    assert retried.dispatch_status == "sent"
    assert retried.external_action_taken is True
    assert len(retried.dispatch_attempts) == 2
    first, second = retried.dispatch_attempts
    assert first.status == "failed" and second.status == "sent"
    assert second.attempt == 2
    assert second.backoff_seconds == 30  # the schedule's first retry delay
    assert second.http_status == 200

    # A sent dispatch has nothing left to retry.
    with pytest.raises(ValueError, match="nothing to retry"):
        service.retry_dispatch("SUB-1001")


def test_retry_backoff_schedule_doubles_and_caps(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("down")

    service, _decided = approve_with_urlopen(monkeypatch, boom)
    for _ in range(3):
        service.retry_dispatch("SUB-1001")
    backoffs = [a.backoff_seconds for a in service.get("SUB-1001").dispatch_attempts]
    assert backoffs == [0, 30, 60, 120]


def test_retry_requires_an_approved_failed_record(monkeypatch):
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    # Not approved yet.
    with pytest.raises(ValueError, match="not approved"):
        service.retry_dispatch("SUB-1001")
    # Approved, but no webhook was configured — no dispatch exists.
    decided = service.approve("SUB-1001", approver="U")
    assert decided.dispatch_status is None and decided.dispatch_attempts == []
    with pytest.raises(ValueError, match="nothing to retry"):
        service.retry_dispatch("SUB-1001")
    # Unknown id.
    with pytest.raises(KeyError):
        service.retry_dispatch("SUB-9999")


def test_ledger_survives_a_store_round_trip(monkeypatch, tmp_path):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", boom)
    store = SQLiteStore(tmp_path / "ledger.db")
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U")

    reloaded = SQLiteStore(tmp_path / "ledger.db").get("SUB-1001")
    assert reloaded is not None
    assert reloaded.dispatch_status == "failed"
    assert len(reloaded.dispatch_attempts) == 1
    assert reloaded.result.dispatch_attempts[0].status == "failed"


def test_audit_export_counts_dispatch_attempts(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    service, _decided = approve_with_urlopen(monkeypatch, boom)
    service.retry_dispatch("SUB-1001")
    rows = service.audit_rows()
    row = next(r for r in rows if r["submission_id"] == "SUB-1001")
    assert row["dispatch_status"] == "failed"
    assert row["dispatch_attempts"] == 2


def test_api_dispatch_ledger_and_retry_guards(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    ledger = client.get("/submissions/SUB-1001/dispatch")
    assert ledger.status_code == 200
    assert ledger.json() == {
        "submission_id": "SUB-1001",
        "dispatch_status": None,
        "attempts": [],
    }
    # Nothing dispatched (no webhook configured) -> retry is a 422,
    # unknown submission -> 404 on both endpoints.
    retry = client.post("/submissions/SUB-1001/dispatch/retry")
    assert retry.status_code == 422
    assert client.get("/submissions/SUB-9999/dispatch").status_code == 404
    assert client.post("/submissions/SUB-9999/dispatch/retry").status_code == 404
