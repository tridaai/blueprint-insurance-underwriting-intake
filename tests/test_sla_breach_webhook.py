"""Round 8: the SLA-breach webhook.

The queue has flagged breaches since round 7 and the status note
drafts for them, but nothing outside the service was ever told.
Now the first observation of a breach fires one signed
``sla_breach`` event through the same output-routing seam as
approvals — recorded on the record (the record is the dedupe),
with its own delivery ledger and an operator retry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

import underwriting_agent.service as service_module
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore, SQLiteStore

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _record(
    sid: str,
    submitted_at: str,
    tier: str = "moderate",
    status: str = "awaiting_underwriter",
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _service_with(*records: ApprovalRecord) -> SubmissionService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def sink(monkeypatch):
    """A capturing urlopen: every request body lands in ``calls``."""
    calls: list[dict] = []

    def urlopen(request, timeout=None):
        calls.append(
            {
                "body": json.loads(request.data.decode("utf-8")),
                "headers": dict(request.headers),
            }
        )
        return FakeResponse(204)

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", urlopen)
    return calls


def test_first_observation_fires_a_signed_breach_event(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "s3cret")
    service = _service_with(_record("B1", "2026-10-01"))  # 9d old, budget 5
    out = service.notify_sla_breaches(now=NOW)
    assert out["breaches"] == 1
    assert out["fired"] == [{"submission_id": "B1", "dispatch_status": "sent"}]
    assert len(sink) == 1
    body = sink[0]["body"]
    assert body["event"] == "sla_breach"
    assert body["submission_id"] == "B1"
    assert body["status"] == "awaiting_underwriter"
    assert body["risk_tier"] == "moderate"
    assert body["sla"] == {"budget_days": 5, "age_days": 9, "days_over": 4}
    assert body["detected_at"] == "2026-10-10T00:00:00+00:00"
    # Signed like every dispatch through the seam.
    expected = hmac.new(
        b"s3cret", json.dumps(body).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert sink[0]["headers"]["X-trida-signature"] == f"sha256={expected}"
    # The notice + its own ledger are on the record; the approval
    # ledger is untouched (a different event, a different ledger).
    record = service._get_store().get("B1")
    assert record.sla_breach_notice["first_observed_at"] == "2026-10-10T00:00:00+00:00"
    assert record.sla_breach_notice["dispatch_status"] == "sent"
    assert len(record.sla_breach_attempts) == 1
    assert record.sla_breach_attempts[0]["status"] == "sent"
    assert record.dispatch_attempts == []


def test_the_notice_is_the_dedupe_second_check_never_refires(sink):
    service = _service_with(_record("B1", "2026-10-01"))
    service.notify_sla_breaches(now=NOW)
    out = service.notify_sla_breaches(now=NOW)
    assert out["fired"] == []
    assert out["already_notified"] == ["B1"]
    assert len(sink) == 1


def test_within_budget_and_decided_cases_never_fire(sink):
    service = _service_with(
        _record("W1", "2026-10-09"),  # 1 day old, within the 5-day budget
        _record("D1", "2026-09-01", status="approved"),  # decided: not queue
    )
    out = service.notify_sla_breaches(now=NOW)
    assert out["breaches"] == 0
    assert out["fired"] == []
    assert sink == []


def test_no_webhook_configured_records_nothing(monkeypatch):
    # No ACTION_WEBHOOK_URL, no urlopen patch: the seam returns None.
    service = _service_with(_record("B1", "2026-10-01"))
    out = service.notify_sla_breaches(now=NOW)
    assert out["fired"] == []
    assert out["unannounced_no_webhook"] == ["B1"]
    record = service._get_store().get("B1")
    assert record.sla_breach_notice is None
    assert record.sla_breach_attempts == []


def test_failed_firing_is_ledgered_not_refired_and_retryable(monkeypatch):
    calls: list[dict] = []
    state = {"status": 500}

    def urlopen(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(state["status"])

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", urlopen)
    service = _service_with(_record("B1", "2026-10-01"))

    out = service.notify_sla_breaches(now=NOW)
    assert out["fired"] == [{"submission_id": "B1", "dispatch_status": "failed"}]
    record = service._get_store().get("B1")
    assert record.sla_breach_notice["dispatch_status"] == "failed"
    assert record.sla_breach_attempts[0]["http_status"] == 500

    # A later check does NOT re-fire on its own — the failure waits
    # for an operator, like every failed dispatch in this system.
    service.notify_sla_breaches(now=NOW)
    assert len(calls) == 1

    state["status"] = 200
    view = service.retry_sla_breach_notice("B1")
    assert view["notice"]["dispatch_status"] == "sent"
    assert len(view["attempts"]) == 2
    assert len(calls) == 2
    # The retry carries the ORIGINAL breach snapshot — the age at
    # first observation — not a recomputed one.
    assert calls[1]["sla"] == {"budget_days": 5, "age_days": 9, "days_over": 4}
    assert calls[1]["detected_at"] == "2026-10-10T00:00:00+00:00"

    # Nothing left to retry once delivered.
    with pytest.raises(ValueError, match="not in a failed state"):
        service.retry_sla_breach_notice("B1")


def test_retry_guards(monkeypatch):
    service = _service_with(_record("B1", "2026-10-01"))
    with pytest.raises(KeyError):
        service.retry_sla_breach_notice("NOPE")
    with pytest.raises(ValueError, match="No SLA breach notice"):
        service.retry_sla_breach_notice("B1")
    assert service.sla_breach("NOPE") is None
    view = service.sla_breach("B1")
    assert view == {"submission_id": "B1", "notice": None, "attempts": []}


def test_breach_notice_survives_a_store_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "records.db")
    record = _record("B1", "2026-10-01")
    record.sla_breach_notice = {
        "first_observed_at": "2026-10-10T00:00:00+00:00",
        "dispatch_status": "sent",
    }
    record.sla_breach_attempts = [{"attempt": 1, "status": "sent"}]
    store.save(record)
    loaded = store.get("B1")
    assert loaded.sla_breach_notice["dispatch_status"] == "sent"
    assert loaded.sla_breach_attempts == [{"attempt": 1, "status": "sent"}]


@pytest.fixture()
def client(monkeypatch):
    """The API over a fresh in-memory service (test_api.py's shape)."""
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def test_breach_endpoints(client, sink):
    from underwriting_agent.samples import load_sample_submissions

    sample = dict(
        next(
            s for s in load_sample_submissions() if s["submission_id"] == "SUB-1001"
        )
    )
    sample["submitted_at"] = "2020-01-01"  # ancient: breached under any tier
    assert client.post("/submissions/analyze", json=sample).status_code == 200

    out = client.post("/queue/sla-notifications")
    assert out.status_code == 200
    body = out.json()
    assert body["breaches"] >= 1
    assert {"submission_id": "SUB-1001", "dispatch_status": "sent"} in body["fired"]

    view = client.get("/submissions/SUB-1001/sla-breach")
    assert view.status_code == 200
    assert view.json()["notice"]["dispatch_status"] == "sent"
    assert len(view.json()["attempts"]) == 1

    # The queue item itself reports the notice now.
    queue = client.get("/queue").json()
    items = {
        item["submission_id"]: item
        for bucket in queue["buckets"]
        for item in bucket["submissions"]
    }
    assert items["SUB-1001"]["sla_breach_notice"]["dispatch_status"] == "sent"

    assert client.get("/submissions/NOPE/sla-breach").status_code == 404
    retry = client.post("/submissions/SUB-1001/sla-breach/retry")
    assert retry.status_code == 422  # delivered — nothing to retry
