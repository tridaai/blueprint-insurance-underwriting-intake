"""Round 4: decision timestamps + queue aging.

Decisions record WHEN alongside who and why — on the result, in
the feedback entries, in the webhook payload, and in the audit
export. The awaiting queue gains an aging view (buckets by days
since ``submitted_at``, outcome mix, oldest case) so a stranded
queue is visible before a broker complains.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import audit_rows, queue_aging
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _record(sid: str, submitted_at: str, status: str = "awaiting_underwriter",
            outcome: str = "borderline") -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome=outcome, risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def test_queue_aging_buckets_by_submitted_date():
    records = [
        _record("A1", "2026-10-10"),   # 0 days
        _record("A2", "2026-10-08"),   # 2 days
        _record("A3", "2026-10-05"),   # 5 days
        _record("A4", "2026-09-28"),   # 12 days
        _record("A5", "2026-09-01"),   # 39 days
        _record("A6", "not-a-date"),   # unparseable
        _record("A7", "2026-09-01", status="approved"),  # decided: not queue
    ]
    view = queue_aging(records, now=NOW)
    counts = {b["bucket"]: b["count"] for b in view["buckets"]}
    assert counts == {
        "0-1 days": 1,
        "2-3 days": 1,
        "4-7 days": 1,
        "8-14 days": 1,
        "15+ days": 1,
    }
    assert view["total_awaiting"] == 6
    assert view["unknown_age"] == ["A6"]
    assert view["by_outcome"] == {"borderline": 6}
    assert view["oldest"]["submission_id"] == "A5"
    assert view["oldest"]["age_days"] == 39


@pytest.fixture()
def service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def test_decisions_record_decided_at_everywhere(service):
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1009"])
    approved = service.approve("SUB-1001", approver="U. Writer", reason="clean")
    rejected = service.reject("SUB-1009", reviewer="U. Writer", reason="outside appetite")
    for result in (approved, rejected):
        assert result.decided_at is not None
        parsed = datetime.fromisoformat(result.decided_at)
        assert parsed.tzinfo is not None

    rows = {row["submission_id"]: row for row in audit_rows(service._get_store().all_records())}
    assert rows["SUB-1001"]["decided_at"] == approved.decided_at
    assert rows["SUB-1009"]["decided_at"] == rejected.decided_at

    fetched = service.get("SUB-1001")
    assert fetched is not None and fetched.decided_at == approved.decided_at


def test_webhook_payload_carries_decided_at(service, monkeypatch):
    import underwriting_agent.service as service_module

    captured = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        import json

        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    service.analyze(SAMPLES["SUB-1001"])
    result = service.approve("SUB-1001", approver="U. Writer")
    assert captured["payload"]["decided_at"] == result.decided_at


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(api_module.app)


def test_queue_endpoint(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1002"])
    response = client.get("/queue")
    assert response.status_code == 200
    body = response.json()
    assert body["total_awaiting"] == 1
    assert sum(b["count"] for b in body["buckets"]) == 1
    assert body["oldest"]["submission_id"] == "SUB-1002"
