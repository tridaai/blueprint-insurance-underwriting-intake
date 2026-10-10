"""Round 17: one sweep for both tellings.

The breach check and the escalation check are two rungs of one
ladder — a case nobody was told about, and a case that was
told, worked, and still waits — and operators run them
together every morning. They ran as two endpoints and two
console buttons, so the second telling waited on somebody
remembering it. ``POST /queue/sla-notifications`` now runs the
escalation sweep in the same call and reports both firings in
one response: the breach report as it always read, with the
escalation report riding under ``escalations``. Nothing is
merged — each telling keeps its own notice, its own ledger,
its own dedupe — and the escalation-only endpoint stays for
schedulers that want only that rung.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import underwriting_agent.service as service_module
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


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


def _attempt(at: datetime, status: str = "failed") -> dict:
    return {
        "attempt": 1,
        "attempted_at": at.isoformat(timespec="seconds"),
        "status": status,
        "http_status": 500 if status == "failed" else 204,
        "signature_id": None,
        "error": "connection refused" if status == "failed" else None,
        "backoff_seconds": 0,
    }


def _rung_record(sid: str, acked_days_ago: int = 35) -> ApprovalRecord:
    """An exhausted breach notice, acknowledged ``acked_days_ago``
    before NOW, on a case still waiting — past the second rung
    (2 × 14 days), its escalation unannounced."""
    observed = NOW - timedelta(days=60)
    record = _record(sid, "2026-08-01")
    record.sla_breach_notice = {
        "first_observed_at": observed.isoformat(timespec="seconds"),
        "submission_id": sid,
        "insured_name": f"Insured {sid}",
        "broker_name": "Broker Q",
        "outcome": "borderline",
        "risk_tier": "moderate",
        "sla": {"budget_days": 5, "age_days": 40, "days_over": 35},
        "dispatch_status": "failed",
    }
    record.sla_breach_attempts = [
        _attempt(observed + timedelta(hours=i)) for i in range(8)
    ]
    record.breach_notice_acknowledgement = {
        "acknowledged_by": "Ops Lead",
        "acknowledged_at": (NOW - timedelta(days=acked_days_ago)).isoformat(
            timespec="seconds"
        ),
        "note": "Called the broker",
    }
    return record


def _service(*records: ApprovalRecord) -> SubmissionService:
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


# --- the combined sweep ------------------------------------------------------


def test_one_sweep_fires_both_tellings_in_ladder_order(sink):
    service = _service(_record("B1", "2026-10-01"), _rung_record("ESC-1"))
    out = service.notify_sla_notifications(now=NOW)
    # The breach report reads as it always did…
    assert out["breaches"] >= 1
    assert {"submission_id": "B1", "dispatch_status": "sent"} in out["fired"]
    # …with the escalation report riding in the same response.
    escalations = out["escalations"]
    assert escalations["escalations"] == 1
    assert escalations["fired"] == [
        {"submission_id": "ESC-1", "dispatch_status": "sent"}
    ]
    # Both events left through the seam, breaches first.
    assert [call["body"]["event"] for call in sink] == [
        "sla_breach",
        "acknowledged_escalation",
    ]
    # Each telling kept its own record state: B1's breach notice,
    # ESC-1's escalation notice (its breach was already told).
    b1 = service._get_store().get("B1")
    assert b1.sla_breach_notice["dispatch_status"] == "sent"
    assert b1.escalation_notice is None
    esc = service._get_store().get("ESC-1")
    assert esc.escalation_notice["dispatch_status"] == "sent"
    assert "ESC-1" in out["already_notified"]


def test_a_second_sweep_fires_nothing_new(sink):
    service = _service(_record("B1", "2026-10-01"), _rung_record("ESC-1"))
    service.notify_sla_notifications(now=NOW)
    out = service.notify_sla_notifications(now=NOW)
    assert out["fired"] == []
    assert "B1" in out["already_notified"]
    assert out["escalations"]["fired"] == []
    assert out["escalations"]["already_notified"] == ["ESC-1"]
    assert len(sink) == 2


def test_the_escalation_only_check_still_runs_alone(sink):
    # The schedulers' endpoint keeps its own sweep: it announces
    # the second rung and does not touch the first.
    service = _service(_record("B1", "2026-10-01"), _rung_record("ESC-1"))
    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["fired"] == [
        {"submission_id": "ESC-1", "dispatch_status": "sent"}
    ]
    assert len(sink) == 1
    assert service._get_store().get("B1").sla_breach_notice is None
    # And the combined sweep afterwards finds only the breach.
    combined = service.notify_sla_notifications(now=NOW)
    assert {"submission_id": "B1", "dispatch_status": "sent"} in combined[
        "fired"
    ]
    assert combined["escalations"]["already_notified"] == ["ESC-1"]


def test_with_no_webhook_both_tellings_stay_unannounced(monkeypatch):
    monkeypatch.delenv("ACTION_WEBHOOK_URL", raising=False)
    service = _service(_record("B1", "2026-10-01"), _rung_record("ESC-1"))
    out = service.notify_sla_notifications(now=NOW)
    assert out["fired"] == []
    assert out["unannounced_no_webhook"] == ["B1"]
    assert out["escalations"]["fired"] == []
    assert out["escalations"]["unannounced_no_webhook"] == ["ESC-1"]
    # Unannounced is unrecorded: configuring the webhook later
    # still announces both crossings.
    assert service._get_store().get("B1").sla_breach_notice is None
    assert service._get_store().get("ESC-1").escalation_notice is None


# --- the API -----------------------------------------------------------------


@pytest.fixture()
def client(sink):
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    store.save(_record("B1", "2020-01-01"))  # breached under any tier
    store.save(_rung_record("ESC-1", acked_days_ago=200))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    from fastapi.testclient import TestClient

    return TestClient(api_module.app)


def test_the_notifications_endpoint_reports_both_tellings(client, sink):
    out = client.post("/queue/sla-notifications")
    assert out.status_code == 200
    body = out.json()
    assert {"submission_id": "B1", "dispatch_status": "sent"} in body["fired"]
    assert body["escalations"]["fired"] == [
        {"submission_id": "ESC-1", "dispatch_status": "sent"}
    ]
    assert [call["body"]["event"] for call in sink] == [
        "sla_breach",
        "acknowledged_escalation",
    ]
    # The escalation-only endpoint still answers on its own, and
    # finds nothing new to say.
    alone = client.post("/queue/escalation-notifications")
    assert alone.status_code == 200
    assert alone.json()["already_notified"] == ["ESC-1"]
    assert len(sink) == 2
