"""Round 16: an escalation notice on the second rung.

An acknowledged breach notice whose case still waits crosses a
first window into the digest's still-waiting section and a
second — twice the follow-through window — onto the queue's
escalated surfaces, but nothing ever left the building: the
breach fired a signed webhook and its second rung fired
nothing. The escalation check now fires one signed
``acknowledged_escalation`` event through the same dispatch
seam, with its own notice and delivery ledger on the record —
the notice is the dedupe, one firing per acknowledgement
episode — an operator retry for a failed firing, and an
exhaustion read that joins the dispatch worklist beside the
breach notices'.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import underwriting_agent.service as service_module
from underwriting_agent.insights import dispatch_overview
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import (
    ApprovalRecord,
    InMemoryStore,
    _dump_record,
    _load_record,
)

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


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


def _rung_record(
    sid: str,
    *,
    acked_days_ago: int | None = 35,
    status: str = "awaiting_underwriter",
) -> ApprovalRecord:
    """An exhausted breach notice, acknowledged ``acked_days_ago``
    before NOW, on a case in ``status``. The default age is past
    the second rung (2 × 14 days)."""
    observed = NOW - timedelta(days=60)
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-08-01",
        },
        sla_breach_notice={
            "first_observed_at": observed.isoformat(timespec="seconds"),
            "submission_id": sid,
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "outcome": "borderline",
            "risk_tier": "moderate",
            "sla": {"budget_days": 5, "age_days": 40, "days_over": 35},
            "dispatch_status": "failed",
        },
        sla_breach_attempts=[
            _attempt(observed + timedelta(hours=i)) for i in range(8)
        ],
        breach_notice_acknowledgement=(
            {
                "acknowledged_by": "Ops Lead",
                "acknowledged_at": (
                    NOW - timedelta(days=acked_days_ago)
                ).isoformat(timespec="seconds"),
                "note": "Called the broker",
            }
            if acked_days_ago is not None
            else None
        ),
    )


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


# --- the firing ---------------------------------------------------------------


def test_the_second_rung_fires_a_signed_escalation_event(sink, monkeypatch):
    monkeypatch.setenv("ACTION_WEBHOOK_SECRET", "s3cret")
    service = _service(_rung_record("ESC-1"))
    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["escalations"] == 1
    assert out["fired"] == [
        {"submission_id": "ESC-1", "dispatch_status": "sent"}
    ]
    assert len(sink) == 1
    body = sink[0]["body"]
    assert body["event"] == "acknowledged_escalation"
    assert body["submission_id"] == "ESC-1"
    assert body["status"] == "awaiting_underwriter"
    assert body["detected_at"] == "2026-10-20T00:00:00+00:00"
    assert body["acknowledged_age_days"] == 35
    assert body["escalation_window_days"] == 28
    assert body["acknowledgement"]["acknowledged_by"] == "Ops Lead"
    assert "X-trida-signature" in sink[0]["headers"]
    # The notice + its own ledger are on the record; the breach
    # ledger and the acknowledgement are untouched.
    record = service._get_store().get("ESC-1")
    assert record.escalation_notice["first_observed_at"] == (
        "2026-10-20T00:00:00+00:00"
    )
    assert record.escalation_notice["dispatch_status"] == "sent"
    assert len(record.escalation_attempts) == 1
    assert len(record.sla_breach_attempts) == 8
    assert record.breach_notice_acknowledgement["note"] == "Called the broker"


def test_the_notice_is_the_dedupe_second_check_never_refires(sink):
    service = _service(_rung_record("ESC-1"))
    service.notify_acknowledged_escalations(now=NOW)
    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["fired"] == []
    assert out["already_notified"] == ["ESC-1"]
    assert len(sink) == 1


def test_only_the_second_rung_fires(sink):
    service = _service(
        # 19 days: still waiting (past one window), not escalated.
        _rung_record("MID-1", acked_days_ago=19),
        # Never acknowledged: not on this ladder at all.
        _rung_record("UNACKED-1", acked_days_ago=None),
        # Past the rung, but the case moved: decided cases are out
        # of the acknowledged read by its own construction.
        _rung_record("DONE-1", status="approved"),
    )
    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["escalations"] == 0
    assert out["fired"] == []
    assert sink == []


def test_no_webhook_configured_records_nothing(monkeypatch):
    service = _service(_rung_record("ESC-1"))
    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["escalations"] == 1
    assert out["fired"] == []
    assert out["unannounced_no_webhook"] == ["ESC-1"]
    record = service._get_store().get("ESC-1")
    assert record.escalation_notice is None
    assert record.escalation_attempts == []


def test_failed_firing_is_ledgered_not_refired_and_retryable(monkeypatch):
    calls: list[dict] = []
    state = {"status": 500}

    def urlopen(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(state["status"])

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", urlopen)
    service = _service(_rung_record("ESC-1"))

    out = service.notify_acknowledged_escalations(now=NOW)
    assert out["fired"] == [
        {"submission_id": "ESC-1", "dispatch_status": "failed"}
    ]
    record = service._get_store().get("ESC-1")
    assert record.escalation_notice["dispatch_status"] == "failed"
    assert record.escalation_attempts[0]["http_status"] == 500

    service.notify_acknowledged_escalations(now=NOW)
    assert len(calls) == 1  # a failure waits for an operator

    state["status"] = 200
    view = service.retry_escalation_notice("ESC-1")
    assert view["notice"]["dispatch_status"] == "sent"
    assert len(view["attempts"]) == 2
    assert len(calls) == 2
    # The retry carries the original snapshot — the acknowledgement
    # age at first observation — not a recomputed one.
    assert calls[1]["acknowledged_age_days"] == 35
    assert calls[1]["detected_at"] == "2026-10-20T00:00:00+00:00"

    with pytest.raises(ValueError, match="not in a failed state"):
        service.retry_escalation_notice("ESC-1")


def test_retry_guards():
    service = _service(_rung_record("ESC-1"))
    with pytest.raises(KeyError):
        service.retry_escalation_notice("NOPE")
    with pytest.raises(ValueError, match="No escalation notice"):
        service.retry_escalation_notice("ESC-1")
    assert service.acknowledged_escalation("NOPE") is None
    view = service.acknowledged_escalation("ESC-1")
    assert view == {"submission_id": "ESC-1", "notice": None, "attempts": []}


def test_escalation_fields_survive_a_store_round_trip():
    record = _rung_record("ESC-1")
    record.escalation_notice = {
        "first_observed_at": "2026-10-20T00:00:00+00:00",
        "dispatch_status": "sent",
    }
    record.escalation_attempts = [{"attempt": 1, "status": "sent"}]
    assert _load_record(_dump_record(record)) == record


# --- the exhaustion read joins the dispatch worklist ---------------------------


def test_an_exhausted_escalation_joins_the_dispatch_worklist():
    record = _rung_record("ESC-1")
    observed = NOW - timedelta(days=3)
    record.escalation_notice = {
        "first_observed_at": observed.isoformat(timespec="seconds"),
        "submission_id": "ESC-1",
        "insured_name": "Insured ESC-1",
        "broker_name": "Broker Q",
        "risk_tier": "moderate",
        "acknowledgement": record.breach_notice_acknowledgement,
        "acknowledged_age_days": 32,
        "escalation_window_days": 28,
        "dispatch_status": "failed",
    }
    record.escalation_attempts = [
        _attempt(observed + timedelta(hours=i)) for i in range(8)
    ]
    overview = dispatch_overview([record], now=NOW)
    assert overview["exhausted_escalation_count"] == 1
    (entry,) = overview["exhausted_escalation_notices"]
    assert entry["submission_id"] == "ESC-1"
    assert entry["notice_kind"] == "acknowledged_escalation"
    assert entry["attempts"] == 8
    assert entry["notice_age_days"] == 3
    assert entry["acknowledgement"]["acknowledged_by"] == "Ops Lead"
    # The breach worklist itself is untouched by the join.
    assert overview["exhausted_breach_count"] == 0


def test_a_delivered_or_decided_escalation_is_not_exhausted():
    sent = _rung_record("ESC-SENT")
    sent.escalation_notice = {"first_observed_at": "2026-10-17T00:00:00+00:00"}
    sent.escalation_attempts = [
        {**_attempt(NOW - timedelta(days=3)), "status": "sent"}
        for _ in range(8)
    ]
    decided = _rung_record("ESC-DONE", status="approved")
    decided.escalation_notice = {
        "first_observed_at": "2026-10-17T00:00:00+00:00"
    }
    decided.escalation_attempts = [
        _attempt(NOW - timedelta(days=3)) for _ in range(8)
    ]
    overview = dispatch_overview([sent, decided], now=NOW)
    assert overview["exhausted_escalation_count"] == 0


# --- the API -----------------------------------------------------------------


@pytest.fixture()
def client(sink):
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    # Aged far past the second rung: the endpoint runs on the
    # real clock, not this file's NOW (the breach endpoints'
    # tests age their fixtures the same way).
    store.save(_rung_record("ESC-1", acked_days_ago=200))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    from fastapi.testclient import TestClient

    return TestClient(api_module.app)


def test_escalation_endpoints(client, sink):
    out = client.post("/queue/escalation-notifications")
    assert out.status_code == 200
    body = out.json()
    assert {"submission_id": "ESC-1", "dispatch_status": "sent"} in body["fired"]
    assert len(sink) == 1
    assert sink[0]["body"]["event"] == "acknowledged_escalation"

    view = client.get("/submissions/ESC-1/acknowledged-escalation")
    assert view.status_code == 200
    assert view.json()["notice"]["dispatch_status"] == "sent"
    assert len(view.json()["attempts"]) == 1

    assert client.get("/submissions/NOPE/acknowledged-escalation").status_code == 404
    retry = client.post("/submissions/ESC-1/acknowledged-escalation/retry")
    assert retry.status_code == 422  # delivered — nothing to retry
