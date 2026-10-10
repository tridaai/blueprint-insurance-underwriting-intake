"""Round 13: exhaustion acknowledgement — a worked state for the worklist.

An exhausted breach notice is a worklist whose only next step is
a human call, but the list had no worked state: a case an
operator had already picked up sat in the digest's count
indistinguishable from one nobody had touched. Acknowledging
records the pickup on the record — who called, when, their
note — and the case leaves the worklist and the digest's count.
The ledger is never rewritten: the webhook still never landed,
and the acknowledged list keeps the case readable with its
acknowledgement, because "we called" is a claim the record
should be able to show.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import dispatch_overview, ops_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import (
    ApprovalRecord,
    InMemoryStore,
    _dump_record,
    _load_record,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


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


def _breach_record(
    sid: str, *, attempts: int = 8, record_status: str = "awaiting_underwriter"
) -> ApprovalRecord:
    observed = NOW - timedelta(days=8)
    ledger = [
        _attempt(observed + timedelta(hours=i)) for i in range(attempts)
    ]
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=record_status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-09-20",
        },
        sla_breach_notice={
            "first_observed_at": observed.isoformat(timespec="seconds"),
            "submission_id": sid,
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "outcome": "borderline",
            "risk_tier": "moderate",
            "sla": {"budget_days": 5, "age_days": 12, "days_over": 7},
            "dispatch_status": "failed",
        },
        sla_breach_attempts=ledger,
    )


def _service(*records: ApprovalRecord) -> SubmissionService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


# ---------------------------------------------------------------------------
# The acknowledgement itself
# ---------------------------------------------------------------------------


def test_acknowledging_records_who_when_and_the_note():
    record = _breach_record("BR-1")
    service = _service(record)
    out = service.acknowledge_exhausted_breach_notice(
        "BR-1", "Ops Lead", note="Called the broker; file is moving", now=NOW
    )
    assert out["acknowledgement"] == {
        "acknowledged_by": "Ops Lead",
        "acknowledged_at": "2026-10-10T12:00:00+00:00",
        "note": "Called the broker; file is moving",
    }
    # The acknowledgement is on the record, and the ledger is
    # exactly as it was — eight failed attempts, never rewritten.
    assert record.breach_notice_acknowledgement == out["acknowledgement"]
    assert len(record.sla_breach_attempts) == 8
    assert record.sla_breach_notice["dispatch_status"] == "failed"
    # It survives a store round-trip like every other record field.
    assert _load_record(_dump_record(record)) == record


def test_acknowledgement_requires_an_exhausted_notice():
    service = _service(_breach_record("BR-1", attempts=3))
    with pytest.raises(ValueError, match="not exhausted"):
        service.acknowledge_exhausted_breach_notice("BR-1", "Ops Lead", now=NOW)
    with pytest.raises(KeyError, match="Unknown submission_id"):
        service.acknowledge_exhausted_breach_notice("NOPE", "Ops Lead", now=NOW)


def test_an_acknowledgement_happens_once():
    record = _breach_record("BR-1")
    service = _service(record)
    service.acknowledge_exhausted_breach_notice("BR-1", "Ops Lead", now=NOW)
    with pytest.raises(ValueError, match="already acknowledged by Ops Lead"):
        service.acknowledge_exhausted_breach_notice(
            "BR-1", "Someone Else", now=NOW
        )


# ---------------------------------------------------------------------------
# The two reads: worklist and worked list
# ---------------------------------------------------------------------------


def test_the_worklist_counts_only_unacknowledged_notices():
    worked = _breach_record("BR-WORKED")
    open_case = _breach_record("BR-OPEN")
    service = _service(worked, open_case)
    service.acknowledge_exhausted_breach_notice(
        "BR-WORKED", "Ops Lead", note="called", now=NOW
    )
    records = service._get_store().all_records()

    overview = dispatch_overview(records, now=NOW)
    assert overview["exhausted_breach_count"] == 1
    assert [
        e["submission_id"] for e in overview["exhausted_breach_notices"]
    ] == ["BR-OPEN"]
    assert overview["exhausted_breach_notices"][0]["acknowledgement"] is None
    assert overview["acknowledged_breach_count"] == 1
    acked = overview["acknowledged_breach_notices"][0]
    assert acked["submission_id"] == "BR-WORKED"
    assert acked["acknowledgement"]["acknowledged_by"] == "Ops Lead"
    assert acked["acknowledgement"]["note"] == "called"
    # Buckets age only the unworked — the worklist's shape is
    # unchanged for everything still on it.
    bucketed = [
        sid
        for bucket in overview["exhausted_breach_age_buckets"]
        for sid in bucket["submission_ids"]
    ]
    assert bucketed == ["BR-OPEN"]


def test_the_digest_stops_counting_an_acknowledged_notice():
    record = _breach_record("BR-1")
    service = _service(record)
    before = ops_digest([record], [], now=NOW)
    assert before["counts"]["exhausted_breach_notices"] == 1
    assert before["counts"]["total_attention"] == 2  # + the breached case

    service.acknowledge_exhausted_breach_notice("BR-1", "Ops Lead", now=NOW)
    after = ops_digest([record], [], now=NOW)
    assert after["counts"]["exhausted_breach_notices"] == 0
    # The queue breach itself still counts — acknowledgement
    # answers "who was told", not "who picks it up".
    assert after["counts"]["breached_queue"] == 1
    assert after["counts"]["total_attention"] == 1
    assert "exhausted breach notice" not in after["headline"]
    acked = after["acknowledged_breach_notices"]
    assert [e["submission_id"] for e in acked] == ["BR-1"]
    assert acked[0]["acknowledgement"]["acknowledged_by"] == "Ops Lead"


def test_service_views_agree_after_an_acknowledgement():
    record = _breach_record("BR-1")
    service = _service(record)
    service.acknowledge_exhausted_breach_notice("BR-1", "Ops Lead", now=NOW)
    assert service.dispatch_ops()["acknowledged_breach_count"] == 1
    assert service.dispatch_ops()["exhausted_breach_count"] == 0
    assert service.ops_digest()["counts"]["exhausted_breach_notices"] == 0


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    store.save(_breach_record("BR-1"))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    return TestClient(api_module.app)


def test_api_acknowledges_and_refuses_a_second_acknowledgement(client):
    res = client.post(
        "/dispatch/exhausted/BR-1/acknowledge",
        json={"actor": "Ops Lead", "note": "Called the broker"},
    )
    assert res.status_code == 200
    assert res.json()["acknowledgement"]["acknowledged_by"] == "Ops Lead"

    overview = client.get("/dispatch").json()
    assert overview["exhausted_breach_count"] == 0
    assert overview["acknowledged_breach_count"] == 1

    again = client.post(
        "/dispatch/exhausted/BR-1/acknowledge", json={"actor": "Ops Lead"}
    )
    assert again.status_code == 422
    assert "already acknowledged" in again.json()["detail"]


def test_api_acknowledge_validates_the_case_and_the_actor(client):
    unknown = client.post(
        "/dispatch/exhausted/NOPE/acknowledge", json={"actor": "Ops Lead"}
    )
    assert unknown.status_code == 404
    anonymous = client.post("/dispatch/exhausted/BR-1/acknowledge", json={})
    assert anonymous.status_code == 422
