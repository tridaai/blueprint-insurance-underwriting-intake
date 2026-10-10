"""Round 11: ops-digest snapshots and the trend across them.

The ops digest is computed at read time and evaporates: every
read is a fresh count with no memory of the last one, so an
operator cannot tell whether the stuck dispatches are the same
ones as last week or new ones. A snapshot is the digest as an
operator chose to keep it — stored, dated, under their name,
never taken on a timer — and the trend compares the headline
counts across snapshots, including which ids entered and which
cleared each section between the latest two.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import (
    DIGEST_SNAPSHOTS_KEY,
    build_digest_snapshot,
    digest_trend,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore


def _digest(**counts) -> dict:
    base = {
        "stuck_dispatches": 0,
        "breached_queue": 0,
        "follow_through_outstanding": 0,
        "follow_through_escalated": 0,
        "proposals_expiring_soon": 0,
        "proposals_expired": 0,
    }
    base.update(counts)
    base["total_attention"] = (
        base["stuck_dispatches"]
        + base["breached_queue"]
        + base["follow_through_escalated"]
        + base["proposals_expired"]
    )
    return {
        "generated_at": "2026-10-20T09:00:00+00:00",
        "headline": "Ops digest: …",
        "counts": base,
        "stuck_dispatches": [
            {"submission_id": sid} for sid in counts.get("_stuck_ids", [])
        ],
        "breached_queue": [
            {"submission_id": sid} for sid in counts.get("_breach_ids", [])
        ],
        "follow_through_outstanding": [],
        "follow_through_escalated": [],
        "proposals_expiring_soon": [],
        "proposals_expired": [],
    }


def _snapshot(sid: str, at: str, counts: dict, sections: dict) -> dict:
    digest = _digest(**counts)
    digest["generated_at"] = at
    for section, ids in sections.items():
        id_field = "proposal_id" if section.startswith("proposals") else "submission_id"
        digest[section] = [{id_field: i} for i in ids]
    return build_digest_snapshot(digest, sid, "Ops Lead")


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_keeps_counts_and_section_ids():
    digest = _digest(stuck_dispatches=2, _stuck_ids=["S-2", "S-1", "S-1"])
    snap = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert snap["snapshot_id"] == "snap-0001"
    assert snap["taken_at"] == "2026-10-20T09:00:00+00:00"
    assert snap["taken_by"] == "Ops Lead"
    assert snap["headline"] == "Ops digest: …"
    assert snap["counts"]["stuck_dispatches"] == 2
    assert snap["counts"]["total_attention"] == 2
    # Ids are kept sorted and deduped — membership, not raw rows.
    assert snap["sections"]["stuck_dispatches"] == ["S-1", "S-2"]
    assert snap["sections"]["breached_queue"] == []


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_trend_reports_series_deltas_and_movement():
    first = _snapshot(
        "snap-0001", "2026-10-06T09:00:00+00:00",
        {"stuck_dispatches": 2, "_stuck_ids": ["S-1", "S-2"]},
        {"stuck_dispatches": ["S-1", "S-2"]},
    )
    second = _snapshot(
        "snap-0002", "2026-10-13T09:00:00+00:00",
        {"stuck_dispatches": 2},
        {"stuck_dispatches": ["S-2", "S-3"]},
    )
    third = _snapshot(
        "snap-0003", "2026-10-20T09:00:00+00:00",
        {"stuck_dispatches": 1, "breached_queue": 1},
        {"stuck_dispatches": ["S-3"], "breached_queue": ["B-1"]},
    )
    trend = digest_trend([first, second, third])
    assert trend["snapshot_count"] == 3
    assert [s["snapshot_id"] for s in trend["series"]] == [
        "snap-0001", "snap-0002", "snap-0003",
    ]
    delta = trend["deltas"]["stuck_dispatches"]
    assert delta == {"first": 2, "latest": 1, "change": -1, "since_previous": -1}
    assert trend["deltas"]["breached_queue"]["since_previous"] == 1
    movement = trend["movement"]
    assert movement["from_snapshot"] == "snap-0002"
    assert movement["to_snapshot"] == "snap-0003"
    # The count hid churn earlier (S-1 cleared, S-3 entered between
    # snap-0001 and snap-0002); the latest step is a plain clearing.
    assert movement["sections"]["stuck_dispatches"] == {
        "entered": [], "cleared": ["S-2"],
    }
    assert movement["sections"]["breached_queue"] == {
        "entered": ["B-1"], "cleared": [],
    }


def test_trend_with_too_few_snapshots_invents_no_movement():
    empty = digest_trend([])
    assert empty["snapshot_count"] == 0
    assert empty["series"] == []
    assert empty["movement"] is None
    assert empty["deltas"]["total_attention"]["latest"] is None

    one = digest_trend([_snapshot("snap-0001", "2026-10-20T09:00:00+00:00", {}, {})])
    assert one["snapshot_count"] == 1
    assert one["movement"] is None
    assert one["deltas"]["total_attention"]["change"] is None
    assert one["deltas"]["total_attention"]["since_previous"] is None


# ---------------------------------------------------------------------------
# Service + API
# ---------------------------------------------------------------------------


def _breached_record() -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id="LATE-1",
            status="awaiting_underwriter",
            insured_name="Late Co",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": "Late Co",
            "broker_name": "Broker Q",
            "submitted_at": "2026-08-01",
        },
    )


def _stuck_record() -> ApprovalRecord:
    attempted = datetime.now(timezone.utc) - timedelta(days=3)
    return ApprovalRecord(
        result=AgentResult(
            submission_id="STUCK-1",
            status="approved",
            insured_name="Stuck Co",
            triage=Triage(outcome="in_appetite", risk_tier="low"),
        ),
        submission={"insured_name": "Stuck Co", "broker_name": "Broker Q"},
        dispatch_status="failed",
        dispatch_attempts=[
            {
                "attempt": 1,
                "attempted_at": attempted.isoformat(timespec="seconds"),
                "status": "failed",
                "http_status": None,
                "signature_id": None,
                "error": "connection refused",
                "backoff_seconds": 0,
            }
        ],
    )


def test_service_snapshots_accumulate_and_trend():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    store = service._get_store()
    store.save(_breached_record())
    first = service.snapshot_ops_digest("Ops Lead")
    assert first["snapshot_id"] == "snap-0001"
    assert first["taken_by"] == "Ops Lead"
    assert first["counts"]["breached_queue"] == 1
    assert first["sections"]["breached_queue"] == ["LATE-1"]

    store.save(_stuck_record())
    second = service.snapshot_ops_digest("Ops Lead")
    assert second["snapshot_id"] == "snap-0002"
    assert second["counts"]["stuck_dispatches"] == 1

    stored = store.get_document(DIGEST_SNAPSHOTS_KEY)
    assert [s["snapshot_id"] for s in stored["snapshots"]] == [
        "snap-0001", "snap-0002",
    ]
    trend = service.ops_digest_trend()
    assert trend["snapshot_count"] == 2
    assert trend["deltas"]["stuck_dispatches"]["since_previous"] == 1
    assert trend["deltas"]["breached_queue"]["since_previous"] == 0
    assert trend["movement"]["sections"]["stuck_dispatches"]["entered"] == [
        "STUCK-1"
    ]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_snapshot_endpoints(client):
    res = client.post("/ops/digest/snapshots", json={"actor": "Ops Lead"})
    assert res.status_code == 200
    snap = res.json()
    assert snap["snapshot_id"] == "snap-0001"
    assert snap["taken_by"] == "Ops Lead"
    assert snap["taken_at"]

    res = client.post("/ops/digest/snapshots", json={"actor": "Ops Lead"})
    assert res.json()["snapshot_id"] == "snap-0002"

    res = client.get("/ops/digest/snapshots")
    assert res.status_code == 200
    assert len(res.json()["snapshots"]) == 2

    res = client.get("/ops/digest/trend")
    assert res.status_code == 200
    trend = res.json()
    assert trend["snapshot_count"] == 2
    assert trend["movement"]["from_snapshot"] == "snap-0001"


def test_snapshot_requires_an_actor(client):
    res = client.post("/ops/digest/snapshots", json={})
    assert res.status_code == 422
