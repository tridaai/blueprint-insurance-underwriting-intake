"""Round 13: calibration snapshots — the report, kept.

The SLA calibration report is computed live and evaporates on
read, like the ops digest did before round 11. A budget debate
cites evidence as of a date, so an operator can now keep the
report — budgets in force, decided cases weighed, per tier the
wait distribution and which candidate rows were proposal-ready —
dated and under their name. The diff between two kept reports
reads the proposal-ready rows' movement: which values became
draftable evidence and which stopped being, beside the budget
and decided-count movement that explains why.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    build_calibration_snapshot,
    calibration_snapshot_diff,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

BASE = date(2026, 9, 1)


def _decided(
    sid: str, tier: str, wait_days: int, status: str = "approved"
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            decided_at=(BASE + timedelta(days=wait_days)).isoformat(),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": BASE.isoformat(),
        },
    )


def _service() -> SubmissionService:
    store = InMemoryStore()
    # High (budget 3): waits 3✗, 3✗, 1✓ — rows 1 and 2 (lower)
    # both move only the two rejected cases: proposal-ready.
    for record in (
        _decided("H-3A", "high", 3, "rejected"),
        _decided("H-3B", "high", 3, "rejected"),
        _decided("H-1", "high", 1),
        # Moderate (budget 5): one fast approval — no ready row.
        _decided("M-2", "moderate", 2),
    ):
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


# ---------------------------------------------------------------------------
# Keeping the report
# ---------------------------------------------------------------------------


def test_a_snapshot_keeps_budgets_sources_and_ready_rows():
    service = _service()
    snapshot = service.snapshot_sla_calibration("Ops Lead")
    assert snapshot["snapshot_id"] == "cal-0001"
    assert snapshot["taken_by"] == "Ops Lead"
    assert snapshot["taken_at"]
    assert snapshot["budgets"] == {
        "elevated": 2, "high": 3, "moderate": 5, "low": 7,
    }
    assert snapshot["decided_cases_weighed"] == 4
    tiers = {t["tier"]: t for t in snapshot["tiers"]}
    assert tiers["high"]["budget_in_force"] == 3
    assert tiers["high"]["budget_source"] == "environment"
    assert tiers["high"]["decided_cases"] == 3
    assert tiers["high"]["proposal_ready_rows"] == [1, 2]
    assert tiers["moderate"]["proposal_ready_rows"] == []
    # Keeping a snapshot changes nothing it reports on.
    assert service.effective_sla_budgets()["high"] == 3
    assert service.sla_calibration_snapshots() == [snapshot]


def test_snapshots_accumulate_in_the_order_kept():
    service = _service()
    first = service.snapshot_sla_calibration("Ops Lead")
    second = service.snapshot_sla_calibration("Night Ops")
    assert (first["snapshot_id"], second["snapshot_id"]) == (
        "cal-0001",
        "cal-0002",
    )
    assert [
        s["snapshot_id"] for s in service.sla_calibration_snapshots()
    ] == ["cal-0001", "cal-0002"]


# ---------------------------------------------------------------------------
# The diff: proposal-ready movement, with its context
# ---------------------------------------------------------------------------


def test_the_diff_reports_ready_rows_that_cleared_as_evidence_grew():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead")
    # A high-tier case decided in 2 days, approved: row 1's moved
    # set (waits 2..3) now mixes it with the two rejections, so
    # row 1 stops being proposal-ready; row 2's moved set (the
    # two 3-day rejections) is untouched and stays ready.
    store = service._get_store()
    store.save(_decided("H-2", "high", 2))
    service.snapshot_sla_calibration("Ops Lead")

    out = service.sla_calibration_snapshot_diff()
    assert out["snapshot_count"] == 2
    diff = out["diff"]
    assert diff["from_snapshot"] == "cal-0001"
    assert diff["to_snapshot"] == "cal-0002"
    assert diff["budgets_changed"] is False
    high = diff["tiers"]["high"]
    assert high["proposal_ready_entered"] == []
    assert high["proposal_ready_cleared"] == [1]
    assert high["decided_cases"] == {"from": 3, "to": 4, "change": 1}
    assert high["budget_in_force"] == {
        "from": 3, "to": 3, "changed": False,
    }
    assert diff["tiers"]["moderate"]["proposal_ready_entered"] == []


def test_the_diff_reads_a_budget_change_between_snapshots():
    older = build_calibration_snapshot(
        {
            "budgets": {"high": 3},
            "decided_cases_weighed": 3,
            "tiers": [
                {
                    "tier": "high",
                    "budget_in_force": 3,
                    "budget_source": "environment",
                    "decided_cases": 3,
                    "waits": {"min": 1, "max": 3},
                    "sweeps": [{"value": 2, "proposal_ready": True}],
                }
            ],
        },
        "cal-0001",
        "Ops Lead",
        taken_at="2026-10-01T09:00:00+00:00",
    )
    newer = build_calibration_snapshot(
        {
            "budgets": {"high": 2},
            "decided_cases_weighed": 3,
            "tiers": [
                {
                    "tier": "high",
                    "budget_in_force": 2,
                    "budget_source": "accepted_proposal",
                    "decided_cases": 3,
                    "waits": {"min": 1, "max": 3},
                    "sweeps": [{"value": 2, "proposal_ready": False}],
                }
            ],
        },
        "cal-0002",
        "Ops Lead",
        taken_at="2026-10-08T09:00:00+00:00",
    )
    diff = calibration_snapshot_diff(older, newer)
    assert diff["budgets_changed"] is True
    assert diff["tiers"]["high"]["budget_in_force"] == {
        "from": 3, "to": 2, "changed": True,
    }
    # The row that was ready against the old budget is the
    # current row against the new one — cleared, not entered.
    assert diff["tiers"]["high"]["proposal_ready_cleared"] == [2]
    # A tier absent from both snapshots reads as unknown, not zero.
    assert diff["tiers"]["low"]["budget_in_force"]["from"] is None
    assert diff["tiers"]["low"]["decided_cases"]["change"] is None


def test_a_diff_needs_two_snapshots():
    service = _service()
    assert service.sla_calibration_snapshot_diff() == {
        "snapshot_count": 0,
        "diff": None,
    }
    service.snapshot_sla_calibration("Ops Lead")
    assert service.sla_calibration_snapshot_diff() == {
        "snapshot_count": 1,
        "diff": None,
    }


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_api_snapshots_and_diff(client):
    kept = client.post(
        "/queue/sla/calibration/snapshots", json={"actor": "Ops Lead"}
    )
    assert kept.status_code == 200
    assert kept.json()["snapshot_id"] == "cal-0001"

    listed = client.get("/queue/sla/calibration/snapshots").json()
    assert [s["snapshot_id"] for s in listed["snapshots"]] == ["cal-0001"]
    assert client.get("/queue/sla/calibration/snapshots/diff").json() == {
        "snapshot_count": 1,
        "diff": None,
    }

    client.post(
        "/queue/sla/calibration/snapshots", json={"actor": "Ops Lead"}
    )
    diff = client.get("/queue/sla/calibration/snapshots/diff").json()
    assert diff["snapshot_count"] == 2
    assert diff["diff"]["tiers"]["high"]["proposal_ready_entered"] == []


def test_api_snapshot_requires_an_actor(client):
    res = client.post("/queue/sla/calibration/snapshots", json={})
    assert res.status_code == 422
