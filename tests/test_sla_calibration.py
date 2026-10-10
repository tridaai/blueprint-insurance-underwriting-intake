"""Round 12: the SLA-budget calibration report.

The diff thresholds have had a what-if since round 7
(``trigger_calibration`` replays the renewals at candidate
settings); the queue SLA budgets only had the proposal scan,
which answers "is there a move?" but not "what does the wait
actually look like?". The calibration report is the budgets'
replay: per tier, the decided cases' waits as a distribution
against the budget in force (its source named), and a sweep of
candidate budgets whose rows carry breach counts, decision
splits, and the moved set — the exact one-sided evidence the
SLA proposals are drafted from, readable before anyone drafts.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    SLA_SETTINGS_KEY,
    sla_budget_calibration,
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


def _records() -> list[ApprovalRecord]:
    return [
        # Moderate (budget 5): waits 2✓, 4✓, 6✗, 8✗, 9✓.
        _decided("M-2", "moderate", 2),
        _decided("M-4", "moderate", 4),
        _decided("M-6", "moderate", 6, "rejected"),
        _decided("M-8", "moderate", 8, "rejected"),
        _decided("M-9", "moderate", 9),
        # High (budget 3): two rejected at exactly 3, one quick approval.
        _decided("H-3A", "high", 3, "rejected"),
        _decided("H-3B", "high", 3, "rejected"),
        _decided("H-1", "high", 1),
        # Still awaiting: not evidence, however long it has sat.
        ApprovalRecord(
            result=AgentResult(
                submission_id="W-1",
                status="awaiting_underwriter",
                insured_name="Insured W-1",
                triage=Triage(outcome="borderline", risk_tier="moderate"),
            ),
            submission={
                "insured_name": "Insured W-1",
                "broker_name": "Broker Q",
                "submitted_at": "2026-08-01",
            },
        ),
    ]


BUDGETS = {"elevated": 2, "high": 3, "moderate": 5, "low": 7}


def _tier(report: dict, tier: str) -> dict:
    return next(t for t in report["tiers"] if t["tier"] == tier)


def _row(tier_block: dict, value: int) -> dict:
    return next(r for r in tier_block["sweeps"] if r["value"] == value)


# ---------------------------------------------------------------------------
# The distribution against the budget in force
# ---------------------------------------------------------------------------


def test_report_centres_on_the_budget_in_force_with_its_source():
    report = sla_budget_calibration(_records(), BUDGETS)
    moderate = _tier(report, "moderate")
    assert moderate["budget_in_force"] == 5
    assert moderate["budget_source"] == "environment"
    assert moderate["decided_cases"] == 5
    assert moderate["waits"] == {
        "min": 2, "p50": 6, "p75": 8, "p90": 9, "max": 9, "mean": 5.8,
    }
    at = moderate["at_current"]
    assert at["is_current"] is True and at["direction"] is None
    assert at["breached"] == 3  # waits 6, 8, 9 sat past the 5-day budget
    assert at["breached_decisions"] == {"approved": 1, "rejected": 2}
    assert at["within_decisions"] == {"approved": 2, "rejected": 0}
    assert at["moved_vs_current"] == []
    assert at["proposal_ready"] is False
    # The awaiting case is never weighed as evidence.
    assert report["decided_cases_weighed"] == 8


def test_a_tier_without_decided_cases_reports_an_empty_distribution():
    report = sla_budget_calibration(_records(), BUDGETS)
    elevated = _tier(report, "elevated")
    assert elevated["decided_cases"] == 0
    assert elevated["waits"] == {
        "min": None, "p50": None, "p75": None, "p90": None,
        "max": None, "mean": None,
    }
    assert elevated["at_current"]["breached"] == 0
    assert all(not r["proposal_ready"] for r in elevated["sweeps"])


# ---------------------------------------------------------------------------
# The sweep: moved sets and the proposal-ready bar
# ---------------------------------------------------------------------------


def test_sweep_rows_carry_the_moved_set_and_its_decisions():
    report = sla_budget_calibration(_records(), BUDGETS)
    moderate = _tier(report, "moderate")

    lower = _row(moderate, 3)
    assert lower["direction"] == "lower"
    assert lower["breached"] == 4  # waits 4, 6, 8, 9
    # Tightening to 3 newly flags only the wait-4 case — approved.
    assert lower["moved_vs_current"] == [
        {"submission_id": "M-4", "wait_days": 4, "decision": "approved"}
    ]
    assert lower["moved_decisions"] == {"approved": 1, "rejected": 0}
    # A tighter budget whose newly flagged cases include an
    # approval is not one-sided evidence — no proposal from it.
    assert lower["one_sided"] is False
    assert lower["proposal_ready"] is False

    raise_row = _row(moderate, 7)
    assert raise_row["direction"] == "raise"
    assert raise_row["breached"] == 2  # waits 8, 9
    # Loosening to 7 un-flags only the wait-6 case — rejected.
    assert raise_row["moved_vs_current"] == [
        {"submission_id": "M-6", "wait_days": 6, "decision": "rejected"}
    ]
    assert raise_row["one_sided"] is False
    assert raise_row["proposal_ready"] is False


def test_a_one_sided_moved_set_is_marked_proposal_ready():
    report = sla_budget_calibration(_records(), BUDGETS)
    high = _tier(report, "high")
    row = _row(high, 2)
    # Tightening high 3 → 2 newly flags both wait-3 cases, and
    # underwriters rejected every one of them: the exact evidence
    # draft_sla_proposals drafts from.
    assert row["moved_decisions"] == {"approved": 0, "rejected": 2}
    assert {m["submission_id"] for m in row["moved_vs_current"]} == {
        "H-3A", "H-3B",
    }
    assert row["one_sided"] is True
    assert row["proposal_ready"] is True
    # The budget in force is always a row in its own sweep.
    assert _row(high, 3)["is_current"] is True


def test_the_report_is_read_only_and_deterministic():
    records = _records()
    before = copy.deepcopy([r.result.model_dump() for r in records])
    first = sla_budget_calibration(records, BUDGETS)
    second = sla_budget_calibration(records, BUDGETS)
    assert first == second
    assert [r.result.model_dump() for r in records] == before


# ---------------------------------------------------------------------------
# Service + API: budgets in force, per-tier provenance
# ---------------------------------------------------------------------------


def test_service_report_names_the_settings_record_as_a_tiers_source():
    store = InMemoryStore()
    for record in _records():
        store.save(record)
    # An accepted proposal moved only the moderate budget (5 → 4).
    store.save_document(
        SLA_SETTINGS_KEY,
        {
            "budgets": {"elevated": 2, "high": 3, "moderate": 4, "low": 7},
            "updated_by": "U. Writer",
            "updated_at": "2026-10-09T12:00:00+00:00",
            "proposal_id": "PROP-0001",
            "history": [
                {"proposal_id": "PROP-0001", "dimension": "moderate",
                 "from": 5, "to": 4, "actor": "U. Writer",
                 "at": "2026-10-09T12:00:00+00:00"},
            ],
        },
    )
    service = SubmissionService(store=store, checkpointer=False)
    report = service.sla_budget_calibration()
    moderate = _tier(report, "moderate")
    assert moderate["budget_in_force"] == 4
    assert moderate["budget_source"] == "accepted_proposal"
    assert _tier(report, "high")["budget_source"] == "environment"
    # Reading the report wrote nothing: no proposals appeared.
    assert store.get_document("calibration_proposals") is None


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_calibration_endpoint(client):
    res = client.get("/queue/sla/calibration")
    assert res.status_code == 200
    body = res.json()
    assert [t["tier"] for t in body["tiers"]] == [
        "elevated", "high", "moderate", "low",
    ]
    assert body["budgets"]["moderate"] == 5
    assert "Read-only" in body["note"]
