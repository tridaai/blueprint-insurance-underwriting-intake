"""Round 9: SLA-budget calibration proposals.

The diff thresholds learned to tune themselves from decision
evidence in round 8; the queue's SLA budgets were still pure
configuration. Now the decided cases' waits are replayed the
same way: a budget move drafts a proposal only when the cases it
moves are entirely decided and one-sided, a human accepts it
into a second settings record, and the queue measures its cases
against the budgets that record resolves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from underwriting_agent.proposals import (
    decide_proposal,
    decided_wait_cases,
    draft_sla_proposals,
    effective_sla_budgets,
    load_sla_settings,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore


def _record(
    sid: str,
    submitted_at: str,
    decided_at: str | None,
    tier: str = "moderate",
    status: str = "approved",
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            decided_at=decided_at,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _store_with(records: list[ApprovalRecord]) -> InMemoryStore:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return store


# ---------------------------------------------------------------------------
# The wait replay
# ---------------------------------------------------------------------------


def test_decided_wait_cases_measures_decided_waits_only():
    records = [
        # Submitted Sep 20, decided Oct 1: an 11-day wait.
        _record("W1", "2026-09-20", "2026-10-01T09:00:00+00:00"),
        # Still awaiting: not evidence, however long it has sat.
        _record("W2", "2026-09-01", None, status="awaiting_underwriter"),
        # Decided but the decision carries no timestamp: unusable.
        _record("W3", "2026-09-20", None, status="rejected"),
        # Decided the day it arrived: a 0-day wait is real evidence.
        _record("W4", "2026-10-02", "2026-10-02T18:30:00+00:00", status="rejected"),
    ]
    cases = decided_wait_cases(records)
    assert {c["submission_id"]: c["wait_days"] for c in cases} == {
        "W1": 11,
        "W4": 0,
    }
    assert cases[0]["risk_tier"] == "moderate"
    assert cases[0]["status"] == "approved"


# ---------------------------------------------------------------------------
# Drafting from one-sided evidence
# ---------------------------------------------------------------------------


def test_raise_proposal_when_breached_cases_were_approved_anyway():
    # Moderate budget is 5 days. This case waited 8 and was
    # approved: the breach alarm fired and decided nothing.
    store = _store_with(
        [_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00")]
    )
    out = draft_sla_proposals(store, store.all_records())
    assert len(out["drafted"]) == 1
    proposal = out["drafted"][0]
    assert proposal["kind"] == "sla_budget"
    assert proposal["dimension"] == "moderate"
    assert proposal["current_value"] == 5
    # 6 and 7 do not un-flag an 8-day wait; 10 does, and beats 14
    # on the smallest-change tie-break.
    assert proposal["proposed_value"] == 10
    assert proposal["direction"] == "raise"
    assert proposal["evidence"]["moved_ids"] == ["M-OK"]
    assert proposal["evidence"]["waits"] == {"M-OK": 8}
    assert proposal["evidence"]["moved_decisions"] == {
        "approved": 1,
        "rejected": 0,
        "undecided": 0,
    }
    assert "approved every one" in proposal["rationale"]


def test_lower_proposal_when_flagged_cases_were_all_rejected():
    # High budget is 3 days. Two cases decided inside it — waits
    # of 2 and 3 days — and underwriters rejected both: a tighter
    # budget would have flagged cases that turned out bad.
    store = _store_with(
        [
            _record("H-1", "2026-09-28", "2026-09-30T09:00:00+00:00",
                    tier="high", status="rejected"),
            _record("H-2", "2026-09-27", "2026-09-30T09:00:00+00:00",
                    tier="high", status="rejected"),
        ]
    )
    out = draft_sla_proposals(store, store.all_records())
    assert len(out["drafted"]) == 1
    proposal = out["drafted"][0]
    assert proposal["dimension"] == "high"
    assert proposal["direction"] == "lower"
    # A 2-day budget flags only H-2; a 1-day budget flags both —
    # most moved cases wins.
    assert proposal["proposed_value"] == 1
    assert set(proposal["evidence"]["moved_ids"]) == {"H-1", "H-2"}
    assert "rejected every one" in proposal["rationale"]


def test_no_proposal_when_the_moved_cases_are_two_sided():
    store = _store_with(
        [
            _record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"),
            _record("M-BAD", "2026-09-24", "2026-10-03T10:00:00+00:00",
                    status="rejected"),
        ]
    )
    # Both moderate cases breached the 5-day budget (waits 8 and
    # 9); one was approved, one rejected — the alarm's evidence
    # points both ways, so no budget move is proposed.
    out = draft_sla_proposals(store, store.all_records())
    assert out["drafted"] == []


def test_no_proposal_from_undecided_cases():
    store = _store_with(
        [_record("M-WAIT", "2026-08-01", None, status="awaiting_underwriter")]
    )
    out = draft_sla_proposals(store, store.all_records())
    assert out["drafted"] == []
    assert out["proposals"] == []


def test_open_tier_and_rejected_direction_bind_the_scan():
    store = _store_with(
        [_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00")]
    )
    first = draft_sla_proposals(store, store.all_records())
    assert len(first["drafted"]) == 1
    # A live draft is the tier's one open question.
    assert draft_sla_proposals(store, store.all_records())["drafted"] == []
    decide_proposal(store, "PROP-0001", "U. Writer", accept=False)
    # New evidence in the same direction does not re-ask it.
    store.save(
        _record("M-OK2", "2026-09-20", "2026-09-29T10:00:00+00:00")
    )
    out = draft_sla_proposals(store, store.all_records())
    assert out["drafted"] == []
    assert [p["status"] for p in out["proposals"]] == ["rejected"]


# ---------------------------------------------------------------------------
# Deciding: acceptance writes the budgets the queue reads
# ---------------------------------------------------------------------------


def test_acceptance_writes_sla_settings_and_the_queue_uses_them():
    seven_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=7)
    ).date().isoformat()
    store = _store_with(
        [
            _record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"),
            # Awaiting, submitted 7 days before the real clock the
            # queue reads — past the 5-day moderate budget, inside a
            # raised 10-day one, whatever day the suite runs.
            _record("M-WAIT", seven_days_ago, None, status="awaiting_underwriter"),
        ]
    )
    service = SubmissionService(store=store, checkpointer=False)
    assert service.queue()["breach_count"] == 1
    assert service.queue()["sla_budgets_source"] == "environment"

    proposal = service.draft_sla_budget_proposals()["drafted"][0]
    decided = service.decide_calibration_proposal(
        proposal["proposal_id"], "U. Writer", accept=True, note="waits were fine"
    )
    assert decided["status"] == "accepted"
    assert decided["settings_written"]["moderate"] == 10

    assert effective_sla_budgets(store)["moderate"] == 10
    assert effective_sla_budgets(store)["high"] == 3  # untouched tiers stay env
    settings = load_sla_settings(store)
    assert settings["updated_by"] == "U. Writer"
    assert settings["history"] == [
        {
            "proposal_id": proposal["proposal_id"],
            "dimension": "moderate",
            "from": 5,
            "to": 10,
            "actor": "U. Writer",
            "at": settings["history"][0]["at"],
        }
    ]
    view = service.sla_budget_settings()
    assert view["source"] == "accepted_proposal"
    assert view["budgets"]["moderate"] == 10
    # The queue now measures the waiting case against 10 days.
    queue = service.queue()
    assert queue["sla_budgets"]["moderate"] == 10
    assert queue["breach_count"] == 0
    assert queue["sla_budgets_source"] == "accepted_proposal"


def test_rejection_leaves_budgets_untouched():
    store = _store_with(
        [_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00")]
    )
    proposal = draft_sla_proposals(store, store.all_records())["drafted"][0]
    decide_proposal(store, proposal["proposal_id"], "U. Writer", accept=False)
    assert load_sla_settings(store) is None
    assert effective_sla_budgets(store)["moderate"] == 5


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_env(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app), api_module.service


def test_sla_proposal_endpoints(api_env):
    client, service = api_env
    store = service._get_store()
    store.save(_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))

    settings = client.get("/queue/sla/settings").json()
    assert settings["source"] == "environment"
    assert settings["budgets"]["moderate"] == 5

    drafted = client.post("/queue/sla-proposals/draft").json()
    assert len(drafted["drafted"]) == 1
    proposal_id = drafted["drafted"][0]["proposal_id"]

    listing = client.get("/calibration/proposals").json()
    assert [p["proposal_id"] for p in listing["proposals"]] == [proposal_id]
    assert listing["proposals"][0]["kind"] == "sla_budget"

    decided = client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert decided.status_code == 200
    assert decided.json()["settings_written"]["moderate"] == 10

    settings = client.get("/queue/sla/settings").json()
    assert settings["source"] == "accepted_proposal"
    assert settings["budgets"]["moderate"] == 10
