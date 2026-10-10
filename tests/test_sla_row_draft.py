"""Round 13: draft an SLA proposal from a calibration row.

The calibration report flags ``proposal_ready`` rows, but the
only drafting path was the scan, which drafts its own best row
per tier. An underwriter reading the report who wanted a
different ready row — the smaller move, the one whose cases they
remember — had no way to ask. ``draft_sla_proposal_from_row``
drafts the chosen (tier, value) row as a proposal carrying the
row's evidence verbatim, under the scan's discipline: one open
question per tier, rejected directions never re-asked, swept
candidates only, and never from a row that is not
proposal-ready.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    decide_proposal,
    draft_sla_proposal_from_row,
    draft_sla_proposals,
    effective_sla_budgets,
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
    # Moderate (budget 5): waits 2✓, 3✗, 4✗, 9✓, 12✗.
    # Row 2 (lower) moves {3✗, 4✗} — the scan's best. Row 3
    # (lower) moves only {4✗}. Row 10 (raise) moves {9✓}. Row 7
    # (raise) moves nothing.
    return [
        _decided("M-2", "moderate", 2),
        _decided("M-3", "moderate", 3, "rejected"),
        _decided("M-4", "moderate", 4, "rejected"),
        _decided("M-9", "moderate", 9),
        _decided("M-12", "moderate", 12, "rejected"),
    ]


def _store(records: list[ApprovalRecord] | None = None) -> InMemoryStore:
    store = InMemoryStore()
    for record in records if records is not None else _records():
        store.save(record)
    return store


def _row(report: dict, tier: str, value: int) -> dict:
    block = next(t for t in report["tiers"] if t["tier"] == tier)
    return next(r for r in block["sweeps"] if r["value"] == value)


# ---------------------------------------------------------------------------
# The row becomes the proposal, evidence verbatim
# ---------------------------------------------------------------------------


def test_row_draft_carries_the_rows_evidence_verbatim():
    store = _store()
    records = store.all_records()
    report = sla_budget_calibration(records, effective_sla_budgets(store))
    row = _row(report, "moderate", 3)
    assert row["proposal_ready"] is True

    proposal = draft_sla_proposal_from_row(
        store, records, "moderate", 3, "U. Writer"
    )
    assert proposal["kind"] == "sla_budget"
    assert proposal["dimension"] == "moderate"
    assert proposal["current_value"] == 5
    assert proposal["proposed_value"] == 3
    assert proposal["direction"] == "lower"
    assert proposal["status"] == "draft"
    assert proposal["drafted_from"] == "calibration_row"
    assert proposal["drafted_by"] == "U. Writer"
    assert proposal["expires_at"] > proposal["drafted_at"]
    # The evidence is the row's: same moved set, same splits, and
    # the row's own breach counts kept beside them.
    assert proposal["evidence"]["moved_ids"] == [
        e["submission_id"] for e in row["moved_vs_current"]
    ]
    assert proposal["evidence"]["moved_ids"] == ["M-4"]
    assert proposal["evidence"]["moved_decisions"] == {
        "approved": 0, "rejected": 1, "undecided": 0,
    }
    assert proposal["evidence"]["waits"] == {"M-4": 4}
    assert proposal["evidence"]["calibration_row"] == {
        "budget_in_force": 5,
        "breached": row["breached"],
        "breached_decisions": row["breached_decisions"],
        "within_decisions": row["within_decisions"],
    }
    assert "calibration report's moderate row at 3 day(s)" in proposal["rationale"]


def test_row_draft_is_the_operators_row_not_the_scans_best():
    # The scan drafts its best row for moderate: 2 (two moved).
    scan_store = _store()
    scanned = draft_sla_proposals(scan_store, scan_store.all_records())
    assert [p["proposed_value"] for p in scanned["drafted"]] == [2]
    assert scanned["drafted"][0]["drafted_from"] == "scan"

    # The operator's row is a different, smaller move — and a
    # raise row drafts too, on its own one-sided evidence.
    row_store = _store()
    proposal = draft_sla_proposal_from_row(
        row_store, row_store.all_records(), "moderate", 3, "U. Writer"
    )
    assert proposal["proposed_value"] == 3

    raise_store = _store()
    raised = draft_sla_proposal_from_row(
        raise_store, raise_store.all_records(), "moderate", 10, "U. Writer"
    )
    assert raised["direction"] == "raise"
    assert raised["evidence"]["moved_ids"] == ["M-9"]


def test_a_row_drafted_proposal_decides_like_any_other():
    store = _store()
    proposal = draft_sla_proposal_from_row(
        store, store.all_records(), "moderate", 3, "U. Writer"
    )
    decided = decide_proposal(
        store, proposal["proposal_id"], "Senior U", accept=True
    )
    assert decided["status"] == "accepted"
    assert effective_sla_budgets(store)["moderate"] == 3


# ---------------------------------------------------------------------------
# The discipline: refusals draft nothing
# ---------------------------------------------------------------------------


def test_row_draft_refuses_the_budget_in_force_and_unswept_values():
    store = _store()
    records = store.all_records()
    with pytest.raises(ValueError, match="already the moderate"):
        draft_sla_proposal_from_row(store, records, "moderate", 5, "U. Writer")
    with pytest.raises(ValueError, match="not one of the calibration"):
        draft_sla_proposal_from_row(store, records, "moderate", 8, "U. Writer")
    with pytest.raises(ValueError, match="Unknown risk tier"):
        draft_sla_proposal_from_row(store, records, "critical", 3, "U. Writer")


def test_row_draft_refuses_rows_without_one_sided_evidence():
    store = _store()
    records = store.all_records()
    # Row 7 (raise) moves no decided case at all.
    with pytest.raises(ValueError, match="moves no decided case"):
        draft_sla_proposal_from_row(store, records, "moderate", 7, "U. Writer")

    # A moved set pointing both ways is not evidence.
    mixed = _store(
        [
            _decided("X-4R", "moderate", 4, "rejected"),
            _decided("X-4A", "moderate", 4, "approved"),
        ]
    )
    with pytest.raises(ValueError, match="not one-sided"):
        draft_sla_proposal_from_row(
            mixed, mixed.all_records(), "moderate", 3, "U. Writer"
        )


def test_one_open_question_per_tier_binds_row_drafts_and_the_scan():
    store = _store()
    records = store.all_records()
    draft_sla_proposal_from_row(store, records, "moderate", 3, "U. Writer")
    with pytest.raises(ValueError, match="one open question per tier"):
        draft_sla_proposal_from_row(store, records, "moderate", 10, "U. Writer")
    # And the scan leaves the tier alone while the row's question
    # is open.
    assert draft_sla_proposals(store, records)["drafted"] == []


def test_a_rejected_direction_is_never_re_asked_from_a_row():
    store = _store()
    records = store.all_records()
    proposal = draft_sla_proposal_from_row(
        store, records, "moderate", 3, "U. Writer"
    )
    decide_proposal(store, proposal["proposal_id"], "Senior U", accept=False)
    with pytest.raises(ValueError, match="already rejected"):
        draft_sla_proposal_from_row(store, records, "moderate", 2, "U. Writer")
    # The other direction was never refused and still drafts.
    raised = draft_sla_proposal_from_row(
        store, records, "moderate", 10, "U. Writer"
    )
    assert raised["direction"] == "raise"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    store = _store()
    api_module.service = SubmissionService(store=store, checkpointer=False)
    return TestClient(api_module.app)


def test_api_drafts_from_a_row_and_lists_the_proposal(client):
    res = client.post(
        "/queue/sla/calibration/draft",
        json={"actor": "U. Writer", "tier": "moderate", "value": 3},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["drafted_from"] == "calibration_row"
    assert body["proposed_value"] == 3

    listed = client.get("/calibration/proposals").json()["proposals"]
    assert [p["proposal_id"] for p in listed] == [body["proposal_id"]]

    # A second row for the same tier is refused while the first
    # question is open — the refusal is a 422 with the reason.
    again = client.post(
        "/queue/sla/calibration/draft",
        json={"actor": "U. Writer", "tier": "moderate", "value": 10},
    )
    assert again.status_code == 422
    assert "one open question" in again.json()["detail"]


def test_api_row_draft_requires_an_actor_and_a_ready_row(client):
    anonymous = client.post(
        "/queue/sla/calibration/draft",
        json={"tier": "moderate", "value": 3},
    )
    assert anonymous.status_code == 422
    not_ready = client.post(
        "/queue/sla/calibration/draft",
        json={"actor": "U. Writer", "tier": "moderate", "value": 7},
    )
    assert not_ready.status_code == 422
    assert "moves no decided case" in not_ready.json()["detail"]
