"""Round 9: settings revert.

Accepting a calibration proposal was audited but one-way: a
threshold or budget change that misbehaved in production had no
recorded way back short of editing the database. The settings
records now revert: a human restores the prior value as a NEW
history entry naming the proposal it undoes — the record stays
append-only, so the way back is itself auditable.
"""

from __future__ import annotations

import pytest

from underwriting_agent.proposals import (
    effective_sla_budgets,
    effective_thresholds,
    load_settings,
    load_sla_settings,
    revert_sla_budget_settings,
    revert_threshold_settings,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

CLAIM_1 = {"date": "2024-02-01", "amount_incurred": 5000, "description": "Water damage", "status": "closed"}
CLAIM_2 = {"date": "2025-06-01", "amount_incurred": 30000, "description": "Fire damage", "status": "open"}


def _submission(sid: str, losses: list[dict]) -> dict:
    return {
        "submission_id": sid,
        "insured_name": "Calib Co",
        "broker_name": "Broker Q",
        "line_of_business": "general",
        "class_of_business": "retail",
        "submitted_at": "2026-10-01",
        "documents": [
            {"doc_type": "acord_form", "document_id": "A1", "raw_text": "Application."}
        ],
        "losses": losses,
    }


def _service_with_accepted_threshold() -> tuple[SubmissionService, str]:
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(_submission("CAL-A", [CLAIM_1]))
    service.analyze(_submission("CAL-B", [CLAIM_1, CLAIM_2]))
    service.reject("CAL-B", reviewer="U. Writer", reason="loss trend")
    proposal = service.draft_calibration_proposals()["drafted"][0]
    service.decide_calibration_proposal(
        proposal["proposal_id"], "U. Writer", accept=True
    )
    return service, proposal["proposal_id"]


def _sla_record(sid: str, submitted_at: str, decided_at: str) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="approved",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            decided_at=decided_at,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


# ---------------------------------------------------------------------------
# Threshold settings revert
# ---------------------------------------------------------------------------


def test_revert_restores_the_prior_threshold_as_a_new_entry():
    service, proposal_id = _service_with_accepted_threshold()
    store = service._get_store()
    assert effective_thresholds(store)["new_claim_min_incurred"] == 25000.0

    view = service.revert_diff_threshold_settings("Ops Lead")
    assert view["thresholds"]["new_claim_min_incurred"] == 50000.0
    assert effective_thresholds(store)["new_claim_min_incurred"] == 50000.0

    settings = load_settings(store)
    assert len(settings["history"]) == 2  # appended, never rewritten
    accept_entry, revert_entry = settings["history"]
    assert accept_entry["to"] == 25000.0 and "revert" not in accept_entry
    assert revert_entry == {
        "proposal_id": proposal_id,  # names the proposal it undoes
        "dimension": "new_claim_min_incurred",
        "from": 25000.0,
        "to": 50000.0,
        "actor": "Ops Lead",
        "at": revert_entry["at"],
        "revert": True,
    }
    assert settings["updated_by"] == "Ops Lead"


def test_reverting_a_revert_reapplies_the_change():
    service, _ = _service_with_accepted_threshold()
    store = service._get_store()
    revert_threshold_settings(store, "Ops Lead")
    assert effective_thresholds(store)["new_claim_min_incurred"] == 50000.0
    revert_threshold_settings(store, "Ops Lead")
    assert effective_thresholds(store)["new_claim_min_incurred"] == 25000.0
    assert len(load_settings(store)["history"]) == 3


def test_revert_without_a_settings_record_is_refused():
    store = InMemoryStore()
    with pytest.raises(ValueError, match="Nothing to revert"):
        revert_threshold_settings(store, "Ops Lead")
    with pytest.raises(ValueError, match="Nothing to revert"):
        revert_sla_budget_settings(store, "Ops Lead")


# ---------------------------------------------------------------------------
# SLA-budget settings revert
# ---------------------------------------------------------------------------


def test_sla_revert_restores_the_prior_budget():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    store = service._get_store()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    proposal = service.draft_sla_budget_proposals()["drafted"][0]
    service.decide_calibration_proposal(
        proposal["proposal_id"], "U. Writer", accept=True
    )
    assert effective_sla_budgets(store)["moderate"] == 10

    view = service.revert_sla_budget_settings("Ops Lead")
    assert view["budgets"]["moderate"] == 5
    assert effective_sla_budgets(store)["moderate"] == 5
    history = load_sla_settings(store)["history"]
    assert len(history) == 2
    assert history[1]["revert"] is True
    assert history[1]["proposal_id"] == proposal["proposal_id"]
    assert history[1]["from"] == 10 and history[1]["to"] == 5


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


def test_revert_endpoints(api_env):
    client, service = api_env
    # Nothing accepted yet: both reverts refuse, with a reason.
    for path in ("/calibration/settings/revert", "/queue/sla/settings/revert"):
        refused = client.post(path, json={"actor": "Ops Lead"})
        assert refused.status_code == 422
        assert "Nothing to revert" in refused.json()["detail"]

    store = service._get_store()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    proposal_id = client.post("/queue/sla-proposals/draft").json()["drafted"][0][
        "proposal_id"
    ]
    client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert client.get("/queue/sla/settings").json()["budgets"]["moderate"] == 10

    reverted = client.post(
        "/queue/sla/settings/revert", json={"actor": "Ops Lead"}
    )
    assert reverted.status_code == 200
    assert reverted.json()["budgets"]["moderate"] == 5
    history = reverted.json()["settings"]["history"]
    assert history[-1]["revert"] is True

    anonymous = client.post("/queue/sla/settings/revert", json={})
    assert anonymous.status_code == 422  # a revert needs its actor
