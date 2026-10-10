"""Round 10: point-in-history settings restore.

Round 9's revert walks one step back. A bad *sequence* of
accepts — a mis-tuned afternoon — needs the whole record
restored to a chosen history entry's state: the values as of
that entry, restored as ONE new audited change naming the entry
it restores to, however many dimensions move. History is never
rewritten, and the restore itself can be reverted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    SETTINGS_KEY,
    SLA_SETTINGS_KEY,
    effective_sla_budgets,
    effective_thresholds,
    load_settings,
    load_sla_settings,
    restore_sla_budget_settings,
    restore_threshold_settings,
    revert_threshold_settings,
)
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

AT = "2026-10-09T12:00:00+00:00"


def _threshold_store() -> InMemoryStore:
    """A thresholds record after three accepted changes:
    limit 50→40 (A), new-claim floor 50000→25000 (B), limit
    40→30 (C). In force: limit 30, floor 25000, development 50000."""
    store = InMemoryStore()
    store.save_document(
        SETTINGS_KEY,
        {
            "thresholds": {
                "limit_jump_pct": 30.0,
                "new_claim_min_incurred": 25000.0,
                "development_min_increase": 50000.0,
            },
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-C",
            "history": [
                {"proposal_id": "PROP-A", "dimension": "limit_jump_pct",
                 "from": 50.0, "to": 40.0, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-B", "dimension": "new_claim_min_incurred",
                 "from": 50000.0, "to": 25000.0, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-C", "dimension": "limit_jump_pct",
                 "from": 40.0, "to": 30.0, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    return store


def test_restore_returns_the_record_to_a_chosen_entry_as_one_change():
    store = _threshold_store()
    updated = restore_threshold_settings(store, 0, "Ops Lead")
    # The state as of entry 0: only PROP-A's change had landed.
    assert updated["thresholds"] == {
        "limit_jump_pct": 40.0,
        "new_claim_min_incurred": 50000.0,
        "development_min_increase": 50000.0,
    }
    assert effective_thresholds(store)["limit_jump_pct"] == 40.0
    # One appended entry names the entry restored to and lists
    # every dimension it moved — the three accepts stay untouched.
    assert len(updated["history"]) == 4
    entry = updated["history"][-1]
    assert entry["restore"] is True
    assert entry["restores_index"] == 0
    assert entry["restores_proposal_id"] == "PROP-A"
    assert entry["actor"] == "Ops Lead"
    assert {c["dimension"]: (c["from"], c["to"]) for c in entry["changes"]} == {
        "limit_jump_pct": (30.0, 40.0),
        "new_claim_min_incurred": (25000.0, 50000.0),
    }


def test_restore_to_the_current_state_is_refused():
    store = _threshold_store()
    with pytest.raises(ValueError, match="already match"):
        restore_threshold_settings(store, 2, "Ops Lead")
    assert len(load_settings(store)["history"]) == 3  # nothing appended


def test_restore_validates_its_target():
    store = _threshold_store()
    with pytest.raises(ValueError, match="out of range"):
        restore_threshold_settings(store, 3, "Ops Lead")
    with pytest.raises(ValueError, match="out of range"):
        restore_threshold_settings(store, -1, "Ops Lead")
    with pytest.raises(ValueError, match="Nothing to restore"):
        restore_threshold_settings(InMemoryStore(), 0, "Ops Lead")


def test_a_restore_can_be_reverted_and_history_replays_across_shapes():
    store = _threshold_store()
    restore_threshold_settings(store, 0, "Ops Lead")
    revert_threshold_settings(store, "Ops Lead")
    # The revert undid the restore's every dimension at once.
    assert effective_thresholds(store) == {
        "limit_jump_pct": 30.0,
        "new_claim_min_incurred": 25000.0,
        "development_min_increase": 50000.0,
    }
    revert_entry = load_settings(store)["history"][-1]
    assert revert_entry["revert"] is True
    assert revert_entry["reverts_restore"] is True
    # And a later restore still replays correctly through the
    # mixed flat/changes history: entry 1's state is limit 40
    # (A landed, C not yet), floor 25000 (B landed).
    restore_threshold_settings(store, 1, "Ops Lead")
    assert effective_thresholds(store)["limit_jump_pct"] == 40.0
    assert effective_thresholds(store)["new_claim_min_incurred"] == 25000.0


def test_sla_budgets_restore_the_same_way():
    store = InMemoryStore()
    store.save_document(
        SLA_SETTINGS_KEY,
        {
            "budgets": {"elevated": 2, "high": 4, "moderate": 6, "low": 7},
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-S2",
            "history": [
                {"proposal_id": "PROP-S1", "dimension": "high",
                 "from": 3, "to": 4, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-S2", "dimension": "moderate",
                 "from": 5, "to": 6, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    updated = restore_sla_budget_settings(store, 0, "Ops Lead")
    assert updated["budgets"] == {"elevated": 2, "high": 4, "moderate": 5, "low": 7}
    assert effective_sla_budgets(store)["moderate"] == 5
    entry = updated["history"][-1]
    assert entry["restore"] is True and entry["restores_index"] == 0
    assert entry["changes"] == [
        {"dimension": "moderate", "from": 6, "to": 5}
    ]
    assert load_sla_settings(store)["updated_by"] == "Ops Lead"


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


def test_service_restore_after_a_revert_brings_the_accepted_value_back():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(_submission("CAL-A", [CLAIM_1]))
    service.analyze(_submission("CAL-B", [CLAIM_1, CLAIM_2]))
    service.reject("CAL-B", reviewer="U. Writer", reason="loss trend")
    proposal = service.draft_calibration_proposals()["drafted"][0]
    service.decide_calibration_proposal(proposal["proposal_id"], "U. Writer", accept=True)
    store = service._get_store()
    assert effective_thresholds(store)["new_claim_min_incurred"] == 25000.0
    # Restoring to the only entry is a no-op — refused.
    with pytest.raises(ValueError, match="already match"):
        service.restore_diff_threshold_settings("Ops Lead", 0)
    service.revert_diff_threshold_settings("Ops Lead")
    assert effective_thresholds(store)["new_claim_min_incurred"] == 50000.0
    view = service.restore_diff_threshold_settings("Ops Lead", 0)
    assert view["thresholds"]["new_claim_min_incurred"] == 25000.0


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_restore_endpoints_round_trip(client):
    # Seed a record through the API's own service, then restore.
    import underwriting_agent.api as api_module

    store = api_module.service._get_store()
    store.save_document(
        SETTINGS_KEY,
        {
            "thresholds": {
                "limit_jump_pct": 30.0,
                "new_claim_min_incurred": 50000.0,
                "development_min_increase": 50000.0,
            },
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-A",
            "history": [
                {"proposal_id": "PROP-A", "dimension": "limit_jump_pct",
                 "from": 50.0, "to": 30.0, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    res = client.post(
        "/calibration/settings/restore",
        json={"actor": "Ops Lead", "history_index": 0},
    )
    assert res.status_code == 422  # already at entry 0's state
    store.save_document(
        SETTINGS_KEY,
        {
            "thresholds": {
                "limit_jump_pct": 20.0,
                "new_claim_min_incurred": 50000.0,
                "development_min_increase": 50000.0,
            },
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-B",
            "history": [
                {"proposal_id": "PROP-A", "dimension": "limit_jump_pct",
                 "from": 50.0, "to": 30.0, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-B", "dimension": "limit_jump_pct",
                 "from": 30.0, "to": 20.0, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    res = client.post(
        "/calibration/settings/restore",
        json={"actor": "Ops Lead", "history_index": 0},
    )
    assert res.status_code == 200
    assert res.json()["thresholds"]["limit_jump_pct"] == 30.0
    res = client.post(
        "/queue/sla/settings/restore",
        json={"actor": "Ops Lead", "history_index": 0},
    )
    assert res.status_code == 422  # no SLA settings record exists
