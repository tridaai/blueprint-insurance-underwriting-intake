"""Round 11: restore previews for the settings records.

Round 10's restore applies in one call: one audited entry, but
an operator committing to it could not see which dimensions it
would move without replaying the history themselves. The preview
is the restore's dry run — the shared plan, not a second
implementation: the values in force now, the values as of the
chosen entry, and the per-dimension moves, computed without
writing or appending anything. A restore that would move
nothing previews as exactly that, instead of failing the way
the real restore does.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    SETTINGS_KEY,
    SLA_SETTINGS_KEY,
    preview_restore_sla_budget_settings,
    preview_restore_threshold_settings,
    restore_threshold_settings,
)
from underwriting_agent.store import InMemoryStore

AT = "2026-10-09T12:00:00+00:00"


def _threshold_store() -> InMemoryStore:
    """The round-10 fixture: three accepted changes (limit
    50→40, floor 50000→25000, limit 40→30). In force: limit 30,
    floor 25000, development floor 50000 (the environment base)."""
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


def test_preview_names_every_dimension_a_restore_would_move():
    store = _threshold_store()
    preview = preview_restore_threshold_settings(store, 0)
    assert preview["label"] == "thresholds"
    assert preview["history_index"] == 0
    assert preview["restores_proposal_id"] == "PROP-A"
    assert preview["current_values"] == {
        "limit_jump_pct": 30.0,
        "new_claim_min_incurred": 25000.0,
        "development_min_increase": 50000.0,
    }
    assert preview["target_values"] == {
        "limit_jump_pct": 40.0,
        "new_claim_min_incurred": 50000.0,
        "development_min_increase": 50000.0,
    }
    assert preview["changes"] == [
        {"dimension": "limit_jump_pct", "from": 30.0, "to": 40.0},
        {"dimension": "new_claim_min_incurred", "from": 25000.0, "to": 50000.0},
    ]
    assert preview["would_change"] is True


def test_preview_writes_nothing_and_the_restore_lands_as_previewed():
    store = _threshold_store()
    before = copy.deepcopy(store.get_document(SETTINGS_KEY))
    preview = preview_restore_threshold_settings(store, 0)
    assert store.get_document(SETTINGS_KEY) == before

    updated = restore_threshold_settings(store, 0, "Ops Lead")
    # What landed is what the preview promised — same plan.
    assert updated["thresholds"] == preview["target_values"]
    restore_entry = updated["history"][-1]
    assert restore_entry["changes"] == preview["changes"]


def test_preview_of_a_no_op_restore_reports_it_instead_of_failing():
    store = _threshold_store()
    preview = preview_restore_threshold_settings(store, 2)
    assert preview["changes"] == []
    assert preview["would_change"] is False
    # …while committing to that same restore stays an error.
    with pytest.raises(ValueError, match="would change nothing"):
        restore_threshold_settings(store, 2, "Ops Lead")


def test_preview_keeps_the_restore_errors_for_bad_targets():
    store = _threshold_store()
    with pytest.raises(ValueError, match="out of range"):
        preview_restore_threshold_settings(store, 9)
    empty = InMemoryStore()
    with pytest.raises(ValueError, match="Nothing to restore"):
        preview_restore_threshold_settings(empty, 0)


def test_sla_preview_uses_budget_values():
    # Two accepted budget changes: moderate 5→9, then low 7→10.
    # Restoring to entry 0 would move only the low budget back.
    store = InMemoryStore()
    store.save_document(
        SLA_SETTINGS_KEY,
        {
            "budgets": {"elevated": 2, "high": 3, "moderate": 9, "low": 10},
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-T",
            "history": [
                {"proposal_id": "PROP-S", "dimension": "moderate",
                 "from": 5, "to": 9, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-T", "dimension": "low",
                 "from": 7, "to": 10, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    preview = preview_restore_sla_budget_settings(store, 0)
    assert preview["label"] == "budgets"
    assert preview["restores_proposal_id"] == "PROP-S"
    assert preview["changes"] == [
        {"dimension": "low", "from": 10, "to": 7},
    ]
    assert preview["would_change"] is True
    assert preview["target_values"] == {
        "elevated": 2, "high": 3, "moderate": 9, "low": 7,
    }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_preview_endpoint_validates_like_the_restore(client):
    # The fresh service has no settings history: same 422 the
    # restore itself would give, from the preview alone.
    res = client.post(
        "/calibration/settings/restore/preview",
        json={"actor": "Ops Lead", "history_index": 0},
    )
    assert res.status_code == 422
    assert "Nothing to restore" in res.json()["detail"]
    res = client.post(
        "/queue/sla/settings/restore/preview",
        json={"actor": "Ops Lead", "history_index": 0},
    )
    assert res.status_code == 422
