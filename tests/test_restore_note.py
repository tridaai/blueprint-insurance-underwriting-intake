"""Round 12: the restore decision note.

A settings record answers what moved, who moved it, and when —
but a restore's whole point is judgement ("the tuned week was a
mistake"), and that judgement had nowhere to live: six months
later the restore entry proved a revert happened without ever
saying why. A restore now carries an optional decision note,
stored on the restore entry the way a proposal decision keeps
its note, and the preview — the natural place to draft it —
echoes the note back against the exact moves it would be
stored with, while still writing nothing.
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
    restore_sla_budget_settings,
    restore_threshold_settings,
    revert_threshold_settings,
)
from underwriting_agent.store import InMemoryStore

AT = "2026-10-09T12:00:00+00:00"
NOTE = "The tuned week mis-referred clean renewals — back to the last known-good state."


def _threshold_store() -> InMemoryStore:
    """Two accepted changes (limit 50→40, floor 50000→25000);
    restoring to entry 0 moves the floor back."""
    store = InMemoryStore()
    store.save_document(
        SETTINGS_KEY,
        {
            "thresholds": {
                "limit_jump_pct": 40.0,
                "new_claim_min_incurred": 25000.0,
                "development_min_increase": 50000.0,
            },
            "updated_by": "U. Writer",
            "updated_at": AT,
            "proposal_id": "PROP-B",
            "history": [
                {"proposal_id": "PROP-A", "dimension": "limit_jump_pct",
                 "from": 50.0, "to": 40.0, "actor": "U. Writer", "at": AT},
                {"proposal_id": "PROP-B", "dimension": "new_claim_min_incurred",
                 "from": 50000.0, "to": 25000.0, "actor": "U. Writer", "at": AT},
            ],
        },
    )
    return store


def test_a_restore_keeps_its_decision_note_on_the_entry():
    store = _threshold_store()
    updated = restore_threshold_settings(store, 0, "Ops Lead", note=NOTE)
    entry = updated["history"][-1]
    assert entry["restore"] is True
    assert entry["decision_note"] == NOTE
    assert entry["actor"] == "Ops Lead"
    # The earlier entries are untouched — the note belongs to the
    # restore, not to the history it passes over.
    assert "decision_note" not in updated["history"][0]


def test_a_restore_without_a_note_records_an_empty_one():
    store = _threshold_store()
    updated = restore_threshold_settings(store, 0, "Ops Lead")
    assert updated["history"][-1]["decision_note"] == ""


def test_a_noted_restore_still_reverts_by_the_same_rule():
    store = _threshold_store()
    restore_threshold_settings(store, 0, "Ops Lead", note=NOTE)
    reverted = revert_threshold_settings(store, "Ops Lead")
    # The revert undoes the restore's changes; its own entry is a
    # separate event and does not inherit the note.
    assert reverted["history"][-1]["revert"] is True
    assert reverted["thresholds"]["new_claim_min_incurred"] == 25000.0


def test_sla_restore_note_lands_on_the_entry():
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
    preview = preview_restore_sla_budget_settings(store, 0, note=NOTE)
    assert preview["decision_note"] == NOTE
    updated = restore_sla_budget_settings(store, 0, "Ops Lead", note=NOTE)
    entry = updated["history"][-1]
    assert entry["decision_note"] == NOTE
    assert entry["changes"] == [{"dimension": "low", "from": 10, "to": 7}]
    assert entry["changes"] == preview["changes"]


def test_the_preview_echoes_the_drafted_note_and_stores_nothing():
    store = _threshold_store()
    before = copy.deepcopy(store.get_document(SETTINGS_KEY))
    preview = preview_restore_threshold_settings(store, 0, note=NOTE)
    assert preview["decision_note"] == NOTE
    assert preview["would_change"] is True
    assert store.get_document(SETTINGS_KEY) == before
    # No note drafted: the echo is empty, not absent.
    assert preview_restore_threshold_settings(store, 0)["decision_note"] == ""


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app), api_module.service


def test_restore_endpoints_carry_the_note(client):
    test_client, service = client
    store = service._get_store()
    store.save_document(SETTINGS_KEY, _threshold_store().get_document(SETTINGS_KEY))

    preview = test_client.post(
        "/calibration/settings/restore/preview",
        json={"actor": "Ops Lead", "history_index": 0, "note": NOTE},
    )
    assert preview.status_code == 200
    assert preview.json()["decision_note"] == NOTE
    # The preview stored nothing — history is still two entries.
    assert len(store.get_document(SETTINGS_KEY)["history"]) == 2

    restored = test_client.post(
        "/calibration/settings/restore",
        json={"actor": "Ops Lead", "history_index": 0, "note": NOTE},
    )
    assert restored.status_code == 200
    entry = restored.json()["settings"]["history"][-1]
    assert entry["decision_note"] == NOTE
    assert entry["restore"] is True
