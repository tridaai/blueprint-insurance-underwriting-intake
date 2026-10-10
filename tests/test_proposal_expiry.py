"""Round 9: proposal expiry + evidence refresh.

A drafted proposal's evidence is a snapshot of the store at
drafting time; decisions land afterwards and can quietly
invalidate it. Drafts now expire after the configured window —
an expired draft cannot be decided — and a human refreshes one
by re-running its evidence scan: renewed in place when the
evidence still carries a one-sided move, withdrawn when it no
longer does. Every refresh is on the proposal's audit list.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from underwriting_agent.proposals import (
    decide_proposal,
    draft_proposals,
    draft_sla_proposals,
    is_expired,
    load_proposals,
    proposal_expires_at,
    refresh_proposal,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _sla_record(
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


def _raise_store() -> InMemoryStore:
    """One moderate case that breached the 5-day budget (waited
    8) and was approved anyway — a raise proposal at T0."""
    return _store_with(
        [_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00")]
    )


# ---------------------------------------------------------------------------
# Expiry stamping and the decide guard
# ---------------------------------------------------------------------------


def test_drafts_are_stamped_with_an_expiry_window():
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    drafted = datetime.fromisoformat(proposal["drafted_at"])
    expires = datetime.fromisoformat(proposal["expires_at"])
    assert expires - drafted == timedelta(days=14)
    assert proposal_expires_at(proposal) == proposal["expires_at"]
    assert not is_expired(proposal, now=T0 + timedelta(days=13))
    assert is_expired(proposal, now=T0 + timedelta(days=15))


def test_expiry_window_is_configurable(monkeypatch):
    monkeypatch.setenv("PROPOSAL_EXPIRY_DAYS", "3")
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    assert proposal["expires_at"] == (T0 + timedelta(days=3)).isoformat(
        timespec="seconds"
    )


def test_an_expired_draft_cannot_be_decided():
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    with pytest.raises(ValueError, match="expired"):
        decide_proposal(
            store,
            proposal["proposal_id"],
            "U. Writer",
            accept=True,
            now=T0 + timedelta(days=20),
        )
    # Still a draft — the refusal changed nothing.
    assert load_proposals(store)[0]["status"] == "draft"
    # Inside the window the same decision goes through.
    decided = decide_proposal(
        store,
        proposal["proposal_id"],
        "U. Writer",
        accept=True,
        now=T0 + timedelta(days=2),
    )
    assert decided["status"] == "accepted"


def test_decided_proposals_do_not_expire():
    proposal = {"status": "accepted", "drafted_at": "2020-01-01T00:00:00+00:00"}
    assert not is_expired(proposal, now=T0)


# ---------------------------------------------------------------------------
# Refresh: renewed when the evidence holds, withdrawn when it goes
# ---------------------------------------------------------------------------


def test_refresh_renews_a_live_snapshot_and_reopens_the_window():
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    original_expires_at = proposal["expires_at"]
    later = T0 + timedelta(days=20)
    refreshed = refresh_proposal(
        store, store.all_records(), proposal["proposal_id"], "U. Writer", now=later
    )
    assert refreshed["status"] == "draft"
    assert refreshed["proposal_id"] == proposal["proposal_id"]
    assert refreshed["proposed_value"] == 10  # the same move, re-evidenced
    assert refreshed["drafted_at"] == later.isoformat(timespec="seconds")
    assert refreshed["expires_at"] == (later + timedelta(days=14)).isoformat(
        timespec="seconds"
    )
    audit = refreshed["refreshes"]
    assert len(audit) == 1
    assert audit[0]["outcome"] == "renewed"
    assert audit[0]["by"] == "U. Writer"
    assert audit[0]["previous"]["proposed_value"] == 10
    assert audit[0]["previous"]["expires_at"] == original_expires_at
    # And it can be decided inside its new window.
    decided = decide_proposal(
        store,
        proposal["proposal_id"],
        "U. Writer",
        accept=True,
        now=later + timedelta(days=1),
    )
    assert decided["status"] == "accepted"


def test_refresh_withdraws_when_the_evidence_no_longer_holds():
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    # A second moderate case breaches the budget (waited 9) and is
    # REJECTED: raising the budget would now un-flag a rejection —
    # the move's evidence is two-sided, so the refresh withdraws.
    store.save(
        _sla_record(
            "M-BAD", "2026-09-24", "2026-10-03T10:00:00+00:00", status="rejected"
        )
    )
    refreshed = refresh_proposal(
        store,
        store.all_records(),
        proposal["proposal_id"],
        "U. Writer",
        now=T0 + timedelta(days=20),
    )
    assert refreshed["status"] == "withdrawn"
    assert refreshed["withdrawn_by"] == "U. Writer"
    assert "no longer holds" in refreshed["withdrawal_reason"]
    assert refreshed["refreshes"][0]["outcome"] == "withdrawn"
    # A withdrawn proposal is history: no deciding, no re-refresh.
    with pytest.raises(ValueError, match="decided once|only a live draft"):
        decide_proposal(
            store, proposal["proposal_id"], "U. Writer", accept=True
        )
    with pytest.raises(ValueError, match="only a live draft"):
        refresh_proposal(
            store, store.all_records(), proposal["proposal_id"], "U. Writer"
        )


def test_refresh_refuses_live_and_unknown_proposals():
    store = _raise_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    with pytest.raises(ValueError, match="still live"):
        refresh_proposal(
            store,
            store.all_records(),
            proposal["proposal_id"],
            "U. Writer",
            now=T0 + timedelta(days=1),
        )
    with pytest.raises(KeyError):
        refresh_proposal(store, store.all_records(), "PROP-9999", "U. Writer")


def test_refresh_renews_threshold_proposals_too():
    # The round-8 chain: CAL-B's new $30k claim sits under the
    # $50k floor and the underwriter rejected it — a lower-floor
    # proposal whose evidence lives in the renewal replay.
    claim_1 = {"date": "2024-02-01", "amount_incurred": 5000,
               "description": "Water damage", "status": "closed"}
    claim_2 = {"date": "2025-06-01", "amount_incurred": 30000,
               "description": "Fire damage", "status": "open"}

    def submission(sid, losses):
        return {
            "submission_id": sid,
            "insured_name": "Calib Co",
            "broker_name": "Broker Q",
            "line_of_business": "general",
            "class_of_business": "retail",
            "submitted_at": "2026-10-01",
            "documents": [
                {"doc_type": "acord_form", "document_id": "A1",
                 "raw_text": "Application."}
            ],
            "losses": losses,
        }

    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(submission("CAL-A", [claim_1]))
    service.analyze(submission("CAL-B", [claim_1, claim_2]))
    service.reject("CAL-B", reviewer="U. Writer", reason="loss trend")
    store = service._get_store()
    proposal = draft_proposals(store, store.all_records(), now=T0)["drafted"][0]
    assert proposal["kind"] == "diff_trigger_threshold"

    refreshed = refresh_proposal(
        store,
        store.all_records(),
        proposal["proposal_id"],
        "U. Writer",
        now=T0 + timedelta(days=30),
    )
    assert refreshed["status"] == "draft"
    assert refreshed["proposed_value"] == proposal["proposed_value"]
    assert refreshed["refreshes"][0]["outcome"] == "renewed"


# ---------------------------------------------------------------------------
# Service + API surface
# ---------------------------------------------------------------------------


def test_service_annotations_and_refresh():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    store = service._get_store()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    proposal = service.draft_sla_budget_proposals()["drafted"][0]
    listing = service.calibration_proposals()
    assert listing[0]["proposal_id"] == proposal["proposal_id"]
    assert listing[0]["expires_at"] == proposal["expires_at"]
    assert listing[0]["expired"] is False

    # Force expiry in the stored document, then refresh via the
    # service: renewed, and the listing no longer flags it.
    from underwriting_agent.proposals import PROPOSALS_KEY, load_proposals

    stored = load_proposals(store)
    stored[0]["expires_at"] = "2020-01-01T00:00:00+00:00"
    store.save_document(PROPOSALS_KEY, {"proposals": stored})
    assert service.calibration_proposals()[0]["expired"] is True
    refreshed = service.refresh_calibration_proposal(
        proposal["proposal_id"], "U. Writer"
    )
    assert refreshed["expired"] is False
    assert refreshed["refreshes"][0]["outcome"] == "renewed"


@pytest.fixture()
def api_env(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app), api_module.service


def test_refresh_endpoint_and_expired_decide(api_env):
    client, service = api_env
    store = service._get_store()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    proposal_id = client.post("/queue/sla-proposals/draft").json()["drafted"][0][
        "proposal_id"
    ]

    from underwriting_agent.proposals import PROPOSALS_KEY, load_proposals

    stored = load_proposals(store)
    stored[0]["expires_at"] = "2020-01-01T00:00:00+00:00"
    store.save_document(PROPOSALS_KEY, {"proposals": stored})

    blocked = client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert blocked.status_code == 422
    assert "expired" in blocked.json()["detail"]

    refreshed = client.post(
        f"/calibration/proposals/{proposal_id}/refresh",
        json={"actor": "U. Writer"},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["status"] == "draft"
    assert refreshed.json()["expired"] is False

    decided = client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert decided.status_code == 200

    missing = client.post(
        "/calibration/proposals/PROP-9999/refresh", json={"actor": "U. Writer"}
    )
    assert missing.status_code == 404
    anonymous = client.post(
        f"/calibration/proposals/{proposal_id}/refresh", json={}
    )
    assert anonymous.status_code == 422  # a refresh needs its actor too
