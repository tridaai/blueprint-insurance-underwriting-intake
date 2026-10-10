"""Round 11: the refresh preview.

Round 10's sweep refreshes every expired draft in one call —
but it writes as it goes, so an operator returning after a quiet
fortnight committed blind: no way to see how many drafts the
evidence still carried and how many it had dropped. The preview
runs the same per-draft scans and reports would-renew /
would-withdraw, side by side, without a single write — and the
sweep that follows on the same store produces exactly the
previewed outcomes, because preview and sweep share the scan.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    PROPOSALS_KEY,
    draft_sla_proposals,
    load_proposals,
    preview_refresh_expired_proposals,
    preview_refresh_proposal,
    refresh_expired_proposals,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord, InMemoryStore

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
LATER = T0 + timedelta(days=20)  # past the 14-day evidence window


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


def _store_with_expired_draft() -> tuple[InMemoryStore, dict]:
    """One expired SLA draft whose evidence still holds: a
    moderate case that waited past the 5-day budget and was
    approved anyway (the one-sided move the draft proposes)."""
    store = InMemoryStore()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    return store, proposal


def test_preview_reports_would_renew_and_writes_nothing():
    store, proposal = _store_with_expired_draft()
    before = copy.deepcopy(store.get_document(PROPOSALS_KEY))
    preview = preview_refresh_proposal(
        store, store.all_records(), proposal["proposal_id"], now=LATER
    )
    assert preview["outcome"] == "would_renew"
    assert preview["proposal_id"] == proposal["proposal_id"]
    assert preview["kind"] == "sla_budget"
    assert preview["dimension"] == "moderate"
    assert preview["withdrawal_reason"] is None
    refreshed = preview["refreshed"]
    assert refreshed["proposed_value"] != preview["proposed_value"] or (
        refreshed["current_value"] == preview["current_value"]
    )
    assert refreshed["rationale"]
    assert refreshed["expires_at"] > preview["expires_at"]
    # Nothing written: the stored document is byte-for-byte the
    # same, the draft is still an expired draft, no audit entry.
    assert store.get_document(PROPOSALS_KEY) == before
    stored = load_proposals(store)[0]
    assert stored["status"] == "draft"
    assert "refreshes" not in stored


def test_preview_reports_would_withdraw_with_the_reason():
    store, proposal = _store_with_expired_draft()
    # A second long-wait case the underwriters REJECTED makes the
    # evidence two-sided — the move no longer holds.
    store.save(
        _sla_record("M-BAD", "2026-09-24", "2026-10-03T10:00:00+00:00", status="rejected")
    )
    preview = preview_refresh_proposal(
        store, store.all_records(), proposal["proposal_id"], now=LATER
    )
    assert preview["outcome"] == "would_withdraw"
    assert preview["refreshed"] is None
    assert "no longer holds" in preview["withdrawal_reason"]


def test_the_sweep_produces_exactly_the_previewed_outcomes():
    store, proposal = _store_with_expired_draft()
    preview = preview_refresh_expired_proposals(store, store.all_records(), now=LATER)
    assert preview["counts"] == {"swept": 1, "would_renew": 1, "would_withdraw": 0}
    projected = preview["would_renew"][0]

    report = refresh_expired_proposals(store, store.all_records(), "U. Writer", now=LATER)
    assert [p["proposal_id"] for p in report["renewed"]] == [
        p["proposal_id"] for p in preview["would_renew"]
    ]
    assert report["withdrawn"] == preview["would_withdraw"] == []
    stored = load_proposals(store)[0]
    assert stored["proposed_value"] == projected["refreshed"]["proposed_value"]
    assert stored["expires_at"] == projected["refreshed"]["expires_at"]


def test_preview_sweep_with_nothing_expired_is_empty():
    store, _ = _store_with_expired_draft()
    preview = preview_refresh_expired_proposals(
        store, store.all_records(), now=T0 + timedelta(days=1)
    )
    assert preview["counts"] == {"swept": 0, "would_renew": 0, "would_withdraw": 0}
    assert preview["would_renew"] == [] and preview["would_withdraw"] == []


def test_preview_keeps_the_refresh_validations():
    store, proposal = _store_with_expired_draft()
    with pytest.raises(KeyError):
        preview_refresh_proposal(store, store.all_records(), "NOPE", now=LATER)
    # Still live at T0+1: the refresh would refuse, so the
    # preview refuses with the same words.
    with pytest.raises(ValueError, match="still live"):
        preview_refresh_proposal(
            store, store.all_records(), proposal["proposal_id"],
            now=T0 + timedelta(days=1),
        )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(app)


def test_preview_endpoints(client):
    res = client.get("/calibration/proposals/refresh-expired/preview")
    assert res.status_code == 200
    body = res.json()
    assert body["counts"] == {"swept": 0, "would_renew": 0, "would_withdraw": 0}

    res = client.get("/calibration/proposals/NOPE/refresh/preview")
    assert res.status_code == 404
