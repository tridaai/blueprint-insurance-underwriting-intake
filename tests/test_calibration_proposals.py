"""Round 8: calibration proposals + the threshold settings record.

The calibration view showed the arithmetic; somebody still had to
notice the row that mattered and go change configuration by hand.
Now the scan drafts a proposal when — and only when — the moved
referrals carry one-sided decision evidence, a human accepts or
rejects it, and accepting writes the new threshold into a settings
record the classifier actually reads, with an audit history.
"""

from __future__ import annotations

import pytest

from underwriting_agent.proposals import (
    decide_proposal,
    draft_proposals,
    effective_thresholds,
    load_proposals,
    load_settings,
)
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, SQLiteStore

CLAIM_1 = {"date": "2024-02-01", "amount_incurred": 5000, "description": "Water damage", "status": "closed"}
CLAIM_2 = {"date": "2025-06-01", "amount_incurred": 30000, "description": "Fire damage", "status": "open"}
CLAIM_3 = {"date": "2026-01-15", "amount_incurred": 30000, "description": "Theft", "status": "open"}


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


def _service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def _chain(service: SubmissionService, *, decide_b: str | None = "rejected") -> None:
    """CAL-A, then renewal CAL-B carrying a new $30k claim — under
    the $50k default floor, so the current thresholds do NOT refer
    it — then the underwriter's decision on CAL-B."""
    service.analyze(_submission("CAL-A", [CLAIM_1]))
    service.analyze(_submission("CAL-B", [CLAIM_1, CLAIM_2]))
    if decide_b == "rejected":
        service.reject("CAL-B", reviewer="U. Writer", reason="loss trend")
    elif decide_b == "approved":
        service.approve("CAL-B", approver="U. Writer", reason="acceptable")


# ---------------------------------------------------------------------------
# Store documents
# ---------------------------------------------------------------------------


def test_store_documents_round_trip(tmp_path):
    for store in (InMemoryStore(), SQLiteStore(tmp_path / "docs.db")):
        assert store.get_document("nope") is None
        store.save_document("k", {"a": 1})
        assert store.get_document("k") == {"a": 1}
        store.save_document("k", {"a": 2})
        assert store.get_document("k") == {"a": 2}


# ---------------------------------------------------------------------------
# Drafting from evidence
# ---------------------------------------------------------------------------


def test_proposal_drafted_from_one_sided_decision_evidence():
    service = _service()
    _chain(service)
    out = service.draft_calibration_proposals()
    assert len(out["drafted"]) == 1
    proposal = out["drafted"][0]
    assert proposal["dimension"] == "new_claim_min_incurred"
    # 25k and 10k both gain CAL-B; the smallest sufficient change wins.
    assert proposal["proposed_value"] == 25000.0
    assert proposal["current_value"] == 50000.0
    assert proposal["direction"] == "lower"
    assert proposal["status"] == "draft"
    evidence = proposal["evidence"]
    assert evidence["gained_referral_ids"] == ["CAL-B"]
    assert evidence["gained_decisions"] == {
        "approved": 0,
        "rejected": 1,
        "undecided": 0,
    }
    assert "rejected every one" in proposal["rationale"]


def test_no_proposal_without_decision_evidence():
    service = _service()
    _chain(service, decide_b=None)  # CAL-B still awaiting a decision
    out = service.draft_calibration_proposals()
    assert out["drafted"] == []
    assert out["proposals"] == []


def test_no_proposal_when_the_move_is_two_sided():
    service = _service()
    _chain(service, decide_b="approved")
    # The $30k claim never referred CAL-B at the current floor, and
    # lowering would add a referral on a case the underwriter
    # approved — no proposal.
    out = service.draft_calibration_proposals()
    assert out["drafted"] == []


def test_raise_proposal_when_referrals_changed_nothing():
    service = _service()
    # A $60k new claim clears the $50k floor: CAL-B IS referred at
    # the current setting — and the underwriter approved it anyway.
    service.analyze(_submission("CAL-A", [CLAIM_1]))
    big = dict(CLAIM_2, amount_incurred=60000)
    service.analyze(_submission("CAL-B", [CLAIM_1, big]))
    service.approve("CAL-B", approver="U. Writer", reason="fine as-is")
    out = service.draft_calibration_proposals()
    assert len(out["drafted"]) == 1
    proposal = out["drafted"][0]
    assert proposal["dimension"] == "new_claim_min_incurred"
    assert proposal["direction"] == "raise"
    # The sweep grid's only step above the $50k current that still
    # loses the referral is $100k.
    assert proposal["proposed_value"] == 100000.0
    assert proposal["evidence"]["lost_referral_ids"] == ["CAL-B"]
    assert proposal["evidence"]["lost_decisions"]["approved"] == 1


def test_drafting_never_duplicates_a_seen_move():
    service = _service()
    _chain(service)
    service.draft_calibration_proposals()
    out = service.draft_calibration_proposals()
    assert out["drafted"] == []
    assert len(out["proposals"]) == 1


# ---------------------------------------------------------------------------
# Deciding: acceptance is the consequential write
# ---------------------------------------------------------------------------


def test_acceptance_writes_settings_the_classifier_reads():
    service = _service()
    _chain(service)
    proposal = service.draft_calibration_proposals()["drafted"][0]

    # Before acceptance, a third renewal with another $30k claim is
    # NOT referred (floor still $50k)…
    control = _service()
    _chain(control, decide_b=None)
    control_result = control.analyze(_submission("CAL-C", [CLAIM_1, CLAIM_2, CLAIM_3]))
    assert not any(
        "renewal diff" in t for t in control_result.triage.referral_triggers
    )

    decided = service.decide_calibration_proposal(
        proposal["proposal_id"], "U. Writer", accept=True, note="evidence is clear"
    )
    assert decided["status"] == "accepted"
    assert decided["settings_written"]["new_claim_min_incurred"] == 25000.0

    store = service._get_store()
    assert effective_thresholds(store)["new_claim_min_incurred"] == 25000.0
    settings = load_settings(store)
    assert settings["updated_by"] == "U. Writer"
    assert settings["history"] == [
        {
            "proposal_id": proposal["proposal_id"],
            "dimension": "new_claim_min_incurred",
            "from": 50000.0,
            "to": 25000.0,
            "actor": "U. Writer",
            "at": settings["history"][0]["at"],
        }
    ]
    view = service.diff_threshold_settings()
    assert view["source"] == "accepted_proposal"
    report = service.diff_trigger_calibration()
    assert report["current_thresholds"]["new_claim_min_incurred"] == 25000.0
    assert report["threshold_source"] == "accepted_proposal"

    # …and after acceptance the same third renewal IS referred, on
    # the change itself.
    result = service.analyze(_submission("CAL-C", [CLAIM_1, CLAIM_2, CLAIM_3]))
    assert any(
        "renewal diff — new claims" in t for t in result.triage.referral_triggers
    )


def test_rejection_changes_nothing_and_is_final():
    service = _service()
    _chain(service)
    proposal = service.draft_calibration_proposals()["drafted"][0]
    decided = service.decide_calibration_proposal(
        proposal["proposal_id"], "U. Writer", accept=False, note="not yet"
    )
    assert decided["status"] == "rejected"
    store = service._get_store()
    assert load_settings(store) is None
    assert effective_thresholds(store)["new_claim_min_incurred"] == 50000.0
    # The rejected move is not re-drafted, and the proposal cannot
    # be decided twice.
    assert service.draft_calibration_proposals()["drafted"] == []
    with pytest.raises(ValueError, match="decided once"):
        service.decide_calibration_proposal(
            proposal["proposal_id"], "U. Writer", accept=True
        )
    with pytest.raises(KeyError):
        service.decide_calibration_proposal("PROP-9999", "U. Writer", accept=True)


def test_proposals_module_functions_directly():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    _chain(service)
    out = draft_proposals(store, store.all_records())
    assert len(out["drafted"]) == 1
    assert load_proposals(store)[0]["proposal_id"] == "PROP-0001"
    decided = decide_proposal(store, "PROP-0001", "U", accept=True)
    assert decided["status"] == "accepted"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def test_proposal_endpoints(client):
    assert client.post("/submissions/analyze", json=_submission("CAL-A", [CLAIM_1])).status_code == 200
    assert client.post("/submissions/analyze", json=_submission("CAL-B", [CLAIM_1, CLAIM_2])).status_code == 200
    assert client.post(
        "/submissions/CAL-B/reject", json={"actor": "U. Writer", "reason": "trend"}
    ).status_code == 200

    settings = client.get("/calibration/settings").json()
    assert settings["source"] == "environment"

    drafted = client.post("/calibration/proposals/draft").json()
    assert len(drafted["drafted"]) == 1
    proposal_id = drafted["drafted"][0]["proposal_id"]

    listing = client.get("/calibration/proposals").json()
    assert [p["proposal_id"] for p in listing["proposals"]] == [proposal_id]

    decided = client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "accepted"

    settings = client.get("/calibration/settings").json()
    assert settings["source"] == "accepted_proposal"
    assert settings["thresholds"]["new_claim_min_incurred"] == 25000.0

    again = client.post(
        f"/calibration/proposals/{proposal_id}/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert again.status_code == 422
    missing = client.post(
        "/calibration/proposals/PROP-9999/decide",
        json={"actor": "U. Writer", "accept": True},
    )
    assert missing.status_code == 404
    anonymous = client.post(
        f"/calibration/proposals/{proposal_id}/decide", json={"accept": True}
    )
    assert anonymous.status_code == 422  # a threshold change needs its decider
