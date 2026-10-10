"""Round 7: the drafted underwriter status note for SLA breaches.

When an awaiting case crosses the SLA budget for its risk tier,
the agent drafts (never sends) an internal status note: where the
case stands, how far past budget it is, what is outstanding. The
note is composed in code, guardrail-checked with the same text
families as the broker letter before it is stored, and kept on
the record — drafting changes nothing else about the case.
"""

from __future__ import annotations

import copy
from datetime import date

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.guardrails import validate_status_note
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, SQLiteStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def old_submission(sid: str = "SUB-9001") -> dict:
    """A submission that has been waiting far past any SLA budget."""
    sub = copy.deepcopy(SAMPLES["SUB-1001"])
    sub["submission_id"] = sid
    sub["submitted_at"] = "2026-08-01"
    return sub


# ---------------------------------------------------------------------------
# The validator: the note is held to the letter's discipline
# ---------------------------------------------------------------------------


def test_validate_status_note_accepts_a_status_only_note():
    note = (
        "INTERNAL DRAFT — underwriter status note for submission SUB-1\n"
        "Status: awaiting underwriter decision. Submitted 2026-08-01; "
        "70 day(s) in the queue.\n"
        "Next step: underwriter review and decision."
    )
    validation = validate_status_note(note, "SUB-1")
    assert validation.passed, [c.detail for c in validation.checks if not c.passed]


@pytest.mark.parametrize(
    "poison,check",
    [
        ("Coverage is bound for SUB-1 from today.", "no_bind_language"),
        ("The premium is $12,000 for SUB-1.", "no_firm_premium"),
        ("SUB-1 will be approved, count on it.", "no_prohibited_promises"),
        ("We will proceed with SUB-1 now.", "no_forward_motion_in_letter"),
        ("We will be in touch shortly about SUB-1.", "no_forward_motion_in_letter"),
        ("SUB-1 sits within our standard authority.", "no_forward_motion_in_letter"),
        ("A status note with no case reference at all.", "references_submission_id"),
        ("SUB-1 file for SSN 123-45-6789.", "no_pii_in_draft"),
    ],
)
def test_validate_status_note_rejects_poisoned_notes(poison, check):
    validation = validate_status_note(poison, "SUB-1")
    assert not validation.passed
    failed = {c.name for c in validation.checks if not c.passed}
    assert check in failed


# ---------------------------------------------------------------------------
# Drafting: service behaviour
# ---------------------------------------------------------------------------


def test_draft_status_note_for_breached_case():
    service = make_service()
    service.analyze(old_submission())
    note = service.draft_status_note("SUB-9001")
    assert note["submission_id"] == "SUB-9001"
    assert note["validation"]["passed"] is True
    assert note["sla"]["days_over"] > 0
    assert note["sla"]["age_days"] > note["sla"]["budget_days"]
    text = note["text"]
    assert "INTERNAL DRAFT" in text and "not sent" in text
    assert "SUB-9001" in text
    assert "past it" in text  # the SLA overrun is stated
    # Stored on the record, readable back, and nothing else moved:
    # no dispatch, no decision, still awaiting.
    assert service.status_note("SUB-9001") == note
    record = service._get_store().get("SUB-9001")
    assert record.result.status == "awaiting_underwriter"
    assert record.result.external_action_taken is False
    assert record.dispatch_attempts == []


def test_draft_status_note_refuses_within_budget_case():
    service = make_service()
    fresh = copy.deepcopy(SAMPLES["SUB-1001"])
    fresh["submission_id"] = "SUB-9002"
    fresh["submitted_at"] = date.today().isoformat()
    service.analyze(fresh)
    with pytest.raises(ValueError, match="within|budget"):
        service.draft_status_note("SUB-9002")
    assert service.status_note("SUB-9002") is None


def test_draft_status_note_refuses_decided_and_unknown():
    service = make_service()
    service.analyze(old_submission())
    service.approve("SUB-9001", approver="U. Writer", reason="clean file")
    with pytest.raises(ValueError, match="not awaiting"):
        service.draft_status_note("SUB-9001")
    with pytest.raises(KeyError):
        service.draft_status_note("SUB-UNKNOWN")


def test_status_note_names_blocking_lines_for_bundles():
    service = make_service()
    bundle = copy.deepcopy(SAMPLES["SUB-1014"])
    bundle["submission_id"] = "SUB-9016"
    bundle["submitted_at"] = "2026-08-01"
    bundle["lines"][1] = {
        "line_of_business": "directors_and_officers",
        "class_of_business": "hospitality management company",
        "documents": [
            {
                "doc_type": "loss runs",
                "document_id": "sub9016-dno-lr",
                "fields": {"loss_history_present": "yes"},
                "raw_text": "Loss runs 2023-2026: no claims reported in the period.",
            }
        ],
        "losses": [],
        "requested_limit": "2000000",
        "requested_deductible": "25000",
    }
    result = service.analyze(bundle)
    assert result.blocking_lines == ["directors_and_officers"]
    note = service.draft_status_note("SUB-9016")
    assert "Blocking line: directors_and_officers" in note["text"]
    assert note["validation"]["passed"] is True


def test_status_note_survives_a_store_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "notes.db")
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(old_submission())
    note = service.draft_status_note("SUB-9001")
    reloaded = SubmissionService(store=store, checkpointer=False)
    assert reloaded.status_note("SUB-9001") == note


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(api_module.app)


def test_status_note_endpoints(client):
    client.post("/submissions/analyze", json=old_submission("SUB-9101"))
    missing = client.get("/submissions/SUB-9101/status-note")
    assert missing.status_code == 404
    drafted = client.post("/submissions/SUB-9101/status-note")
    assert drafted.status_code == 200
    body = drafted.json()
    assert body["validation"]["passed"] is True
    fetched = client.get("/submissions/SUB-9101/status-note")
    assert fetched.status_code == 200
    assert fetched.json()["text"] == body["text"]


def test_status_note_endpoint_refuses_fresh_case(client):
    fresh = copy.deepcopy(SAMPLES["SUB-1001"])
    fresh["submission_id"] = "SUB-9102"
    fresh["submitted_at"] = date.today().isoformat()
    client.post("/submissions/analyze", json=fresh)
    response = client.post("/submissions/SUB-9102/status-note")
    assert response.status_code == 422
    unknown = client.post("/submissions/SUB-NOPE/status-note")
    assert unknown.status_code == 404
