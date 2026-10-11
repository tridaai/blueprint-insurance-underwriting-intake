"""Final underwriter decisions + triage-vs-decision agreement.

The gate decision approves or rejects the drafted packet; the
FINAL decision — bound / declined / not_taken_up — is what the
desk ultimately did with the risk. It is captured append-only
(a correction supersedes, never edits), and the agreement report
pins its math on this seeded set with known decisions:

    SUB-1001  in_appetite / low        → bound          (agrees)
    SUB-1009  in_appetite / low        → declined       (disagrees)
    SUB-1004  out_of_appetite / elev.  → bound          (bound anyway)
    SUB-2001  out_of_appetite / elev.  → declined       (agrees)
    SUB-2002  out_of_appetite / elev.  → not_taken_up   (never decisive)
    SUB-1002  borderline / high        → bound          (no expectation)
    SUB-1003  incomplete / moderate    → declined       (outside the bands)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.api import app
from underwriting_agent.samples import load_sample_submissions, sample_submission_models
from underwriting_agent.schemas import SubmissionInput
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, SQLiteStore

SAMPLES = {s.submission_id: s for s in sample_submission_models()}
RAW = {s["submission_id"]: s for s in load_sample_submissions()}


def _out_of_appetite(submission_id: str, insured: str) -> SubmissionInput:
    """A second fireworks-manufacturing risk: the excluded class
    drives the same out_of_appetite / elevated triage as SUB-1004."""
    return SubmissionInput.model_validate(
        {
            "submission_id": submission_id,
            "insured_name": insured,
            "broker_name": "Sunbelt Retail Insurance",
            "broker_email_text": f"New property submission for {insured} — fireworks manufacturing and storage.",
            "line_of_business": "property",
            "class_of_business": "fireworks manufacturing",
            "submitted_at": "2026-10-05",
            "effective_date_requested": "2026-11-01",
            "revenue": 4100000,
            "years_in_business": 7,
            "requested_limit": "1200000",
            "requested_deductible": "25000",
            "notes": "Facility stores finished fireworks products; construction is frame, built in 2001.",
            "documents": [
                {
                    "doc_type": "application",
                    "document_id": f"{submission_id.lower()}-app",
                    "fields": {"insured_name": insured, "line_of_business": "property"},
                    "raw_text": f"ACORD application: {insured}, fireworks manufacturing, commercial property.",
                }
            ],
        }
    )


def seeded_service() -> SubmissionService:
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    for submission_id in ("SUB-1001", "SUB-1009", "SUB-1004", "SUB-1002", "SUB-1003"):
        service.analyze(SAMPLES[submission_id])
    service.analyze(_out_of_appetite("SUB-2001", "Pinnacle Pyrotechnics LLC"))
    service.analyze(_out_of_appetite("SUB-2002", "Red Mesa Fireworks LLC"))
    service.record_final_decision("SUB-1001", "bound", actor="U. Writer", decided_on="2026-10-08")
    service.record_final_decision(
        "SUB-1009", "declined", actor="U. Writer", decided_on="2026-10-08",
        reason_code="price",
    )
    service.record_final_decision("SUB-1004", "bound", actor="S. Senior", decided_on="2026-10-09")
    service.record_final_decision(
        "SUB-2001", "declined", actor="S. Senior", decided_on="2026-10-09",
        reason_code="class",
    )
    service.record_final_decision(
        "SUB-2002", "not_taken_up", actor="S. Senior", decided_on="2026-10-10",
        reason_code="broker_withdrew",
    )
    service.record_final_decision("SUB-1002", "bound", actor="U. Writer", decided_on="2026-10-10")
    service.record_final_decision("SUB-1003", "declined", actor="U. Writer", decided_on="2026-10-10")
    return service


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_record_projects_onto_the_result_without_moving_the_gate():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    result = service.record_final_decision(
        "SUB-1001", "bound", actor="U. Writer", decided_on="2026-10-08",
        reason_code="terms_agreed",
    )
    assert result.final_decision == "bound"
    assert result.final_decided_by == "U. Writer"
    assert result.final_decided_on == "2026-10-08"
    assert result.final_reason_code == "terms_agreed"
    # The gate is untouched: capturing the outcome is bookkeeping,
    # not a second gate decision.
    assert result.status == "awaiting_underwriter"
    assert result.decision is None


def test_record_validates_decision_actor_and_date():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    with pytest.raises(ValueError, match="final decision"):
        service.record_final_decision("SUB-1001", "quoted", actor="U. Writer")
    with pytest.raises(ValueError, match="actor"):
        service.record_final_decision("SUB-1001", "bound", actor="")
    with pytest.raises(ValueError, match="ISO date"):
        service.record_final_decision(
            "SUB-1001", "bound", actor="U. Writer", decided_on="last Tuesday"
        )
    with pytest.raises(KeyError):
        service.record_final_decision("SUB-9999", "bound", actor="U. Writer")


def test_correction_supersedes_and_never_edits():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.record_final_decision("SUB-1001", "bound", actor="U. Writer", decided_on="2026-10-08")
    result = service.record_final_decision(
        "SUB-1001", "not_taken_up", actor="U. Writer", decided_on="2026-10-12",
        reason_code="broker_withdrew", note="Broker placed it elsewhere.",
    )
    assert result.final_decision == "not_taken_up"
    record = service._get_store().get("SUB-1001")
    assert len(record.final_decisions) == 2
    first, second = record.final_decisions
    assert first["decision"] == "bound" and first["supersedes"] is None
    assert second["supersedes"] == {
        "decision": "bound", "decided_by": "U. Writer", "decided_on": "2026-10-08",
    }
    rows = service.final_decisions()["decisions"]
    assert rows[0]["decision"] == "not_taken_up"
    assert rows[0]["superseded_count"] == 1


def test_final_decisions_survive_a_store_round_trip(tmp_path):
    path = tmp_path / "store.db"
    service = SubmissionService(store=SQLiteStore(path), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.record_final_decision("SUB-1001", "bound", actor="U. Writer", decided_on="2026-10-08")
    reopened = SubmissionService(store=SQLiteStore(path), checkpointer=False)
    result = reopened.get("SUB-1001")
    assert result.final_decision == "bound"
    assert reopened.triage_agreement()["decided_total"] == 1


# ---------------------------------------------------------------------------
# Agreement math — exact counts per cell on the seeded set
# ---------------------------------------------------------------------------


def test_agreement_counts_are_exact_per_band_and_tier():
    report = seeded_service().triage_agreement()
    assert report["decided_total"] == 7

    bands = report["by_outcome_band"]
    assert bands["in_appetite"] == {
        "decided": 2, "bound": 1, "declined": 1, "not_taken_up": 0,
        "decisive": 2, "agreed": 1, "disagreed": 1,
        "expected_decision": "bound",
    }
    assert bands["out_of_appetite"] == {
        "decided": 3, "bound": 1, "declined": 1, "not_taken_up": 1,
        "decisive": 2, "agreed": 1, "disagreed": 1,
        "expected_decision": "declined",
    }
    assert bands["borderline"] == {
        "decided": 1, "bound": 1, "declined": 0, "not_taken_up": 0,
        "decisive": 0, "agreed": 0, "disagreed": 0,
        "expected_decision": None,
    }
    assert report["incomplete_decided"] == 1

    tiers = report["by_risk_tier"]
    assert tiers["low"] == {"decided": 2, "bound": 1, "declined": 1, "not_taken_up": 0}
    assert tiers["elevated"] == {"decided": 3, "bound": 1, "declined": 1, "not_taken_up": 1}
    assert tiers["high"] == {"decided": 1, "bound": 1, "declined": 0, "not_taken_up": 0}
    assert tiers["moderate"] == {"decided": 1, "bound": 0, "declined": 1, "not_taken_up": 0}

    assert report["decisive"] == 4
    assert report["agreed"] == 2
    assert report["disagreed"] == 2
    assert report["agreement_rate"] == 0.5
    assert report["reason_codes"] == {"price": 1, "class": 1, "broker_withdrew": 1}


def test_agreement_carries_the_small_sample_note():
    report = seeded_service().triage_agreement()
    assert report["small_sample"] is True
    assert "read the counts, not the rate" in report["note"]


def test_agreement_counts_the_current_decision_only():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])  # in_appetite
    service.record_final_decision("SUB-1001", "bound", actor="U. Writer")
    service.record_final_decision("SUB-1001", "declined", actor="U. Writer")
    band = service.triage_agreement()["by_outcome_band"]["in_appetite"]
    assert band["decided"] == 1
    assert band["bound"] == 0 and band["declined"] == 1
    assert band["disagreed"] == 1


def test_no_decisions_is_an_empty_honest_report():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    report = service.triage_agreement()
    assert report["decided_total"] == 0
    assert report["agreement_rate"] is None
    assert service.final_decisions() == {"decisions": [], "count": 0}


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    test_client = TestClient(app)
    res = test_client.post("/submissions/analyze", json=RAW["SUB-1004"])
    assert res.status_code == 200
    return test_client


def test_final_decision_endpoints(client):
    recorded = client.post(
        "/submissions/SUB-1004/final-decision",
        json={"actor": "S. Senior", "decision": "bound", "decided_on": "2026-10-09"},
    )
    assert recorded.status_code == 200
    assert recorded.json()["final_decision"] == "bound"

    rows = client.get("/decisions").json()
    assert rows["count"] == 1
    assert rows["decisions"][0]["submission_id"] == "SUB-1004"
    assert rows["decisions"][0]["triage_outcome"] == "out_of_appetite"

    agreement = client.get("/decisions/agreement").json()
    assert agreement["by_outcome_band"]["out_of_appetite"]["bound"] == 1
    assert agreement["by_outcome_band"]["out_of_appetite"]["disagreed"] == 1
    assert agreement["small_sample"] is True

    bad = client.post(
        "/submissions/SUB-1004/final-decision",
        json={"actor": "S. Senior", "decision": "quoted"},
    )
    assert bad.status_code == 422
    unknown = client.post(
        "/submissions/SUB-9999/final-decision",
        json={"actor": "S. Senior", "decision": "bound"},
    )
    assert unknown.status_code == 404
    no_actor = client.post(
        "/submissions/SUB-1004/final-decision", json={"decision": "bound"}
    )
    assert no_actor.status_code == 422
