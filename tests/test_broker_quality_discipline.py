"""Round 19: the broker quality report through the drafting discipline.

The quality report was the third broker-facing text and the
only one answering to no discipline: composed in code from
record prose — the broker's name, the missing-element labels
its submissions carried — and served unchecked, where the
letter and the status note are guardrail-validated and
steered. It now runs through the same gate
(``guardrails.validate_broker_quality_draft``) and, on a
wording-only failure, the same steered repair in the report's
own shapes. Substance — a premium figure riding in on a
missing-element label — is refused: the report serves its
computed sections with the draft withheld and the refusal
named, and no premium or coverage language can survive into
a served draft. Each drafting the service serves is recorded
as an event, and the repairs and refusals join the
drafting-repair family in /metrics as their own named
source, beside the letter advisories and the note repairs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.guardrails import validate_broker_quality_draft
from underwriting_agent.insights import broker_quality_report
from underwriting_agent.schemas import AgentResult, SubmissionFacts, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

BROKER = "Harborline Brokers"


def _record(
    sid: str,
    broker: str = BROKER,
    *,
    completeness: float = 0.8,
    discrepancies: int = 0,
    missing: list[str] | None = None,
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            facts=SubmissionFacts(
                completeness_score=completeness,
                discrepancy_count=discrepancies,
                missing_elements=missing or [],
            ),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": broker,
            "submitted_at": "2026-09-01",
        },
    )


def _clean_records() -> list[ApprovalRecord]:
    return [
        _record("Q-1", completeness=0.7, discrepancies=1),
        _record("Q-2", completeness=0.9, missing=["loss runs"]),
    ]


# --- the gate ----------------------------------------------------------------------


def test_a_clean_draft_passes_the_gate():
    report = broker_quality_report(_clean_records(), BROKER)
    validation = validate_broker_quality_draft(report["draft"], BROKER)
    assert validation.passed


def test_the_gate_names_a_missing_broker_reference():
    validation = validate_broker_quality_draft(
        "DRAFT — a report about nobody in particular.", BROKER
    )
    failed = {c.name for c in validation.checks if not c.passed}
    assert failed == {"references_broker"}


# --- the report under the discipline -------------------------------------------------


def test_a_clean_report_serves_its_draft_with_its_validation():
    report = broker_quality_report(_clean_records(), BROKER)
    assert report["draft"].startswith(
        f"DRAFT — submission quality report for {BROKER}"
    )
    assert report["draft_validation"]["passed"] is True
    assert "draft_repair" not in report
    assert "draft_refusal" not in report


def test_a_wording_failure_is_steered_in_the_reports_own_shapes():
    records = [
        _record(
            "Q-1",
            missing=["we will proceed with the file once audits land"],
        ),
        _record("Q-2", completeness=0.9),
    ]
    report = broker_quality_report(records, BROKER)
    draft = report["draft"]
    assert draft is not None
    assert "will proceed" not in draft
    assert "decided on its own file" in draft
    repair = report["draft_repair"]
    assert repair["repaired"] is True
    assert repair["source"] == "broker_quality"
    assert repair["repair_families"] == ["forward_motion"]
    assert repair["original_validation"]["passed"] is False
    assert report["draft_validation"]["passed"] is True


def test_an_authority_sentence_comes_out_rather_than_being_rewritten():
    records = [
        _record(
            "Q-1",
            missing=["the case sits within our standard authority"],
        ),
        _record("Q-2", completeness=0.9),
    ]
    report = broker_quality_report(records, BROKER)
    assert report["draft"] is not None
    assert "standard authority" not in report["draft"]
    assert report["draft_repair"]["repair_families"] == ["authority"]


def test_a_premium_figure_is_refused_and_the_draft_withheld():
    records = [
        _record("Q-1", missing=["premium at $12,000 for the term"]),
        _record("Q-2", completeness=0.9),
    ]
    report = broker_quality_report(records, BROKER)
    # The computed sections are facts and still serve; the
    # draft that carried the figure does not.
    assert report["submissions"] == 2
    assert report["top_missing_elements"][0]["element"].startswith(
        "premium at $12,000"
    )
    assert report["draft"] is None
    refusal = report["draft_refusal"]
    assert refusal["source"] == "broker_quality"
    assert refusal["failed_checks"] == ["no_firm_premium"]
    assert refusal["steered"] is False
    assert report["draft_validation"]["passed"] is False


def test_a_mixed_failure_is_refused_unsteered_with_its_families():
    records = [
        _record(
            "Q-1",
            missing=["we will proceed; the premium at $12,000 stands"],
        ),
        _record("Q-2", completeness=0.9),
    ]
    report = broker_quality_report(records, BROKER)
    assert report["draft"] is None
    refusal = report["draft_refusal"]
    assert set(refusal["failed_checks"]) == {
        "no_forward_motion_in_letter",
        "no_firm_premium",
    }
    assert refusal["steered"] is False
    assert refusal["repair_families"] == ["forward_motion"]


# --- the events and the family ---------------------------------------------------------


def _service_with(*records: ApprovalRecord) -> SubmissionService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


def test_each_served_report_records_one_drafting_event():
    service = _service_with(*_clean_records())
    service.broker_quality_report(BROKER)
    service.broker_quality_report(BROKER)
    events = service.broker_quality_drafting_events()
    # The report is drafted on read: two reads, two draftings.
    assert [e["outcome"] for e in events] == ["clean", "clean"]
    assert all(e["broker"] == BROKER for e in events)


def test_repairs_and_refusals_join_the_metrics_family_by_source():
    service = _service_with(
        _record(
            "Q-1",
            broker="Wording Brokers",
            missing=["we will proceed with the file once audits land"],
        ),
        _record("Q-2", broker="Wording Brokers", completeness=0.9),
        _record(
            "Q-3",
            broker="Premium Brokers",
            missing=["premium at $12,000 for the term"],
        ),
        _record("Q-4", broker="Premium Brokers", completeness=0.9),
        _record("Q-5", completeness=0.7, discrepancies=1),
        _record("Q-6", completeness=0.9, missing=["loss runs"]),
    )
    service.broker_quality_report("Wording Brokers")
    service.broker_quality_report("Premium Brokers")
    service.broker_quality_report(BROKER)
    family = service.metrics()["status_note_repairs"]
    broker = family["broker_quality"]
    assert broker["reports_drafted"] == 3
    assert broker["repaired_reports"] == 1
    assert broker["refusals"] == 1
    assert broker["refusals_unsteered"] == 1
    assert broker["by_family"] == {"forward_motion": 1}
    assert broker["by_refused_check"] == {"no_firm_premium": 1}
    assert broker["steer_attempts"] == 1
    assert broker["steer_success_rate"] == 1.0
    # The note side of the family is untouched by the new source.
    assert family["notes_drafted"] == 0
    assert family["refusals"] == 0


def test_an_unread_broker_leaves_no_events():
    service = _service_with(*_clean_records())
    assert service.broker_quality_report("Nobody Brokers") is None
    assert service.broker_quality_drafting_events() == []
    family = service.metrics()["status_note_repairs"]
    assert family["broker_quality"]["reports_drafted"] == 0
    assert family["broker_quality"]["steer_success_rate"] is None


# --- the API ------------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app

    service = _service_with(*_clean_records())
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(app)


def test_the_report_reads_over_the_api_with_its_validation(client):
    res = client.get(f"/reports/brokers/{BROKER}/quality")
    assert res.status_code == 200
    body = res.json()
    assert body["draft"].startswith("DRAFT — submission quality report")
    assert body["draft_validation"]["passed"] is True


def test_an_unknown_broker_still_404s(client):
    res = client.get("/reports/brokers/Nobody Brokers/quality")
    assert res.status_code == 404
