"""Round 19: a refused draft, visible on the case itself.

Round 18 made refused status-note drafts countable — recorded
on the record before the refusal raises, aggregated in
/metrics beside the repairs. But countable is not where the
work is: the underwriter who opens the case itself saw only
that no note exists, indistinguishable from nobody having
tried. The refusal now rides the case's own surfaces — the
queue item for the breached case and a case read beside the
status-note read — carrying when the last draft was refused,
which checks it failed, whether a steered candidate was
produced and still failed or the refusal was outright on
substance, and how many refusals the record carries. The
refusal is usually about prose on this very record; the
person who can fix that prose should meet it where the work
is, not in an aggregate.
"""

from __future__ import annotations

import copy

import pytest

from underwriting_agent.insights import (
    latest_status_note_refusal,
    queue_item_for,
)
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def old_submission(sid: str = "SUB-9001") -> dict:
    sub = copy.deepcopy(SAMPLES["SUB-1001"])
    sub["submission_id"] = sid
    sub["submitted_at"] = "2026-08-01"
    return sub


def _poison_gaps(
    service: SubmissionService, gap_sources: list[str], sid: str = "SUB-9001"
) -> None:
    record = service._get_store().get(sid)
    record.result.facts.missing_elements = []
    record.result.facts.gap_sources = gap_sources
    service._get_store().save(record)


def _refuse_a_draft(service: SubmissionService) -> None:
    service.analyze(old_submission())
    _poison_gaps(
        service, ["Broker quoted the premium at $12,000 for this file"]
    )
    with pytest.raises(ValueError, match="not stored"):
        service.draft_status_note("SUB-9001")


def _record(
    sid: str, refusals: list[dict] | None = None
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={"insured_name": f"Insured {sid}"},
        status_note_refusals=refusals or [],
    )


def _refusal(
    refused_at: str, checks: list[str], steered: bool = False
) -> dict:
    return {
        "refused_at": refused_at,
        "failed_checks": checks,
        "steered": steered,
        "repair_families": [],
    }


# --- the projection --------------------------------------------------------------


def test_no_refusals_reads_as_none_not_an_empty_story():
    assert latest_status_note_refusal(_record("A")) is None


def test_the_projection_carries_the_latest_refusal_and_the_count():
    record = _record(
        "A",
        refusals=[
            _refusal("2026-10-01T09:00:00+00:00", ["no_bind_language"]),
            _refusal(
                "2026-10-03T11:30:00+00:00",
                ["no_forward_motion_in_letter"],
                steered=True,
            ),
        ],
    )
    latest = latest_status_note_refusal(record)
    assert latest["refused_at"] == "2026-10-03T11:30:00+00:00"
    assert latest["failed_checks"] == ["no_forward_motion_in_letter"]
    assert latest["steered"] is True
    assert latest["refusal_count"] == 2


# --- the queue item ---------------------------------------------------------------


def test_the_queue_item_for_a_breached_case_carries_the_refusal():
    service = make_service()
    _refuse_a_draft(service)
    record = service._get_store().get("SUB-9001")
    item = queue_item_for(record)
    assert item is not None and item["sla_state"] == "breached"
    refusal = item["status_note_refusal"]
    assert refusal is not None
    assert refusal["failed_checks"] == ["no_firm_premium"]
    assert refusal["steered"] is False
    assert refusal["refusal_count"] == 1
    assert refusal["refused_at"]


def test_the_queue_view_carries_the_refusal_on_the_breach_row():
    service = make_service()
    _refuse_a_draft(service)
    view = service.queue()
    (row,) = [
        item
        for item in view["breaches"]
        if item["submission_id"] == "SUB-9001"
    ]
    assert row["status_note_refusal"]["failed_checks"] == [
        "no_firm_premium"
    ]


def test_a_case_with_no_refusal_carries_none_on_its_queue_item():
    service = make_service()
    service.analyze(old_submission())
    record = service._get_store().get("SUB-9001")
    item = queue_item_for(record)
    assert item["status_note_refusal"] is None


# --- the case read ------------------------------------------------------------------


def test_the_case_read_names_the_refusal_and_keeps_the_history():
    service = make_service()
    _refuse_a_draft(service)
    view = service.status_note_refusal("SUB-9001")
    assert view["submission_id"] == "SUB-9001"
    assert view["refusal_count"] == 1
    assert view["latest_refusal"]["failed_checks"] == ["no_firm_premium"]
    assert view["latest_refusal"]["steered"] is False
    (stored,) = view["refusals"]
    assert stored["failed_checks"] == ["no_firm_premium"]


def test_the_case_read_for_a_clean_case_says_no_refusal_not_no_case():
    service = make_service()
    service.analyze(old_submission())
    view = service.status_note_refusal("SUB-9001")
    assert view is not None
    assert view["refusal_count"] == 0
    assert view["latest_refusal"] is None
    assert view["refusals"] == []


def test_the_case_read_for_an_unknown_submission_is_none():
    service = make_service()
    assert service.status_note_refusal("SUB-0000") is None


# --- the API -------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app

    service = make_service()
    _refuse_a_draft(service)
    service.analyze(old_submission("SUB-9002"))
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(app)


def test_the_refusal_reads_over_the_api(client):
    res = client.get("/submissions/SUB-9001/status-note/refusal")
    assert res.status_code == 200
    body = res.json()
    assert body["refusal_count"] == 1
    assert body["latest_refusal"]["failed_checks"] == ["no_firm_premium"]


def test_a_clean_case_reads_as_no_refusal_over_the_api(client):
    res = client.get("/submissions/SUB-9002/status-note/refusal")
    assert res.status_code == 200
    assert res.json()["latest_refusal"] is None


def test_an_unknown_submission_404s_over_the_api(client):
    res = client.get("/submissions/SUB-0000/status-note/refusal")
    assert res.status_code == 404
