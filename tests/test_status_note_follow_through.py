"""Round 8: status-note follow-through.

The round-7 status note records what a breached case was waiting
on. Without follow-through, that note just sits on the old record
while the case moves — and risks reading as the current state of
the case forever. Now the note snapshots its outstanding items in
data, a resubmission from the same insured is checked against the
snapshot in code, the new packet carries the follow-through
section, and the note's own record is marked answered / partially
answered / still outstanding.
"""

from __future__ import annotations

from underwriting_agent.insights import (
    follow_through_packet_lines,
    status_note_follow_through,
)
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


def _note(**overrides) -> dict:
    note = {
        "submission_id": "OLD-1",
        "drafted_at": "2026-10-08T09:00:00+00:00",
        "text": "INTERNAL DRAFT …",
        "outstanding": {
            "missing_elements": ["application", "loss runs"],
            "gap_sources": [],
            "blocking_lines": [],
        },
    }
    note.update(overrides)
    return note


def test_follow_through_states():
    note = _note()
    answered = status_note_follow_through(note, missing_elements=[], blocking_lines=[])
    assert answered["state"] == "answered"
    assert {e["item"] for e in answered["answered"]} == {"application", "loss runs"}
    assert answered["still_outstanding"] == []

    partial = status_note_follow_through(
        note, missing_elements=["loss runs"], blocking_lines=[]
    )
    assert partial["state"] == "partially_answered"
    assert [e["item"] for e in partial["answered"]] == ["application"]
    assert [e["item"] for e in partial["still_outstanding"]] == ["loss runs"]

    still = status_note_follow_through(
        note, missing_elements=["application", "loss runs"], blocking_lines=[]
    )
    assert still["state"] == "still_outstanding"
    assert still["answered"] == []


def test_follow_through_blocking_lines_and_line_scoping():
    note = _note(
        outstanding={
            "missing_elements": [],
            "gap_sources": [],
            "blocking_lines": [
                {"line_of_business": "property", "missing_items": ["values schedule"]},
                {"line_of_business": "cyber", "missing_items": ["mfa attestation"]},
            ],
        }
    )
    # Scoped to the property line's run: only its items are judged.
    scoped = status_note_follow_through(
        note,
        missing_elements=[],
        blocking_lines=[],
        line_of_business="property",
    )
    assert scoped["state"] == "answered"
    assert scoped["answered"] == [{"line": "property", "item": "values schedule"}]
    # Unscoped, against a bundle result still blocked on cyber.
    whole = status_note_follow_through(
        note,
        missing_elements=[],
        blocking_lines=[
            {"line_of_business": "cyber", "missing_items": ["mfa attestation"]}
        ],
    )
    assert whole["state"] == "partially_answered"
    assert whole["still_outstanding"] == [
        {"line": "cyber", "item": "mfa attestation"}
    ]


def test_follow_through_unrecorded_and_empty_notes():
    legacy = _note()
    del legacy["outstanding"]
    follow = status_note_follow_through(legacy, missing_elements=[], blocking_lines=[])
    assert follow["recorded"] is False
    assert follow["state"] == "not_recorded"
    lines = follow_through_packet_lines(legacy, follow)
    assert any("predates outstanding-item tracking" in line for line in lines)

    empty = _note(
        outstanding={"missing_elements": [], "gap_sources": [], "blocking_lines": []}
    )
    follow = status_note_follow_through(empty, missing_elements=[], blocking_lines=[])
    assert follow["state"] == "nothing_outstanding"


def _service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def _submission(sid: str, submitted_at: str, **overrides) -> dict:
    payload = {
        "submission_id": sid,
        "insured_name": "Follow Co",
        "broker_name": "Broker Q",
        "line_of_business": "general",
        "class_of_business": "retail",
        "submitted_at": submitted_at,
        "documents": [],
        "losses": [],
    }
    payload.update(overrides)
    return payload


_COMPLETE = {
    "documents": [
        {"doc_type": "acord_form", "document_id": "A1", "raw_text": "Application."}
    ],
    "losses": [
        {"date": "2025-03-01", "amount_incurred": 1200, "description": "Slip and fall"}
    ],
}


def _breached_case_with_note(service: SubmissionService) -> None:
    """FT-1: an ancient, incomplete submission — it breaches any
    SLA budget, and its drafted note snapshots both missing items."""
    service.analyze(_submission("FT-1", "2020-01-01"))
    note = service.draft_status_note("FT-1")
    assert set(note["outstanding"]["missing_elements"]) == {
        "application",
        "loss_runs_or_no_loss",
    }


def test_resubmission_answers_the_note_and_the_packet_says_so():
    service = _service()
    _breached_case_with_note(service)

    result = service.analyze(_submission("FT-2", "2026-10-09", **_COMPLETE))
    packet = result.drafts.underwriter_packet
    assert "Follow-through on the underwriter status note for FT-1" in packet
    assert "the note is answered" in packet
    assert "still outstanding: none" in packet

    note = service.status_note("FT-1")
    follow = note["follow_through"]
    assert follow["state"] == "answered"
    assert follow["by_submission_id"] == "FT-2"
    assert follow["still_outstanding"] == []
    # The original note text and snapshot are never rewritten.
    assert note["text"].startswith("INTERNAL DRAFT")
    assert note["outstanding"]["missing_elements"]


def test_resubmission_that_brings_nothing_leaves_the_note_open():
    service = _service()
    _breached_case_with_note(service)

    result = service.analyze(_submission("FT-2", "2026-10-09"))
    packet = result.drafts.underwriter_packet
    assert "none of the note's outstanding items arrived" in packet
    follow = service.status_note("FT-1")["follow_through"]
    assert follow["state"] == "still_outstanding"
    assert {e["item"] for e in follow["still_outstanding"]} == {
        "application",
        "loss_runs_or_no_loss",
    }


def test_partial_resubmission_marks_the_note_partially_answered():
    service = _service()
    _breached_case_with_note(service)

    result = service.analyze(
        _submission(
            "FT-2",
            "2026-10-09",
            documents=[
                {"doc_type": "acord_form", "document_id": "A1", "raw_text": "App."}
            ],
        )
    )
    assert "the note is partially answered" in result.drafts.underwriter_packet
    follow = service.status_note("FT-1")["follow_through"]
    assert follow["state"] == "partially_answered"
    assert [e["item"] for e in follow["answered"]] == ["application"]
    assert [e["item"] for e in follow["still_outstanding"]] == [
        "loss_runs_or_no_loss"
    ]


def test_no_prior_note_no_follow_through_section():
    service = _service()
    result = service.analyze(_submission("FT-9", "2026-10-09", **_COMPLETE))
    assert "Follow-through" not in result.drafts.underwriter_packet
    assert service.status_note("FT-9") is None


def test_bundle_resubmission_marks_the_note_from_the_rollup():
    service = _service()
    _breached_case_with_note(service)

    bundle = {
        "submission_id": "FT-B1",
        "insured_name": "Follow Co",
        "broker_name": "Broker Q",
        "submitted_at": "2026-10-09",
        "lines": [
            {"line_of_business": "general", **_COMPLETE},
            {"line_of_business": "property", "documents": [], "losses": []},
        ],
    }
    result = service.analyze(bundle)
    assert result.multi_line
    follow = service.status_note("FT-1")["follow_through"]
    assert follow["by_submission_id"] == "FT-B1"
    # The property line still owes an application, so at bundle
    # level the note is not answered — the rollup is the record.
    assert follow["state"] in ("partially_answered", "still_outstanding")
    assert follow["state"] != "answered"
