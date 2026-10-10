"""Round 9: the follow-through digest in the inbox.

Round 8 marked each status note answered / partially / still
outstanding on its own record — but the underwriter had to open
the record to see it. The inbox now digests every note: which
loops a resubmission closed, which it closed halfway, and which
are still open, oldest outstanding first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import follow_through_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _record(
    sid: str,
    insured: str,
    note: dict | None,
    status: str = "awaiting_underwriter",
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=insured,
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={"insured_name": insured, "broker_name": "Broker Q"},
        status_note=note,
    )


def _note(drafted_at: str, follow: dict | None = None, **overrides) -> dict:
    note = {
        "submission_id": "X",
        "drafted_at": drafted_at,
        "text": "INTERNAL DRAFT …",
        "outstanding": {
            "missing_elements": ["application", "loss runs"],
            "gap_sources": [],
            "blocking_lines": [],
        },
    }
    if follow is not None:
        note["follow_through"] = follow
    note.update(overrides)
    return note


def _follow(state: str, by: str = "NEW-1") -> dict:
    answered = (
        [{"line": "", "item": "application"}, {"line": "", "item": "loss runs"}]
        if state == "answered"
        else [{"line": "", "item": "application"}]
        if state == "partially_answered"
        else []
    )
    still = (
        []
        if state == "answered"
        else [{"line": "", "item": "loss runs"}]
        if state == "partially_answered"
        else [{"line": "", "item": "application"}, {"line": "", "item": "loss runs"}]
    )
    return {
        "recorded": True,
        "state": state,
        "answered": answered,
        "still_outstanding": still,
        "by_submission_id": by,
        "checked_at": "2026-10-09T12:00:00+00:00",
    }


def test_digest_of_an_empty_store():
    digest = follow_through_digest([], now=NOW)
    assert digest["total_notes"] == 0
    assert digest["entries"] == []
    assert digest["counts"] == {}


def test_unreached_note_is_awaiting_with_its_snapshot_outstanding():
    records = [_record("OLD-1", "Insured A", _note("2026-10-05T09:00:00+00:00"))]
    digest = follow_through_digest(records, now=NOW)
    assert digest["counts"] == {"awaiting_resubmission": 1}
    entry = digest["entries"][0]
    assert entry["state"] == "awaiting_resubmission"
    assert entry["note_age_days"] == 5
    assert entry["outstanding_at_drafting"] == 2
    assert {e["item"] for e in entry["still_outstanding_items"]} == {
        "application",
        "loss runs",
    }
    assert entry["answered_items"] == []
    assert entry["by_submission_id"] == ""


def test_digest_states_counts_and_oldest_outstanding_first():
    records = [
        _record("N-ANS", "Insured A", _note("2026-10-01T09:00:00+00:00", _follow("answered"))),
        _record("N-NEW-STILL", "Insured B", _note("2026-10-08T09:00:00+00:00", _follow("still_outstanding"))),
        _record("N-PART", "Insured C", _note("2026-10-03T09:00:00+00:00", _follow("partially_answered"))),
        _record("N-OLD-STILL", "Insured D", _note("2026-09-20T09:00:00+00:00", _follow("still_outstanding"))),
        _record("N-OPEN", "Insured E", _note("2026-09-25T09:00:00+00:00")),
        _record("N-NONE", "Insured F", None),
    ]
    digest = follow_through_digest(records, now=NOW)
    assert digest["total_notes"] == 5
    assert digest["counts"] == {
        "answered": 1,
        "partially_answered": 1,
        "still_outstanding": 2,
        "awaiting_resubmission": 1,
    }
    order = [e["submission_id"] for e in digest["entries"]]
    # Unresolved first, oldest note first; then partial; answered last.
    assert order == ["N-OLD-STILL", "N-OPEN", "N-NEW-STILL", "N-PART", "N-ANS"]
    partial = digest["entries"][3]
    assert [e["item"] for e in partial["answered_items"]] == ["application"]
    assert [e["item"] for e in partial["still_outstanding_items"]] == ["loss runs"]
    assert partial["by_submission_id"] == "NEW-1"
    answered = digest["entries"][4]
    assert answered["still_outstanding_items"] == []
    assert answered["checked_at"] == "2026-10-09T12:00:00+00:00"


def test_digest_maps_edge_states_honestly():
    legacy = _note(
        "2026-10-02T09:00:00+00:00",
        {"recorded": False, "state": "not_recorded", "answered": [],
         "still_outstanding": [], "by_submission_id": "NEW-9",
         "checked_at": "2026-10-09T12:00:00+00:00"},
    )
    del legacy["outstanding"]
    nothing = _note(
        "2026-10-04T09:00:00+00:00",
        {"recorded": True, "state": "nothing_outstanding", "answered": [],
         "still_outstanding": [], "by_submission_id": "NEW-8",
         "checked_at": "2026-10-09T12:00:00+00:00"},
        outstanding={"missing_elements": [], "gap_sources": [], "blocking_lines": []},
    )
    digest = follow_through_digest(
        [_record("N-LEG", "Insured A", legacy), _record("N-NOTH", "Insured B", nothing)],
        now=NOW,
    )
    states = {e["submission_id"]: e["state"] for e in digest["entries"]}
    # A pre-snapshot note was never checkable — it says so, and is
    # not quietly counted as answered. A note that recorded
    # nothing outstanding is answered: nothing was owed.
    assert states == {"N-LEG": "not_recorded", "N-NOTH": "answered"}


# ---------------------------------------------------------------------------
# Through the service: the inbox carries the digest end to end
# ---------------------------------------------------------------------------


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


def test_inbox_digest_closes_the_loop_end_to_end():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(_submission("FT-1", "2020-01-01"))
    service.draft_status_note("FT-1")

    inbox = service.inbox()
    digest = inbox["follow_through"]
    assert digest["total_notes"] == 1
    assert digest["entries"][0]["state"] == "awaiting_resubmission"
    assert digest["entries"][0]["submission_id"] == "FT-1"

    service.analyze(_submission("FT-2", "2026-10-09", **_COMPLETE))
    digest = service.inbox()["follow_through"]
    entry = digest["entries"][0]
    assert entry["state"] == "answered"
    assert entry["by_submission_id"] == "FT-2"
    assert entry["still_outstanding_items"] == []
    assert digest["counts"] == {"answered": 1}
