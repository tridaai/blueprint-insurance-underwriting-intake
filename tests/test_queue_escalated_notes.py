"""Round 11: escalated follow-through notes in the queue worklist.

Round 10 aged status-note follow-through: an open loop past the
escalation window is flagged in the follow-through digest and the
ops digest. But the queue — the worklist an underwriter actually
opens — showed neither the flag nor the note's age, and the note
lives on its own (often decided) record while the insured's live
work sits on a newer one. The queue now matches escalated notes
to queue items by insured: the item carries the flag and the
note's age, and every escalated note is listed with the awaiting
submissions its insured currently has.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import queue_aging
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _record(
    sid: str,
    insured: str,
    status: str = "awaiting_underwriter",
    submitted_at: str = "2026-10-15",
    note: dict | None = None,
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=insured,
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": insured,
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
        status_note=note,
    )


def _note(drafted_at: str, follow: dict | None = None) -> dict:
    note = {
        "submission_id": "OLD-1",
        "drafted_at": drafted_at,
        "text": "INTERNAL DRAFT …",
        "outstanding": {
            "missing_elements": ["loss runs"],
            "gap_sources": [],
            "blocking_lines": [],
        },
    }
    if follow is not None:
        note["follow_through"] = follow
    return note


def _items(view: dict) -> list[dict]:
    return [
        item
        for bucket in view["buckets"]
        for item in bucket["submissions"]
    ]


def test_queue_item_carries_the_escalated_note_of_its_insured():
    # The note rots on the OLD, decided record; the insured's new
    # submission is the one sitting in the queue.
    old = _record(
        "OLD-1", "Rotting Co", status="approved",
        submitted_at="2026-08-01",
        note=_note("2026-09-25T10:00:00+00:00"),  # 25 days old at NOW
    )
    live = _record("NEW-1", "Rotting Co")
    view = queue_aging([old, live], NOW)
    (item,) = _items(view)
    assert item["submission_id"] == "NEW-1"
    assert item["escalated_note"] == {
        "submission_id": "OLD-1",
        "drafted_at": "2026-09-25T10:00:00+00:00",
        "note_age_days": 25,
        "state": "awaiting_resubmission",
    }
    assert view["items_with_escalated_notes"] == 1
    (entry,) = view["escalated_notes"]
    assert entry["submission_id"] == "OLD-1"
    assert entry["insured_name"] == "Rotting Co"
    assert entry["note_age_days"] == 25
    assert entry["awaiting_submission_ids"] == ["NEW-1"]


def test_a_young_open_note_does_not_flag_the_queue_item():
    old = _record(
        "OLD-1", "Fresh Co", status="approved",
        submitted_at="2026-08-01",
        note=_note("2026-10-15T10:00:00+00:00"),  # 5 days: open, not aged
    )
    live = _record("NEW-1", "Fresh Co")
    view = queue_aging([old, live], NOW)
    (item,) = _items(view)
    assert item["escalated_note"] is None
    assert view["escalated_notes"] == []
    assert view["items_with_escalated_notes"] == 0


def test_an_answered_note_never_escalates():
    follow = {
        "recorded": True,
        "state": "answered",
        "answered": [{"line": "", "item": "loss runs"}],
        "still_outstanding": [],
        "by_submission_id": "NEW-0",
        "checked_at": "2026-10-09T12:00:00+00:00",
    }
    old = _record(
        "OLD-1", "Closed Co", status="approved",
        submitted_at="2026-08-01",
        note=_note("2026-09-01T10:00:00+00:00", follow),
    )
    live = _record("NEW-1", "Closed Co")
    view = queue_aging([old, live], NOW)
    (item,) = _items(view)
    assert item["escalated_note"] is None
    assert view["escalated_notes"] == []


def test_escalated_note_without_awaiting_work_is_still_listed():
    # The case was decided; the loop rots with nothing in the
    # queue. The section lists it with no awaiting submissions —
    # the underwriter sees the loop only they can close.
    old = _record(
        "OLD-1", "Decided Co", status="rejected",
        submitted_at="2026-08-01",
        note=_note("2026-09-20T10:00:00+00:00"),  # 30 days old
    )
    view = queue_aging([old], NOW)
    assert view["total_awaiting"] == 0
    (entry,) = view["escalated_notes"]
    assert entry["insured_name"] == "Decided Co"
    assert entry["note_age_days"] == 30
    assert entry["awaiting_submission_ids"] == []
    assert view["items_with_escalated_notes"] == 0


def test_the_oldest_rotting_loop_wins_the_flag():
    older = _record(
        "OLD-1", "Twice Co", status="approved",
        submitted_at="2026-07-01",
        note=_note("2026-09-01T10:00:00+00:00"),  # 49 days
    )
    newer = _record(
        "OLD-2", "Twice Co", status="approved",
        submitted_at="2026-08-01",
        note={**_note("2026-09-30T10:00:00+00:00"), "submission_id": "OLD-2"},
    )
    live = _record("NEW-1", "Twice Co")
    view = queue_aging([newer, older, live], NOW)
    (item,) = _items(view)
    assert item["escalated_note"]["submission_id"] == "OLD-1"
    assert item["escalated_note"]["note_age_days"] == 49
    assert len(view["escalated_notes"]) == 2


def test_an_unrelated_insureds_note_flags_nothing():
    old = _record(
        "OLD-1", "Other Co", status="approved",
        submitted_at="2026-08-01",
        note=_note("2026-09-01T10:00:00+00:00"),
    )
    live = _record("NEW-1", "Mine Co")
    view = queue_aging([old, live], NOW)
    (item,) = _items(view)
    assert item["escalated_note"] is None
    (entry,) = view["escalated_notes"]
    assert entry["awaiting_submission_ids"] == []
