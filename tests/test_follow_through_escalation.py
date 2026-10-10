"""Round 10: follow-through aging escalation.

The round-9 digest listed still-outstanding status notes, but
nothing aged them: a note whose loop had been open for weeks sat
in the same list as one drafted yesterday. An open loop older
than FOLLOW_THROUGH_ESCALATION_DAYS (default 14) is now flagged
escalated — the note's own flag, not an SLA breach, since the
case may be decided while the note rots — and the digest lists
the escalated subset the ops digest works from.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import follow_through_digest
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _record(sid: str, note: dict | None) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={"insured_name": f"Insured {sid}", "broker_name": "Broker Q"},
        status_note=note,
    )


def _note(drafted_at: str, follow: dict | None = None) -> dict:
    note = {
        "submission_id": "X",
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


def _follow(state: str) -> dict:
    return {
        "recorded": True,
        "state": state,
        "answered": [],
        "still_outstanding": [{"line": "", "item": "loss runs"}],
        "by_submission_id": "NEW-1",
        "checked_at": "2026-10-09T12:00:00+00:00",
    }


def test_old_open_loop_escalates_young_one_does_not():
    records = [
        # Drafted 19 days before NOW, never reached by a resubmission.
        _record("OLD-1", _note("2026-10-01T09:00:00+00:00")),
        # Drafted 2 days before NOW — open, but not aged.
        _record("NEW-1", _note("2026-10-18T09:00:00+00:00")),
    ]
    digest = follow_through_digest(records, now=NOW)
    by_id = {e["submission_id"]: e for e in digest["entries"]}
    assert by_id["OLD-1"]["escalated"] is True
    assert by_id["NEW-1"]["escalated"] is False
    assert digest["escalation_days"] == 14
    assert digest["escalated_count"] == 1
    assert [e["submission_id"] for e in digest["escalated"]] == ["OLD-1"]


def test_checked_but_still_outstanding_note_escalates_too():
    records = [
        _record("OLD-2", _note("2026-09-20T09:00:00+00:00", _follow("still_outstanding"))),
        _record("OLD-3", _note("2026-09-20T09:00:00+00:00", _follow("partially_answered"))),
    ]
    digest = follow_through_digest(records, now=NOW)
    assert digest["escalated_count"] == 2


def test_answered_note_never_escalates_however_old():
    records = [
        _record("DONE-1", _note("2026-08-01T09:00:00+00:00", _follow("answered"))),
    ]
    digest = follow_through_digest(records, now=NOW)
    assert digest["entries"][0]["escalated"] is False
    assert digest["escalated_count"] == 0


def test_escalation_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("FOLLOW_THROUGH_ESCALATION_DAYS", "30")
    records = [_record("MID-1", _note("2026-10-01T09:00:00+00:00"))]
    digest = follow_through_digest(records, now=NOW)
    assert digest["escalation_days"] == 30
    assert digest["entries"][0]["escalated"] is False
    # An explicit argument beats the environment, as elsewhere.
    digest = follow_through_digest(records, now=NOW, escalation_days=10)
    assert digest["entries"][0]["escalated"] is True
