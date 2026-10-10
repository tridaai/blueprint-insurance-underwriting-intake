"""Round 16: a worked state for escalated follow-through notes.

The breach notices got an acknowledgement in round 13 because a
worked state must not become a quieter place to rot; the
escalated follow-through notes had none — an underwriter who
acted on an escalated note could not record it, so the note
escalated forever and the second touch was invisible. Marking a
note worked records the touch on the note itself (who, when,
their note) and quiets the escalation for one escalation
window: the digest, the ops digest, and the queue — all reading
the one follow-through computation — stop counting the note as
untouched, and the digest's ``worked`` subset keeps the touch
auditable. If the loop is still open when the window lapses,
the note escalates again and a further touch can be recorded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import (
    follow_through_digest,
    ops_digest,
    queue_aging,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _note(drafted_at: str, follow: dict | None = None, **extra) -> dict:
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
    note.update(extra)
    return note


def _record(
    sid: str,
    note: dict | None,
    insured: str = "Rotting Co",
    status: str = "awaiting_underwriter",
    submitted_at: str = "2026-10-15",
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


def _follow(state: str) -> dict:
    return {
        "recorded": True,
        "state": state,
        "answered": [] if state != "answered" else [{"line": "", "item": "loss runs"}],
        "still_outstanding": [] if state == "answered" else [{"line": "", "item": "loss runs"}],
        "by_submission_id": "NEW-0",
        "checked_at": "2026-10-09T12:00:00+00:00",
    }


def _touch(by: str, at: str, note: str = "") -> dict:
    return {"worked_by": by, "worked_at": at, "note": note}


def _service(*records: ApprovalRecord) -> SubmissionService:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return SubmissionService(store=store, checkpointer=False)


# --- the digest computation ------------------------------------------------


def test_a_fresh_touch_quiets_the_escalation():
    # Note drafted 30 days before NOW (well past the 14-day
    # window), touched 2 days before NOW.
    record = _record(
        "OLD-1",
        _note(
            "2026-09-20T09:00:00+00:00",
            escalation_touches=[_touch("Uma", "2026-10-18T09:00:00+00:00", "chased the broker")],
        ),
    )
    digest = follow_through_digest([record], now=NOW)
    (entry,) = digest["entries"]
    assert entry["escalated"] is False
    assert entry["quieted_by_work"] is True
    assert entry["worked"] == _touch("Uma", "2026-10-18T09:00:00+00:00", "chased the broker")
    assert entry["worked_touches"] == 1
    assert entry["worked_age_days"] == 2
    assert digest["escalated_count"] == 0
    assert digest["escalated"] == []
    assert digest["worked_count"] == 1
    assert [e["submission_id"] for e in digest["worked"]] == ["OLD-1"]


def test_a_touch_older_than_the_window_lets_the_note_escalate_again():
    # Touched 20 days before NOW: the quiet window (14) has
    # lapsed and the loop is still open, so the note is back on
    # the escalated surfaces — the worked state is not a pardon.
    record = _record(
        "OLD-1",
        _note(
            "2026-09-01T09:00:00+00:00",
            escalation_touches=[_touch("Uma", "2026-09-30T09:00:00+00:00")],
        ),
    )
    digest = follow_through_digest([record], now=NOW)
    (entry,) = digest["entries"]
    assert entry["escalated"] is True
    assert entry["quieted_by_work"] is False
    assert entry["worked_age_days"] == 20
    assert digest["escalated_count"] == 1
    assert digest["worked"] == []


def test_a_touch_exactly_a_window_old_no_longer_quiets():
    record = _record(
        "OLD-1",
        _note(
            "2026-09-01T09:00:00+00:00",
            escalation_touches=[_touch("Uma", "2026-10-06T09:00:00+00:00")],
        ),
    )
    digest = follow_through_digest([record], now=NOW)
    assert digest["entries"][0]["escalated"] is True


def test_an_untouched_note_reads_as_before():
    record = _record("OLD-1", _note("2026-09-20T09:00:00+00:00"))
    digest = follow_through_digest([record], now=NOW)
    (entry,) = digest["entries"]
    assert entry["escalated"] is True
    assert entry["worked"] is None
    assert entry["worked_touches"] == 0
    assert entry["worked_age_days"] is None
    assert entry["quieted_by_work"] is False
    assert digest["worked_count"] == 0
    assert digest["worked"] == []


def test_a_touch_on_an_answered_note_is_history_not_quieting():
    record = _record(
        "OLD-1",
        _note(
            "2026-09-01T09:00:00+00:00",
            _follow("answered"),
            escalation_touches=[_touch("Uma", "2026-10-18T09:00:00+00:00")],
        ),
    )
    digest = follow_through_digest([record], now=NOW)
    (entry,) = digest["entries"]
    assert entry["escalated"] is False
    assert entry["quieted_by_work"] is False
    # The touch stays auditable on the worked read even though
    # the loop closed on its own evidence.
    assert digest["worked_count"] == 1


# --- the service act --------------------------------------------------------


def test_mark_worked_records_the_touch_and_quiets_every_surface():
    record = _record("OLD-1", _note("2026-09-20T09:00:00+00:00"))
    service = _service(record)
    outcome = service.mark_escalated_note_worked(
        "OLD-1", "Uma", note="called the broker; loss runs promised", now=NOW
    )
    assert outcome["touch"] == {
        "worked_by": "Uma",
        "worked_at": "2026-10-20T00:00:00+00:00",
        "note": "called the broker; loss runs promised",
    }
    assert outcome["quiets_until"] == "2026-11-03"
    assert record.status_note["escalation_touches"] == [outcome["touch"]]
    # The note's own text and snapshot are untouched.
    assert record.status_note["text"] == "INTERNAL DRAFT …"
    digest = follow_through_digest([record], now=NOW)
    assert digest["escalated_count"] == 0
    assert digest["worked_count"] == 1
    # The ops digest reads the same computation: the escalated
    # count it headlines drops too.
    digest_view = ops_digest([record], now=NOW)
    assert digest_view["counts"]["follow_through_escalated"] == 0
    assert digest_view["follow_through_escalated"] == []
    # The ops digest's worked section is what the console renders.
    worked = digest_view["follow_through_worked"]
    assert [e["submission_id"] for e in worked] == ["OLD-1"]
    assert worked[0]["worked"]["worked_by"] == "Uma"


def test_mark_worked_refuses_a_second_touch_while_quieted():
    record = _record("OLD-1", _note("2026-09-20T09:00:00+00:00"))
    service = _service(record)
    service.mark_escalated_note_worked("OLD-1", "Uma", now=NOW)
    with pytest.raises(ValueError, match="already worked by Uma"):
        service.mark_escalated_note_worked("OLD-1", "Other", now=NOW)
    assert len(record.status_note["escalation_touches"]) == 1


def test_a_second_touch_lands_after_the_note_reescalates():
    record = _record("OLD-1", _note("2026-08-01T09:00:00+00:00"))
    service = _service(record)
    first_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    service.mark_escalated_note_worked("OLD-1", "Uma", now=first_at)
    # Twenty days later the window has lapsed, the loop is still
    # open, and the note has re-escalated: a further touch is a
    # new record, not a blur of the first.
    second_at = first_at + timedelta(days=20)
    outcome = service.mark_escalated_note_worked(
        "OLD-1", "Other", note="second chase", now=second_at
    )
    assert outcome["touch"]["worked_by"] == "Other"
    assert len(record.status_note["escalation_touches"]) == 2
    digest = follow_through_digest([record], now=second_at)
    assert digest["escalated_count"] == 0
    assert digest["entries"][0]["worked_touches"] == 2


def test_mark_worked_refuses_a_note_that_is_not_escalated():
    # Young open loop: inside its window, nothing to work yet.
    record = _record("NEW-1", _note("2026-10-18T09:00:00+00:00"))
    service = _service(record)
    with pytest.raises(ValueError, match="is not escalated"):
        service.mark_escalated_note_worked("NEW-1", "Uma", now=NOW)
    # Answered loop: closed on evidence, not on a touch.
    answered = _record(
        "DONE-1", _note("2026-09-01T09:00:00+00:00", _follow("answered"))
    )
    service = _service(answered)
    with pytest.raises(ValueError, match="is not escalated"):
        service.mark_escalated_note_worked("DONE-1", "Uma", now=NOW)


def test_mark_worked_unknown_and_noteless_records():
    service = _service(_record("BARE-1", None))
    with pytest.raises(KeyError):
        service.mark_escalated_note_worked("NOPE", "Uma", now=NOW)
    with pytest.raises(ValueError, match="No status note"):
        service.mark_escalated_note_worked("BARE-1", "Uma", now=NOW)


def test_the_queue_stops_flagging_the_insured_once_the_note_is_worked():
    old = _record(
        "OLD-1",
        _note("2026-09-25T10:00:00+00:00"),  # 25 days old at NOW
        status="approved",
        submitted_at="2026-08-01",
    )
    live = _record("NEW-1", None)
    service = _service(old, live)
    before = queue_aging([old, live], NOW)
    assert before["items_with_escalated_notes"] == 1
    service.mark_escalated_note_worked("OLD-1", "Uma", now=NOW)
    after = queue_aging([old, live], NOW)
    assert after["items_with_escalated_notes"] == 0
    assert after["escalated_notes"] == []
    (item,) = [
        item
        for bucket in after["buckets"]
        for item in bucket["submissions"]
    ]
    assert item["escalated_note"] is None


# --- the API -----------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    store.save(_record("OLD-1", _note("2026-08-20T09:00:00+00:00")))
    store.save(_record("NEW-1", _note("2026-08-20T09:00:00+00:00")))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    return TestClient(api_module.app)


def test_api_marks_worked_and_the_digest_reads_it(client):
    # The endpoint stamps the touch with the real clock; age the
    # fixture notes well past the window so they escalate today.
    res = client.post(
        "/submissions/OLD-1/status-note/worked",
        json={"actor": "Uma", "note": "chased the broker"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["touch"]["worked_by"] == "Uma"
    assert body["touch"]["note"] == "chased the broker"
    assert len(body["touches"]) == 1

    inbox = client.get("/inbox").json()
    follow = inbox["follow_through"]
    assert follow["worked_count"] == 1
    assert follow["escalated_count"] == 1  # NEW-1 still escalates
    assert [e["submission_id"] for e in follow["worked"]] == ["OLD-1"]

    # While the touch quiets it, a second mark is refused.
    again = client.post(
        "/submissions/OLD-1/status-note/worked", json={"actor": "Other"}
    )
    assert again.status_code == 422
    assert "already worked by Uma" in again.json()["detail"]


def test_api_mark_worked_validates_the_case_and_the_actor(client):
    unknown = client.post(
        "/submissions/NOPE/status-note/worked", json={"actor": "Uma"}
    )
    assert unknown.status_code == 404
    anonymous = client.post("/submissions/OLD-1/status-note/worked", json={})
    assert anonymous.status_code == 422
