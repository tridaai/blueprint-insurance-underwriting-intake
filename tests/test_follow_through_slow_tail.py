"""Round 21: the slow tail of the follow-through response.

Round 18 priced the response (median / mean / max), but a
median can look healthy while a handful of notes waited
weeks — and the max was one number with no names behind it.
The aggregate now reads its tail: ``p90_days`` (nearest-rank)
beside the median, and ``slowest_responses`` names the
slowest measured notes, slowest first, capped at five, each
with the author of its first touch — the tail as a worklist,
not a statistic. The ops digest carries the named tail beside
the aggregate, on the same body-not-counts discipline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.insights import (
    follow_through_digest,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 12, 31, tzinfo=timezone.utc)
DRAFTED = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def _record(sid: str, response_days: int | None) -> ApprovalRecord:
    """A note drafted 2026-08-01 (escalating 2026-08-15, one
    14-day window later); when ``response_days`` is given, its
    first touch lands that many days after the escalation."""
    note = {
        "submission_id": sid,
        "drafted_at": DRAFTED.isoformat(timespec="seconds"),
        "text": "INTERNAL DRAFT …",
        "outstanding": {
            "missing_elements": ["loss runs"],
            "gap_sources": [],
            "blocking_lines": [],
        },
        "follow_through": {
            "recorded": True,
            "state": "still_outstanding",
            "answered": [],
            "still_outstanding": [{"line": "", "item": "loss runs"}],
            "by_submission_id": "",
            "checked_at": "2026-10-09T12:00:00+00:00",
        },
    }
    if response_days is not None:
        first = DRAFTED + timedelta(days=14 + response_days)
        note["escalation_touches"] = [
            {
                "worked_by": f"First {sid}",
                "worked_at": first.isoformat(timespec="seconds"),
                "note": "chased the broker",
            },
            {
                # A later re-quieting touch by someone else:
                # the response is still the first touch's.
                "worked_by": "Someone Later",
                "worked_at": (first + timedelta(days=30)).isoformat(
                    timespec="seconds"
                ),
                "note": "re-quieted",
            },
        ]
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": "2026-08-01",
        },
        status_note=note,
    )


def _records(*responses: int | None) -> list[ApprovalRecord]:
    return [
        _record(f"TAIL-{i + 1:02d}", days)
        for i, days in enumerate(responses)
    ]


# --- the p90 --------------------------------------------------------------------


def test_p90_reads_the_tail_beside_the_median():
    digest = follow_through_digest(
        _records(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), NOW
    )
    latency = digest["response_latency"]
    assert latency["notes_measured"] == 10
    assert latency["median_days"] == 5.5
    # Nearest rank: nine tenths of ten responses sit at or
    # below the 9th value.
    assert latency["p90_days"] == 9
    assert latency["max_days"] == 10


def test_p90_with_one_measured_response_is_that_response():
    digest = follow_through_digest(_records(4, None), NOW)
    assert digest["response_latency"]["p90_days"] == 4


def test_p90_is_none_when_nothing_was_measured():
    digest = follow_through_digest(_records(None, None), NOW)
    assert digest["response_latency"]["p90_days"] is None
    assert digest["slowest_responses"] == []


# --- the named tail ---------------------------------------------------------------


def test_the_slowest_responses_are_named_slowest_first():
    digest = follow_through_digest(
        _records(3, 40, 12, 40, 1, None), NOW
    )
    slowest = digest["slowest_responses"]
    assert [s["response_days"] for s in slowest] == [40, 40, 12, 3, 1]
    # Ties walk by submission id, so the read is stable.
    assert [s["submission_id"] for s in slowest[:2]] == [
        "TAIL-02",
        "TAIL-04",
    ]
    first = slowest[0]
    assert first["insured_name"] == "Insured TAIL-02"
    assert first["broker_name"] == "Broker Q"
    # The first touch's author — not the later re-quieting one.
    assert first["first_touch_by"] == "First TAIL-02"
    assert first["first_touch_at"].startswith("2026-09-24")


def test_the_tail_is_capped_at_five():
    digest = follow_through_digest(
        _records(1, 2, 3, 4, 5, 6, 7, 8), NOW
    )
    slowest = digest["slowest_responses"]
    assert len(slowest) == 5
    assert [s["response_days"] for s in slowest] == [8, 7, 6, 5, 4]


def test_untouched_notes_never_enter_the_tail():
    digest = follow_through_digest(_records(None, 2), NOW)
    assert [s["submission_id"] for s in digest["slowest_responses"]] == [
        "TAIL-02"
    ]


# --- the ops digest carries the tail ------------------------------------------------


def test_the_ops_digest_carries_the_named_tail_beside_the_aggregate():
    records = _records(3, 40, 12)
    follow = follow_through_digest(records, NOW)
    digest = ops_digest(records, now=NOW)
    assert digest["follow_through_slowest_responses"] == follow[
        "slowest_responses"
    ]
    assert digest["follow_through_response_latency"]["p90_days"] == 40
    # The tail is still not attention, and the headline still
    # never speaks the pace.
    assert "slow" not in digest["headline"]
