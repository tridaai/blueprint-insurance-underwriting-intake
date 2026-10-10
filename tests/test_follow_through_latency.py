"""Round 18: follow-through response latency in the counts.

Round 17's worked counts say how many escalated notes an
underwriter touched; nothing priced how fast. A note escalates
one follow-through window (14 days) after it is drafted, and
the first touch is the response that escalation got — so each
touched note now carries ``response_days`` (first escalation
moment to first touch), and the digests aggregate it: the
follow-through digest's ``response_latency`` (notes measured,
median / mean / max days) and the ops digest's counted form
(``follow_through_responses``, riding the snapshot keys) with
the aggregate riding the digest body. Effort timing, like
effort, never enters ``total_attention`` and never reaches
the headline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import (
    build_digest_snapshot,
    follow_through_digest,
    ops_digest,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _note(drafted_at: str, touches: list[dict] | None = None) -> dict:
    note = {
        "submission_id": "OLD-1",
        "drafted_at": drafted_at,
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
            "by_submission_id": "NEW-0",
            "checked_at": "2026-10-09T12:00:00+00:00",
        },
    }
    if touches is not None:
        note["escalation_touches"] = touches
    return note


def _record(sid: str, note: dict) -> ApprovalRecord:
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


def _touch(at: str, by: str = "U. One") -> dict:
    return {"worked_by": by, "worked_at": at, "note": "chased the broker"}


# Drafted 2026-09-20 → escalated 2026-10-04 (14-day window);
# first touch 2026-10-06 → a 2-day response.
QUICK = _record(
    "LAT-2", _note("2026-09-20T09:00:00+00:00", [_touch("2026-10-06T09:00:00+00:00")])
)
# Drafted 2026-08-21 → escalated 2026-09-04; first touch
# 2026-09-14 → a 10-day response. The second touch (a
# re-quieting after re-escalation) does not re-answer it.
SLOW = _record(
    "LAT-10",
    _note(
        "2026-08-21T09:00:00+00:00",
        [
            _touch("2026-09-14T09:00:00+00:00"),
            _touch("2026-10-19T09:00:00+00:00", by="U. Two"),
        ],
    ),
)
# Drafted 2026-10-06 → escalated 2026-10-20 (today); touched
# the same day → a 0-day response.
SAME_DAY = _record(
    "LAT-0", _note("2026-10-06T09:00:00+00:00", [_touch("2026-10-20T08:00:00+00:00")])
)
# Escalated and never touched — no response to measure.
UNTOUCHED = _record("LAT-NONE", _note("2026-09-01T09:00:00+00:00"))


def _entry(digest: dict, sid: str) -> dict:
    return next(
        e for e in digest["entries"] if e["submission_id"] == sid
    )


# --- the per-note latency -------------------------------------------------------


def test_response_days_run_from_the_first_escalation_to_the_first_touch():
    digest = follow_through_digest([QUICK, SLOW, SAME_DAY], NOW)
    assert _entry(digest, "LAT-2")["response_days"] == 2
    assert _entry(digest, "LAT-10")["response_days"] == 10
    assert _entry(digest, "LAT-0")["response_days"] == 0
    assert _entry(digest, "LAT-2")["first_touch_at"] == (
        "2026-10-06T09:00:00+00:00"
    )
    # The slow note's second touch is its latest (the worked
    # read), but the response is the first touch's.
    slow = _entry(digest, "LAT-10")
    assert slow["worked"]["worked_at"] == "2026-10-19T09:00:00+00:00"
    assert slow["first_touch_at"] == "2026-09-14T09:00:00+00:00"


def test_an_untouched_note_has_no_response_to_measure():
    digest = follow_through_digest([UNTOUCHED], NOW)
    entry = _entry(digest, "LAT-NONE")
    assert entry["response_days"] is None
    assert entry["first_touch_at"] == ""
    assert digest["response_latency"] == {
        "notes_measured": 0,
        "median_days": None,
        "mean_days": None,
        "max_days": None,
        "p90_days": None,
    }


def test_a_touch_stamp_before_the_escalation_moment_clamps_to_zero():
    # A stored touch stamped inside the first window (the data
    # can say it, even though the service only records touches
    # on escalated notes): the response is zero days, never a
    # negative latency.
    record = _record(
        "LAT-EARLY",
        _note(
            "2026-10-01T09:00:00+00:00",
            [_touch("2026-10-05T09:00:00+00:00")],
        ),
    )
    digest = follow_through_digest([record], NOW)
    assert _entry(digest, "LAT-EARLY")["response_days"] == 0


# --- the aggregate ---------------------------------------------------------------


def test_the_aggregate_reports_median_mean_and_max():
    digest = follow_through_digest(
        [QUICK, SLOW, SAME_DAY, UNTOUCHED], NOW
    )
    assert digest["response_latency"] == {
        "notes_measured": 3,
        "median_days": 2.0,
        "mean_days": 4.0,
        "max_days": 10,
        "p90_days": 10,
    }


def test_an_even_count_median_averages_the_middle_pair():
    digest = follow_through_digest([QUICK, SLOW], NOW)
    assert digest["response_latency"]["median_days"] == 6.0
    assert digest["response_latency"]["notes_measured"] == 2


# --- the ops digest counts --------------------------------------------------------


def test_the_ops_digest_counts_responses_and_carries_the_aggregate():
    digest = ops_digest([QUICK, SLOW, SAME_DAY, UNTOUCHED], now=NOW)
    counts = digest["counts"]
    assert counts["follow_through_responses"] == 3
    assert digest["follow_through_response_latency"] == {
        "notes_measured": 3,
        "median_days": 2.0,
        "mean_days": 4.0,
        "max_days": 10,
        "p90_days": 10,
    }
    # Effort timing is not attention, exactly as effort is not.
    worklist_keys = (
        "stuck_dispatches",
        "exhausted_breach_notices",
        "acknowledged_still_waiting",
        "breached_queue",
        "follow_through_escalated",
        "proposals_expired",
    )
    assert counts["total_attention"] == sum(
        counts[key] for key in worklist_keys
    )
    # And the headline speaks the problem, never the pace.
    assert "response" not in digest["headline"]


def test_the_snapshot_keeps_the_response_count():
    digest = ops_digest([QUICK, SLOW], now=NOW)
    snap = build_digest_snapshot(digest, "snap-0001", "Ops Lead")
    assert snap["counts"]["follow_through_responses"] == 2
