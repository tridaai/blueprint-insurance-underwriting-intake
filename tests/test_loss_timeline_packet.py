"""Round 8: the loss-development timeline inside the packet.

Round 7 chained an insured's renewal diffs into a timeline —
behind a console drill-down. The underwriter decides in the
packet, so the arc belongs there: counts, cumulative incurred
movement, and the claims still open, composed in code from the
stored timeline plus this submission's own diff.
"""

from __future__ import annotations

from underwriting_agent.insights import loss_timeline_packet_lines
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

WATER = {"date": "2024-02-01", "amount_incurred": 5000, "description": "Water damage", "status": "closed"}
WATER_DEVELOPED = dict(WATER, amount_incurred=8000)
FIRE = {"date": "2025-06-01", "amount_incurred": 20000, "description": "Fire damage", "status": "open"}
FIRE_DEVELOPED = dict(FIRE, amount_incurred=26000)


def _submission(sid: str, losses: list[dict]) -> dict:
    return {
        "submission_id": sid,
        "insured_name": "Arc Co",
        "broker_name": "Broker Q",
        "line_of_business": "general",
        "class_of_business": "retail",
        "submitted_at": "2026-10-01",
        "documents": [
            {"doc_type": "acord_form", "document_id": "A1", "raw_text": "Application."}
        ],
        "losses": losses,
    }


def _service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def test_first_submission_packet_has_no_arc():
    service = _service()
    result = service.analyze(_submission("LT-A", [WATER]))
    assert "Loss history arc" not in result.drafts.underwriter_packet


def test_second_submission_packet_carries_the_arc_including_itself():
    service = _service()
    service.analyze(_submission("LT-A", [WATER]))
    result = service.analyze(_submission("LT-B", [WATER_DEVELOPED, FIRE]))
    packet = result.drafts.underwriter_packet
    assert "Loss history arc" in packet
    # Water developed +3,000 and fire appeared at 20,000: the
    # cumulative movement includes THIS submission's development.
    assert "cumulative net incurred movement across the chained renewals: +23,000" in packet
    assert "1 renewal(s) chained including this one; 2 claim(s) tracked" in packet
    # Only the open claim is listed as still open; the closed water
    # claim is tracked but not open.
    assert "still open on the latest run: Fire damage" in packet
    assert "$20,000 incurred, open" in packet


def test_third_submission_packet_shows_the_whole_arc():
    service = _service()
    service.analyze(_submission("LT-A", [WATER]))
    service.analyze(_submission("LT-B", [WATER_DEVELOPED, FIRE]))
    result = service.analyze(_submission("LT-C", [FIRE_DEVELOPED]))
    packet = result.drafts.underwriter_packet
    assert "2 renewal(s) chained including this one" in packet
    # +23,000 through LT-B; LT-C's own renewal nets -2,000 (fire
    # developed +6,000, but the dropped water claim takes 8,000 off
    # the run) — the cumulative is the chained sum, +21,000.
    assert "+21,000" in packet
    assert "no longer listed on the latest run: 1 claim(s)" in packet
    assert "$26,000 incurred, open" in packet


def test_bundle_line_packet_carries_the_arc():
    service = _service()
    service.analyze(_submission("LT-A", [WATER]))
    bundle = {
        "submission_id": "LT-BND",
        "insured_name": "Arc Co",
        "broker_name": "Broker Q",
        "submitted_at": "2026-10-02",
        "lines": [
            {
                "line_of_business": "general",
                "documents": [
                    {"doc_type": "acord_form", "document_id": "A1", "raw_text": "App."}
                ],
                "losses": [WATER_DEVELOPED, FIRE],
            }
        ],
    }
    result = service.analyze(bundle)
    assert result.multi_line
    assert "Loss history arc" in result.drafts.underwriter_packet


def test_packet_lines_pure_cases():
    assert loss_timeline_packet_lines(None) == []
    assert loss_timeline_packet_lines({}) == []
    # Stored history with submissions but no renewal and no claim
    # movement: no arc to tell.
    quiet = {
        "insured": "Quiet Co",
        "submissions": 1,
        "entries": [],
        "claims": [],
        "totals": {
            "renewals": 0,
            "appeared": 0,
            "developed": 0,
            "no_longer_listed": 0,
            "cumulative_net_incurred_change": 0.0,
            "claims_tracked": 0,
        },
    }
    assert loss_timeline_packet_lines(quiet) == []
    # A stored arc alone (no current diff) renders from the record.
    stored = {
        "insured": "Arc Co",
        "submissions": 2,
        "entries": [],
        "claims": [
            {
                "key": "2025-06-01|fire damage",
                "date": "2025-06-01",
                "description": "Fire damage",
                "state": "on_latest_run",
                "latest_incurred": 20000.0,
                "latest_status": "open",
            }
        ],
        "totals": {
            "renewals": 1,
            "appeared": 1,
            "developed": 0,
            "no_longer_listed": 0,
            "cumulative_net_incurred_change": 20000.0,
            "claims_tracked": 1,
        },
    }
    lines = loss_timeline_packet_lines(stored)
    assert any("+20,000" in line for line in lines)
    assert any("Fire damage" in line for line in lines)
