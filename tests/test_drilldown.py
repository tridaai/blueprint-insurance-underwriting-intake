"""Scorecard drill-down: the counts open into the submissions
behind them, and the approval rate gains a direction.

An inbox group row used to be a dead end — counts and a latest
line. The group now carries its submissions (store order, with
statuses), and each scorecard carries its approval trend: the
approval rate over the earlier half of the group's decisions
against the recent half, computed from decision order. Fewer than
two decisions cannot have a direction, and the trend says so
instead of inventing one.
"""

from __future__ import annotations

import pytest

from underwriting_agent.insights import _approval_trend
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def minimal(sid: str, broker: str, insured: str | None = None) -> dict:
    return {
        "submission_id": sid,
        "insured_name": insured or f"Insured {sid}",
        "broker_name": broker,
        "line_of_business": "cyber",
        "class_of_business": "professional services",
        "submitted_at": "2026-10-01",
        "documents": [
            {
                "document_id": f"{sid}-app",
                "doc_type": "application",
                "fields": {"insured_name": insured or f"Insured {sid}"},
                "raw_text": "Application on file.",
            }
        ],
        "losses": [],
    }


def run_decisions(service: SubmissionService, broker: str, decisions: list[str]):
    for index, decision in enumerate(decisions):
        sid = f"{broker.replace(' ', '')}-{index}"
        service.analyze(minimal(sid, broker))
        if decision == "approve":
            service.approve(sid, approver="U")
        elif decision == "reject":
            service.reject(sid, reviewer="U", reason="not for us")


# ---------------------------------------------------------------------------
# The trend computation
# ---------------------------------------------------------------------------


def entry(status: str) -> dict:
    return {"status": status}


def test_trend_declining_when_recent_decisions_go_bad():
    trend = _approval_trend(
        [entry("approved"), entry("approved"), entry("rejected"), entry("rejected")]
    )
    assert trend["direction"] == "declining"
    assert trend["early_rate"] == 1.0 and trend["recent_rate"] == 0.0
    assert trend["delta"] == -1.0


def test_trend_improving_and_flat():
    improving = _approval_trend(
        [entry("rejected"), entry("rejected"), entry("approved"), entry("approved")]
    )
    assert improving["direction"] == "improving" and improving["delta"] == 1.0
    flat = _approval_trend(
        [entry("approved"), entry("rejected"), entry("approved"), entry("rejected")]
    )
    assert flat["direction"] == "flat" and flat["delta"] == 0.0


def test_trend_refuses_a_direction_without_two_decisions():
    assert _approval_trend([])["direction"] == "insufficient_history"
    single = _approval_trend([entry("approved")])
    assert single["direction"] == "single_decision"
    assert single["delta"] is None


# ---------------------------------------------------------------------------
# Drill-down through the service
# ---------------------------------------------------------------------------


def test_inbox_group_carries_its_submissions_with_statuses():
    service = make_service()
    run_decisions(service, "Drill Brokers", ["approve", "leave", "reject"])
    inbox = service.inbox()
    group = next(g for g in inbox["brokers"] if g["name"] == "Drill Brokers")
    subs = group["submissions"]
    assert [s["submission_id"] for s in subs] == [
        "DrillBrokers-0",
        "DrillBrokers-1",
        "DrillBrokers-2",
    ]
    assert [s["status"] for s in subs] == [
        "approved",
        "awaiting_underwriter",
        "rejected",
    ]
    assert all(s["outcome"] and s["risk_tier"] for s in subs)
    # The counts the row always showed still agree with the list.
    assert group["submission_count"] == len(subs) == 3
    assert group["decided_count"] == 2 and group["awaiting_count"] == 1


def test_scorecard_carries_the_trend_from_decision_order():
    service = make_service()
    run_decisions(service, "Falling Brokers", ["approve", "approve", "reject", "reject"])
    run_decisions(service, "Rising Brokers", ["reject", "reject", "approve", "approve"])
    falling = service.scorecard("broker", "Falling Brokers")
    rising = service.scorecard("broker", "Rising Brokers")
    assert falling["approval_trend"]["direction"] == "declining"
    assert falling["approval_rate"] == 0.5  # the flat average hides it; the trend doesn't
    assert rising["approval_trend"]["direction"] == "improving"
    assert rising["approval_trend"]["decisions"] == 4


def test_scorecard_trend_with_no_decisions_is_honest():
    service = make_service()
    service.analyze(minimal("SOLO-1", "Quiet Brokers"))
    card = service.scorecard("broker", "Quiet Brokers")
    assert card["approval_trend"]["direction"] == "insufficient_history"


def test_drilldown_reaches_the_api_payloads(client):
    from fastapi.testclient import TestClient  # noqa: F401  (fixture builds it)

    service = make_service()
    run_decisions(service, "Api Brokers", ["approve", "reject"])

    import underwriting_agent.api as api_module

    api_module.service = service
    inbox = client.get("/inbox").json()
    group = next(g for g in inbox["brokers"] if g["name"] == "Api Brokers")
    assert len(group["submissions"]) == 2
    cards = client.get("/scorecards").json()
    card = next(c for c in cards["brokers"] if c["name"] == "Api Brokers")
    # approve then reject: the recent half (0.0) trails the earlier (1.0).
    assert card["approval_trend"]["direction"] == "declining"
    assert card["approval_trend"]["delta"] == -1.0
