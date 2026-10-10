"""Round 7: queue SLA state.

Every awaiting item carries an SLA state against a budget set by
its risk tier (elevated 2 days, high 3, moderate 5, low 7 — the
budget shrinks as the tier rises, because the harder cases need a
senior underwriter soonest). Breaches are listed worst-first as
the queue's worklist.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import queue_aging, sla_budgets
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.store import ApprovalRecord

NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _record(
    sid: str,
    submitted_at: str,
    tier: str = "moderate",
    status: str = "awaiting_underwriter",
    **result_kwargs,
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            **result_kwargs,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _items(view: dict) -> dict:
    return {
        item["submission_id"]: item
        for bucket in view["buckets"]
        for item in bucket["submissions"]
    }


def test_sla_budgets_default_by_tier():
    assert sla_budgets() == {"elevated": 2, "high": 3, "moderate": 5, "low": 7}


def test_sla_budgets_env_override(monkeypatch):
    monkeypatch.setenv("QUEUE_SLA_DAYS_HIGH", "9")
    assert sla_budgets()["high"] == 9
    assert sla_budgets()["low"] == 7


def test_queue_items_carry_sla_state_by_tier():
    records = [
        _record("E1", "2026-10-07", tier="elevated"),  # 3 days > 2 budget
        _record("H1", "2026-10-07", tier="high"),      # 3 days = 3 budget: within
        _record("M1", "2026-10-01", tier="moderate"),  # 9 days > 5 budget
        _record("L1", "2026-10-05", tier="low"),       # 5 days < 7 budget
    ]
    view = queue_aging(records, now=NOW)
    items = _items(view)
    assert items["E1"]["sla_budget_days"] == 2
    assert items["E1"]["sla_state"] == "breached"
    assert items["E1"]["sla_days_over"] == 1
    assert items["H1"]["sla_state"] == "within"
    assert items["H1"]["sla_days_over"] == 0
    assert items["M1"]["sla_state"] == "breached"
    assert items["M1"]["sla_days_over"] == 4
    assert items["L1"]["sla_state"] == "within"
    assert view["sla_budgets"] == {"elevated": 2, "high": 3, "moderate": 5, "low": 7}
    # Breaches: worst overrun first.
    assert [b["submission_id"] for b in view["breaches"]] == ["M1", "E1"]
    assert view["breach_count"] == 2


def test_queue_sla_unknown_age_and_decided_records():
    records = [
        _record("U1", "not-a-date", tier="elevated"),
        _record("D1", "2026-09-01", tier="elevated", status="approved"),
    ]
    view = queue_aging(records, now=NOW)
    assert view["total_awaiting"] == 1
    assert view["unknown_age"] == ["U1"]
    assert view["breaches"] == []
    assert view["breach_count"] == 0


def test_queue_sla_custom_budgets():
    records = [_record("M1", "2026-10-08", tier="moderate")]  # 2 days old
    budgets = {"elevated": 1, "high": 1, "moderate": 1, "low": 1}
    view = queue_aging(records, now=NOW, sla_budgets_map=budgets)
    assert _items(view)["M1"]["sla_state"] == "breached"
    assert _items(view)["M1"]["sla_budget_days"] == 1
