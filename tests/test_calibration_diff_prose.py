"""Round 17: movement prose for the calibration diff.

The calibration diff read as data — "rows became ready: 3,
5" — while the ops digest headline learned in round 12 to
speak membership in prose. The diff now carries a ``movement``
line in the same register: entered and cleared rows named by
tier and value, anchored on the older report by the handle
the debate uses (its label, or its id), with the budget
context that decides what the movement means — a row that
became ready as the budget held is evidence growing; one
that became ready as the budget moved is a different
argument, and the sentence says which.
"""

from __future__ import annotations

from underwriting_agent.proposals import (
    build_calibration_snapshot,
    calibration_snapshot_diff,
)


def _report(tiers: list[dict]) -> dict:
    return {
        "budgets": {t["tier"]: t["budget_in_force"] for t in tiers},
        "decided_cases_weighed": sum(t["decided_cases"] for t in tiers),
        "tiers": tiers,
    }


def _tier(tier: str, budget, ready: list[int], decided: int = 4) -> dict:
    return {
        "tier": tier,
        "budget_in_force": budget,
        "budget_source": "environment",
        "decided_cases": decided,
        "waits": {},
        "sweeps": [
            {"value": value, "proposal_ready": value in ready}
            for value in (2, 3, 4, 5, 7)
        ],
    }


def _snap(report: dict, sid: str, label: str = "") -> dict:
    return build_calibration_snapshot(report, sid, "Ops Lead", label=label)


def test_a_row_becoming_ready_speaks_with_the_budget_held():
    older = _snap(
        _report([_tier("high", 3, [])]), "cal-0001", label="pre-budget-change"
    )
    newer = _snap(
        _report([_tier("high", 3, [3])]), "cal-0002", label="post-budget-change"
    )
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since 'pre-budget-change': the high tier's 3-day row "
        "became proposal-ready as the budget held at 3 days."
    )


def test_rows_entering_and_clearing_share_one_budget_clause():
    older = _snap(_report([_tier("moderate", 5, [5])]), "cal-0001")
    newer = _snap(_report([_tier("moderate", 7, [3])]), "cal-0002")
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since cal-0001: the moderate tier's 3-day row became "
        "proposal-ready as its budget moved from 5 to 7 days; "
        "the moderate tier's 5-day row stopped being "
        "proposal-ready."
    )


def test_several_rows_entering_are_named_together():
    older = _snap(_report([_tier("low", 7, [])]), "cal-0001")
    newer = _snap(_report([_tier("low", 7, [2, 4])]), "cal-0002")
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since cal-0001: the low tier's 2-day and 4-day rows "
        "became proposal-ready as the budget held at 7 days."
    )


def test_a_budget_move_with_no_row_movement_still_speaks():
    older = _snap(_report([_tier("high", 3, [3])]), "cal-0001")
    newer = _snap(_report([_tier("high", 5, [3])]), "cal-0002")
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since cal-0001: the high tier's budget moved from 3 "
        "to 5 days."
    )


def test_no_movement_says_the_rows_held():
    report = _report([_tier("high", 3, [3]), _tier("low", 7, [])])
    older = _snap(report, "cal-0001", label="march")
    newer = _snap(report, "cal-0002", label="april")
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since 'march': no proposal-ready movement — every "
        "tier's ready rows held."
    )


def test_a_tier_absent_from_the_older_report_gets_no_budget_story():
    # Absence in a snapshot is a fact about the snapshot: the
    # tier's rows can enter, but there is no earlier budget to
    # hold or move, so the sentence claims neither.
    older = _snap(_report([]), "cal-0001")
    newer = _snap(_report([_tier("high", 3, [3])]), "cal-0002")
    diff = calibration_snapshot_diff(older, newer)
    assert diff["movement"] == (
        "Since cal-0001: the high tier's 3-day row became "
        "proposal-ready."
    )


def test_the_service_diff_carries_the_movement_line():
    from datetime import date, timedelta

    from underwriting_agent.schemas import AgentResult, Triage
    from underwriting_agent.service import SubmissionService
    from underwriting_agent.store import ApprovalRecord, InMemoryStore

    base = date(2026, 9, 1)
    store = InMemoryStore()
    store.save(
        ApprovalRecord(
            result=AgentResult(
                submission_id="M-2",
                status="approved",
                insured_name="Insured M-2",
                triage=Triage(outcome="borderline", risk_tier="moderate"),
                decided_at=(base + timedelta(days=2)).isoformat(),
            ),
            submission={
                "insured_name": "Insured M-2",
                "broker_name": "Broker Q",
                "submitted_at": base.isoformat(),
            },
        )
    )
    service = SubmissionService(store=store, checkpointer=False)
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service.snapshot_sla_calibration("Ops Lead", label="april")
    diff = service.sla_calibration_snapshot_diff()["diff"]
    assert diff["movement"].startswith("Since 'march':")
