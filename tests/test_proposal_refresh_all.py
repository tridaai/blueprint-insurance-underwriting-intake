"""Round 10: the refresh-all sweep for expired drafts.

Round 9 made an expired draft refreshable — one at a time. An
operator returning after a quiet fortnight faced them one at a
time. The sweep refreshes every expired draft in one call, but
it is a loop, not a shortcut: each draft's evidence scan re-runs
individually, each refresh lands on its own proposal's audit
list, and renewals and withdrawals are reported side by side.
Nothing is bulk-decided.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from underwriting_agent.proposals import (
    draft_sla_proposals,
    load_proposals,
    refresh_expired_proposals,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
LATER = T0 + timedelta(days=20)  # past the 14-day evidence window


def _sla_record(
    sid: str,
    submitted_at: str,
    decided_at: str | None,
    tier: str = "moderate",
    status: str = "approved",
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status=status,
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            decided_at=decided_at,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _two_tier_store() -> InMemoryStore:
    """Two one-sided SLA moves in two tiers: a moderate case that
    breached the 5-day budget and was approved anyway (raise), and
    a low case decided well inside the 7-day budget (the sweep
    only needs the moderate one to be draftable — see the test)."""
    store = InMemoryStore()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    return store


def test_sweep_with_nothing_expired_is_an_empty_report():
    store = _two_tier_store()
    draft_sla_proposals(store, store.all_records(), now=T0)
    report = refresh_expired_proposals(
        store, store.all_records(), "U. Writer", now=T0 + timedelta(days=1)
    )
    assert report["counts"] == {"swept": 0, "renewed": 0, "withdrawn": 0}
    assert report["renewed"] == [] and report["withdrawn"] == []


def test_sweep_renews_an_expired_draft_whose_evidence_still_holds():
    store = _two_tier_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    report = refresh_expired_proposals(store, store.all_records(), "U. Writer", now=LATER)
    assert report["counts"] == {"swept": 1, "renewed": 1, "withdrawn": 0}
    renewed = report["renewed"][0]
    assert renewed["proposal_id"] == proposal["proposal_id"]
    assert renewed["status"] == "draft"
    assert renewed["expired"] is False
    assert renewed["refreshes"][0]["outcome"] == "renewed"
    assert renewed["refreshes"][0]["by"] == "U. Writer"
    # The stored proposal carries the same audit entry — the sweep
    # wrote through the per-proposal path, not around it.
    stored = load_proposals(store)[0]
    assert stored["refreshes"][0]["outcome"] == "renewed"


def test_sweep_withdraws_a_draft_whose_evidence_went_two_sided():
    store = _two_tier_store()
    proposal = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    store.save(
        _sla_record("M-BAD", "2026-09-24", "2026-10-03T10:00:00+00:00", status="rejected")
    )
    report = refresh_expired_proposals(store, store.all_records(), "U. Writer", now=LATER)
    assert report["counts"] == {"swept": 1, "renewed": 0, "withdrawn": 1}
    withdrawn = report["withdrawn"][0]
    assert withdrawn["proposal_id"] == proposal["proposal_id"]
    assert withdrawn["status"] == "withdrawn"
    assert "no longer holds" in withdrawn["withdrawal_reason"]


def test_sweep_leaves_live_drafts_and_history_untouched():
    store = _two_tier_store()
    old = draft_sla_proposals(store, store.all_records(), now=T0)["drafted"][0]
    # A second draft created later is still live at sweep time.
    store.save(
        _sla_record("H-OK", "2026-09-20", "2026-10-05T10:00:00+00:00", tier="high")
    )
    live = draft_sla_proposals(store, store.all_records(), now=LATER - timedelta(days=1))[
        "drafted"
    ]
    report = refresh_expired_proposals(store, store.all_records(), "U. Writer", now=LATER)
    swept_ids = {p["proposal_id"] for p in report["renewed"] + report["withdrawn"]}
    assert old["proposal_id"] in swept_ids
    for proposal in live:
        assert proposal["proposal_id"] not in swept_ids
    stored = {p["proposal_id"]: p for p in load_proposals(store)}
    for proposal in live:
        assert "refreshes" not in stored[proposal["proposal_id"]]


def test_service_sweep_reports_and_decides_nothing():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    store = service._get_store()
    store.save(_sla_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    service.draft_sla_budget_proposals()
    # Nothing is expired yet at wall-clock now, so the sweep is a
    # no-op report — and crucially it decided nothing either way.
    report = service.refresh_expired_calibration_proposals("U. Writer")
    assert report["counts"]["swept"] == 0
    assert all(p["status"] == "draft" for p in service.calibration_proposals())
