"""Round 21: a pruned direction, named when re-asked.

The drafting scans never re-ask a rejected direction — while
they remember it. The proposals retention (round 20) governs
the forgetting: a prune may remove a rejected proposal from
the decided history, and from that moment the scan may
legitimately ask the direction again. Round 20 named the risk
and left it as this round's work: nothing distinguished that
re-ask from a first ask. Now the prune record keeps the
pruned rejections whole (kind, dimension, direction, and the
decision — who refused it, when), and a draft whose direction
a prune forgot carries the decision on its face:
``previously_decided`` on the proposal and a rationale
sentence naming the earlier refusal and the prune that
removed its record. The re-ask is honest about its history
at the moment it matters — when a human reads the draft.
"""

from __future__ import annotations

from underwriting_agent.proposals import (
    decide_proposal,
    draft_sla_proposal_from_row,
    draft_sla_proposals,
    load_proposals,
    pruned_rejection_for,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore


def _record(
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


def _moderate_record() -> ApprovalRecord:
    # Moderate budget is 5 days; an 8-day wait, approved anyway
    # — the scan's raise evidence (see test_sla_budget_proposals).
    return _record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00")


def _high_records() -> list[ApprovalRecord]:
    return [
        _record("H-1", "2026-09-28", "2026-09-30T09:00:00+00:00",
                tier="high", status="rejected"),
        _record("H-2", "2026-09-27", "2026-09-30T09:00:00+00:00",
                tier="high", status="rejected"),
    ]


def _store_with(records: list[ApprovalRecord]) -> InMemoryStore:
    store = InMemoryStore()
    for record in records:
        store.save(record)
    return store


def _by_dimension(proposals: list[dict], dimension: str) -> dict:
    return next(p for p in proposals if p["dimension"] == dimension)


def _rejected_then_accepted_store() -> tuple[InMemoryStore, dict]:
    """A store whose decided history is, in document order: a
    rejected moderate raise (PROP-0001), then an accepted high
    lower (PROP-0002). Returns the store and the rejected
    proposal as it stood before any prune."""
    store = _store_with([_moderate_record()])
    first = draft_sla_proposals(store, store.all_records())
    assert len(first["drafted"]) == 1
    decide_proposal(store, "PROP-0001", "U. Writer", accept=False)
    rejected = _by_dimension(load_proposals(store), "moderate")
    assert rejected["status"] == "rejected"
    assert rejected["decided_by"] == "U. Writer"
    for record in _high_records():
        store.save(record)
    second = draft_sla_proposals(store, store.all_records())
    assert [p["dimension"] for p in second["drafted"]] == ["high"]
    decide_proposal(store, "PROP-0002", "U. Writer", accept=True)
    return store, rejected


# --- the remembered refusal still binds -----------------------------------------


def test_a_remembered_rejection_is_not_reasked_and_carries_no_marker():
    store, _ = _rejected_then_accepted_store()
    out = draft_sla_proposals(store, store.all_records())
    assert out["drafted"] == []
    moderate = _by_dimension(out["proposals"], "moderate")
    assert moderate["status"] == "rejected"
    assert "previously_decided" not in moderate


def test_a_first_ask_carries_no_marker():
    store = _store_with([_moderate_record()])
    out = draft_sla_proposals(store, store.all_records())
    assert "previously_decided" not in out["drafted"][0]
    assert "asked before" not in out["drafted"][0]["rationale"]


# --- the prune record keeps the rejection whole ---------------------------------


def test_the_prune_record_keeps_the_pruned_rejection_whole():
    store, rejected = _rejected_then_accepted_store()
    service = SubmissionService(store=store, checkpointer=False)
    report = service.prune_calibration_proposals("Uma Pruner", 1)
    assert report["pruned_ids"] == ["PROP-0001"]
    entry = report["prune"]
    assert entry["pruned_rejections"] == [
        {
            "proposal_id": "PROP-0001",
            "kind": "sla_budget",
            "dimension": "moderate",
            "direction": "raise",
            "decided_at": rejected["decided_at"],
            "decided_by": "U. Writer",
        }
    ]
    # An accepted proposal leaves no rejection behind it.
    store2, _ = _rejected_then_accepted_store()
    service2 = SubmissionService(store=store2, checkpointer=False)
    report2 = service2.prune_calibration_proposals("Uma Pruner", 1)
    assert [
        r["proposal_id"] for r in report2["prune"]["pruned_rejections"]
    ] == ["PROP-0001"]


def test_pruned_rejection_for_reads_the_latest_prune():
    prunes = [
        {
            "pruned_at": "2026-10-01T09:00:00+00:00",
            "pruned_by": "Uma",
            "pruned_rejections": [
                {
                    "proposal_id": "PROP-0001",
                    "kind": "diff_trigger_threshold",
                    "dimension": "limit_jump_pct",
                    "direction": "lower",
                    "decided_at": "2026-09-20T09:00:00+00:00",
                    "decided_by": "First Refuser",
                }
            ],
        },
        {
            "pruned_at": "2026-10-08T09:00:00+00:00",
            "pruned_by": "Uma",
            "pruned_rejections": [
                {
                    "proposal_id": "PROP-0009",
                    "kind": "diff_trigger_threshold",
                    "dimension": "limit_jump_pct",
                    "direction": "lower",
                    "decided_at": "2026-10-05T09:00:00+00:00",
                    "decided_by": "Second Refuser",
                }
            ],
        },
    ]
    found = pruned_rejection_for(
        prunes, "diff_trigger_threshold", "limit_jump_pct", "lower"
    )
    assert found["proposal_id"] == "PROP-0009"
    assert found["decided_by"] == "Second Refuser"
    assert found["pruned_at"] == "2026-10-08T09:00:00+00:00"
    # A different direction, kind, or dimension was never refused.
    assert pruned_rejection_for(
        prunes, "diff_trigger_threshold", "limit_jump_pct", "raise"
    ) is None
    assert pruned_rejection_for(
        prunes, "sla_budget", "limit_jump_pct", "lower"
    ) is None
    assert pruned_rejection_for([], "sla_budget", "high", "lower") is None


# --- the re-ask names the pruned decision ----------------------------------------


def test_after_the_prune_the_scan_reasks_and_names_the_decision():
    store, rejected = _rejected_then_accepted_store()
    service = SubmissionService(store=store, checkpointer=False)
    service.prune_calibration_proposals("Uma Pruner", 1)

    out = draft_sla_proposals(store, store.all_records())
    assert len(out["drafted"]) == 1
    proposal = out["drafted"][0]
    assert proposal["dimension"] == "moderate"
    assert proposal["direction"] == "raise"
    # A fresh id — never the pruned proposal's number, and never
    # the survivor's.
    assert proposal["proposal_id"] == "PROP-0003"
    marker = proposal["previously_decided"]
    assert marker == {
        "status": "rejected",
        "proposal_id": "PROP-0001",
        "decided_at": rejected["decided_at"],
        "decided_by": "U. Writer",
        "pruned_at": marker["pruned_at"],
        "pruned_by": "Uma Pruner",
    }
    assert marker["pruned_at"]
    # The rationale names the refusal, its decider, and the prune.
    assert "asked before" in proposal["rationale"]
    assert "PROP-0001" in proposal["rationale"]
    assert "U. Writer" in proposal["rationale"]
    assert "Uma Pruner" in proposal["rationale"]


def test_a_row_draft_after_the_prune_names_the_decision_too():
    store, rejected = _rejected_then_accepted_store()
    service = SubmissionService(store=store, checkpointer=False)
    service.prune_calibration_proposals("Uma Pruner", 1)

    proposal = draft_sla_proposal_from_row(
        store, store.all_records(), "moderate", 10, "U. Writer"
    )
    assert proposal["drafted_from"] == "calibration_row"
    marker = proposal["previously_decided"]
    assert marker["proposal_id"] == "PROP-0001"
    assert marker["decided_by"] == "U. Writer"
    assert marker["decided_at"] == rejected["decided_at"]
    assert "asked before" in proposal["rationale"]


def test_the_marker_survives_in_the_stored_proposal():
    store, _ = _rejected_then_accepted_store()
    service = SubmissionService(store=store, checkpointer=False)
    service.prune_calibration_proposals("Uma Pruner", 1)
    draft_sla_proposals(store, store.all_records())
    stored = load_proposals(store)
    # The stored list holds the accepted high proposal and the
    # re-asked moderate draft — the draft wearing the marker,
    # under an id nothing else in the document wears.
    drafts = [p for p in stored if p["status"] == "draft"]
    assert len(drafts) == 1
    assert drafts[0]["dimension"] == "moderate"
    assert drafts[0]["previously_decided"]["proposal_id"] == "PROP-0001"
    assert len({p["proposal_id"] for p in stored}) == len(stored)
