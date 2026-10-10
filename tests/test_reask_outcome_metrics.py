"""Round 22: re-ask outcomes, counted in /metrics.

Round 21 made a re-ask identifiable — the draft carries the
pruned decision it re-asks as ``previously_decided`` — but
nothing read the markers back. The governance question is a
population question: how many directions came back after
retention forgot their refusal, and were they decided
differently the second time, or does the prune-then-re-ask
loop circle? ``/metrics`` now carries a ``proposal_reasks``
family over the proposals document: re-asks drafted, still
open, decided — and of the decided, accepted (the second
decision **reversed** the refusal the marker carries) against
rejected (it **repeated** it), with withdrawals counted as
neither, and the split by the scan that drafted them.
"""

from __future__ import annotations

from underwriting_agent.insights import compute_metrics
from underwriting_agent.proposals import (
    decide_proposal,
    draft_sla_proposals,
    load_proposals,
)
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore


def _marker() -> dict:
    return {
        "status": "rejected",
        "proposal_id": "PROP-0001",
        "decided_at": "2026-09-20T09:00:00+00:00",
        "decided_by": "First Refuser",
        "pruned_at": "2026-10-01T09:00:00+00:00",
        "pruned_by": "Uma Pruner",
    }


def _proposal(status: str, *, reask: bool, kind: str = "sla_budget") -> dict:
    proposal = {"kind": kind, "status": status, "dimension": "moderate"}
    if reask:
        proposal["previously_decided"] = _marker()
    return proposal


# --- the aggregate over the document --------------------------------------------


def test_no_proposals_reads_as_no_reasks():
    family = compute_metrics([], proposals=[])["proposal_reasks"]
    assert family == {
        "reasks": 0,
        "open": 0,
        "decided": 0,
        "accepted": 0,
        "rejected": 0,
        "withdrawn": 0,
        "repeated_first_decision": 0,
        "reversed_first_decision": 0,
        "by_kind": {},
    }


def test_first_asks_are_not_reasks_however_they_were_decided():
    proposals = [
        _proposal("draft", reask=False),
        _proposal("accepted", reask=False),
        _proposal("rejected", reask=False),
    ]
    family = compute_metrics([], proposals=proposals)["proposal_reasks"]
    assert family["reasks"] == 0
    assert family["decided"] == 0


def test_the_second_decision_is_read_against_the_first():
    proposals = [
        _proposal("draft", reask=True),
        _proposal("accepted", reask=True),
        _proposal("rejected", reask=True),
        _proposal("withdrawn", reask=True),
        _proposal("accepted", reask=True, kind="diff_trigger_threshold"),
    ]
    family = compute_metrics([], proposals=proposals)["proposal_reasks"]
    assert family["reasks"] == 5
    assert family["open"] == 1
    assert family["decided"] == 4
    assert family["accepted"] == 2
    assert family["rejected"] == 1
    assert family["withdrawn"] == 1
    # The marker's first decision is always a rejection: a
    # second acceptance reversed it, a second rejection
    # repeated it, a withdrawal did neither.
    assert family["reversed_first_decision"] == 2
    assert family["repeated_first_decision"] == 1
    assert family["by_kind"] == {
        "diff_trigger_threshold": 1,
        "sla_budget": 4,
    }


def test_metrics_without_proposals_still_carries_the_family():
    # compute_metrics' older callers pass no proposals; the
    # family reads empty rather than going missing.
    family = compute_metrics([])["proposal_reasks"]
    assert family["reasks"] == 0


# --- the live flow: marker to metric ---------------------------------------------


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


def _reask_service() -> SubmissionService:
    """A service whose store holds one live re-ask: a moderate
    raise rejected (PROP-0001), a high lower accepted
    (PROP-0002), the rejection's record pruned away under
    keep-last-1, and the moderate direction re-drafted wearing
    the marker (the round-21 flow)."""
    store = InMemoryStore()
    # Moderate budget is 5 days; an 8-day wait, approved anyway
    # — the scan's raise evidence (see test_pruned_direction_reasks).
    store.save(_record("M-OK", "2026-09-25", "2026-10-03T10:00:00+00:00"))
    service = SubmissionService(store=store, checkpointer=False)
    first = draft_sla_proposals(store, store.all_records())
    assert len(first["drafted"]) == 1
    decide_proposal(store, "PROP-0001", "U. Writer", accept=False)
    for record in (
        _record("H-1", "2026-09-28", "2026-09-30T09:00:00+00:00",
                tier="high", status="rejected"),
        _record("H-2", "2026-09-27", "2026-09-30T09:00:00+00:00",
                tier="high", status="rejected"),
    ):
        store.save(record)
    second = draft_sla_proposals(store, store.all_records())
    assert [p["dimension"] for p in second["drafted"]] == ["high"]
    decide_proposal(store, "PROP-0002", "U. Writer", accept=True)
    service.prune_calibration_proposals("Uma Pruner", 1)
    out = draft_sla_proposals(store, store.all_records())
    assert len(out["drafted"]) == 1
    assert "previously_decided" in out["drafted"][0]
    return service


def test_a_live_reask_counts_as_open_until_it_is_decided():
    service = _reask_service()
    family = service.metrics()["proposal_reasks"]
    assert family["reasks"] == 1
    assert family["open"] == 1
    assert family["decided"] == 0
    assert family["by_kind"] == {"sla_budget": 1}


def test_accepting_the_reask_counts_as_a_reversal():
    service = _reask_service()
    store = service._get_store()
    (reask,) = [
        p for p in load_proposals(store) if p.get("previously_decided")
    ]
    decide_proposal(store, reask["proposal_id"], "U. Writer", accept=True)
    family = service.metrics()["proposal_reasks"]
    assert family["open"] == 0
    assert family["decided"] == 1
    assert family["accepted"] == 1
    assert family["reversed_first_decision"] == 1
    assert family["repeated_first_decision"] == 0


def test_rejecting_the_reask_counts_as_a_repeat():
    service = _reask_service()
    store = service._get_store()
    (reask,) = [
        p for p in load_proposals(store) if p.get("previously_decided")
    ]
    decide_proposal(store, reask["proposal_id"], "U. Writer", accept=False)
    family = service.metrics()["proposal_reasks"]
    assert family["decided"] == 1
    assert family["rejected"] == 1
    assert family["repeated_first_decision"] == 1
    assert family["reversed_first_decision"] == 0
