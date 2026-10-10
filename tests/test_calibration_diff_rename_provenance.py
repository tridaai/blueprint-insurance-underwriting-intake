"""Round 22: rename provenance in the calibration diff.

A diff cites its sides by their current labels. A side
renamed since it was kept (round 21) therefore reads as if
it had always worn the new name — and a citation written at
the time, under the old label, names nothing by the time it
is followed. The diff now carries each side's label history
as data (``from_label_history`` / ``to_label_history``) and
the provenance line names the move beside the kept-from
provenance: what the side was kept as, what it was renamed
to, by whom, and when. The rename note rides in
``provenance``, beside ``movement`` — the movement sentence
a citation carries stays the sentence the operator read.
"""

from __future__ import annotations

from datetime import date, timedelta

from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

BASE = date(2026, 9, 1)


def _decided(sid: str, tier: str, wait_days: int) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="approved",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            decided_at=(BASE + timedelta(days=wait_days)).isoformat(),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": BASE.isoformat(),
        },
    )


def _service_with_two_keeps() -> SubmissionService:
    store = InMemoryStore()
    store.save(_decided("M-2", "moderate", 2))
    service = SubmissionService(store=store, checkpointer=False)
    service.snapshot_sla_calibration("Ops Lead", label="march")
    store.save(_decided("M-9", "moderate", 4))
    service.snapshot_sla_calibration("Ops Lead", label="april")
    return service


def _diff(service: SubmissionService, **kwargs) -> dict:
    return service.sla_calibration_snapshot_diff(**kwargs)["diff"]


# --- the history rides the diff as data ------------------------------------------


def test_a_renamed_side_carries_its_history_on_the_diff():
    service = _service_with_two_keeps()
    service.rename_sla_calibration_snapshot_label(
        "march", "pre-change", "Uma"
    )
    diff = _diff(service, from_label="pre-change", to_label="april")
    assert diff["from_label"] == "pre-change"
    (move,) = diff["from_label_history"]
    assert move["from"] == "march"
    assert move["to"] == "pre-change"
    assert move["renamed_by"] == "Uma"
    assert move["renamed_at"]
    assert diff["to_label_history"] == []


def test_an_unrenamed_diff_carries_empty_histories():
    service = _service_with_two_keeps()
    diff = _diff(service, from_label="march", to_label="april")
    assert diff["from_label_history"] == []
    assert diff["to_label_history"] == []
    assert "renamed" not in diff["provenance"]


# --- the provenance line names the move -------------------------------------------


def test_the_provenance_names_the_rename_from_to_who_and_when():
    service = _service_with_two_keeps()
    service.rename_sla_calibration_snapshot_label(
        "march", "pre-change", "Uma"
    )
    diff = _diff(service, from_label="pre-change", to_label="april")
    provenance = diff["provenance"]
    assert "earlier side" in provenance
    assert "kept as 'march'" in provenance
    assert "renamed to 'pre-change'" in provenance
    assert "by Uma" in provenance
    # The citation written under the old name is reconciled,
    # not silently re-pointed: the old name names nothing now.
    assert "citations by the old name name nothing now" in provenance
    # The newer side was never renamed — no note for it.
    assert "newer side" not in provenance


def test_both_sides_renamed_are_both_named():
    service = _service_with_two_keeps()
    service.rename_sla_calibration_snapshot_label(
        "march", "pre-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "april", "post-change", "Ops Lead"
    )
    diff = _diff(service, from_label="pre-change", to_label="post-change")
    assert "earlier side" in diff["provenance"]
    assert "kept as 'march'" in diff["provenance"]
    assert "newer side" in diff["provenance"]
    assert "kept as 'april'" in diff["provenance"]
    assert "renamed to 'post-change'" in diff["provenance"]


def test_a_twice_renamed_side_names_the_count_and_the_latest_move():
    service = _service_with_two_keeps()
    service.rename_sla_calibration_snapshot_label(
        "march", "mid-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "mid-change", "pre-change", "Ops Lead"
    )
    diff = _diff(service, from_label="pre-change", to_label="april")
    assert len(diff["from_label_history"]) == 2
    provenance = diff["provenance"]
    assert "kept as 'march'" in provenance
    assert "renamed 2 times" in provenance
    assert "most recently to 'pre-change'" in provenance
    assert "by Ops Lead" in provenance


def test_the_movement_sentence_is_not_rewritten_by_a_rename():
    service = _service_with_two_keeps()
    before = _diff(service, from_label="march", to_label="april")
    service.rename_sla_calibration_snapshot_label(
        "march", "pre-change", "Uma"
    )
    after = _diff(service, from_label="pre-change", to_label="april")
    # The movement anchors on the older side's handle, which
    # the rename legitimately changed — so compare the tiers'
    # movement and the sentence's shape, not the anchor: the
    # rename annotation lives in provenance, and the per-tier
    # movement the sentence speaks is untouched by it.
    assert after["tiers"] == before["tiers"]
    assert after["movement"].replace(
        "'pre-change'", "'march'"
    ) == before["movement"]
    assert "renamed" in after["provenance"]
    assert "renamed" not in after["movement"]
