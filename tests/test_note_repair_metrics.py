"""Round 18: note-repair visibility in /metrics.

Round 15 gave the status note the letter's drafting discipline
— one steered redraft for a wording-only failure, refusal for
substance — but the repairs were recorded per note and never
aggregated, and the refusals evaporated into the raised error.
A repaired note means interpolated record prose carried a
forbidden phrasing: a drafting-quality signal exactly parallel
to the letter-advisory trend. The metrics now carry the family
(``status_note_repairs``): notes drafted, notes repaired with
the family mix, drafts refused, and the refusal-vs-steer split
with the steer success rate — and the two headline counts ride
the ops digest's snapshot counts on the advisory discipline.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest

from underwriting_agent.insights import compute_metrics, ops_digest
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore
from underwriting_agent.schemas import AgentResult, Triage

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def old_submission(sid: str = "SUB-9001") -> dict:
    sub = copy.deepcopy(SAMPLES["SUB-1001"])
    sub["submission_id"] = sid
    sub["submitted_at"] = "2026-08-01"
    return sub


def _poison_gaps(
    service: SubmissionService, gap_sources: list[str], sid: str = "SUB-9001"
) -> None:
    record = service._get_store().get(sid)
    record.result.facts.missing_elements = []
    record.result.facts.gap_sources = gap_sources
    service._get_store().save(record)


def _record(
    sid: str,
    note: dict | None = None,
    refusals: list[dict] | None = None,
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
        ),
        submission={"insured_name": f"Insured {sid}"},
        status_note=note,
        status_note_refusals=refusals or [],
    )


def _repaired_note(families: list[str]) -> dict:
    return {
        "submission_id": "X",
        "text": "INTERNAL DRAFT …",
        "drafted_at": "2026-10-01T09:00:00+00:00",
        "repair_attempted": True,
        "repaired": True,
        "repair_families": families,
        "validation": {"passed": True, "checks": []},
    }


def _refusal(
    *, steered: bool, checks: list[str], families: list[str] | None = None
) -> dict:
    return {
        "refused_at": "2026-10-02T09:00:00+00:00",
        "failed_checks": checks,
        "steered": steered,
        "repair_families": families or [],
    }


# --- the aggregate over stored records ------------------------------------------


def test_the_family_aggregates_repairs_families_and_the_split():
    records = [
        _record("A", note=_repaired_note(["forward_motion"])),
        _record(
            "B",
            note=_repaired_note(["forward_motion", "soft_time"]),
            refusals=[
                _refusal(steered=False, checks=["no_firm_premium"])
            ],
        ),
        _record(
            "C",
            refusals=[
                _refusal(
                    steered=True,
                    checks=["no_forward_motion_in_letter"],
                    families=["forward_motion", "soft_time"],
                ),
                _refusal(steered=False, checks=["no_bind_language"]),
            ],
        ),
        _record("D", note={"text": "a clean note, no repair"}),
    ]
    family = compute_metrics(records)["status_note_repairs"]
    assert family["notes_drafted"] == 3
    assert family["repaired_notes"] == 2
    assert family["by_family"] == {"forward_motion": 3, "soft_time": 2}
    assert family["refusals"] == 3
    assert family["refusals_after_steering"] == 1
    assert family["refusals_unsteered"] == 2
    assert family["by_refused_check"] == {
        "no_bind_language": 1,
        "no_firm_premium": 1,
        "no_forward_motion_in_letter": 1,
    }
    # Steer attempts = repairs + refusals after steering.
    assert family["steer_attempts"] == 3
    assert family["steer_success_rate"] == round(2 / 3, 3)


def test_an_empty_store_reports_zeros_and_no_rate():
    family = compute_metrics([])["status_note_repairs"]
    assert family["notes_drafted"] == 0
    assert family["repaired_notes"] == 0
    assert family["refusals"] == 0
    assert family["steer_attempts"] == 0
    assert family["steer_success_rate"] is None


def test_a_repair_recorded_before_families_were_kept_still_counts():
    legacy = _repaired_note([])
    del legacy["repair_families"]
    family = compute_metrics([_record("L", note=legacy)])[
        "status_note_repairs"
    ]
    assert family["repaired_notes"] == 1
    assert family["by_family"] == {}


# --- the service records what the aggregate counts -------------------------------


def test_a_steered_repair_records_its_families():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service,
        ["Broker advised we will proceed with the file once the audits land"],
    )
    note = service.draft_status_note("SUB-9001")
    assert note["repaired"] is True
    assert note["repair_families"] == ["forward_motion"]
    family = compute_metrics(service._get_store().all_records())[
        "status_note_repairs"
    ]
    assert family["repaired_notes"] == 1
    assert family["by_family"] == {"forward_motion": 1}
    assert family["refusals"] == 0


def test_a_substance_refusal_is_recorded_unsteered():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service, ["Broker quoted the premium at $12,000 for this file"]
    )
    with pytest.raises(ValueError, match="not stored"):
        service.draft_status_note("SUB-9001")
    assert service.status_note("SUB-9001") is None
    record = service._get_store().get("SUB-9001")
    (refusal,) = record.status_note_refusals
    assert refusal["steered"] is False
    assert refusal["failed_checks"] == ["no_firm_premium"]
    assert refusal["repair_families"] == []
    family = compute_metrics([record])["status_note_repairs"]
    assert family["refusals"] == 1
    assert family["refusals_unsteered"] == 1
    assert family["refusals_after_steering"] == 0
    assert family["steer_success_rate"] is None


def test_a_mixed_refusal_is_recorded_with_its_wording_families():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service,
        [
            "We will proceed with the file; "
            "broker quoted the premium at $12,000"
        ],
    )
    with pytest.raises(ValueError, match="not stored"):
        service.draft_status_note("SUB-9001")
    record = service._get_store().get("SUB-9001")
    (refusal,) = record.status_note_refusals
    # Substance present → no steered candidate is produced, so
    # the refusal is unsteered; the wording families the draft
    # carried are still recorded against it.
    assert refusal["steered"] is False
    assert set(refusal["failed_checks"]) == {
        "no_forward_motion_in_letter",
        "no_firm_premium",
    }
    assert refusal["repair_families"] == ["forward_motion"]


def test_refusals_accumulate_across_drafts():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service, ["Broker quoted the premium at $12,000 for this file"]
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="not stored"):
            service.draft_status_note("SUB-9001")
    record = service._get_store().get("SUB-9001")
    assert len(record.status_note_refusals) == 2


def test_a_clean_draft_records_nothing():
    service = make_service()
    service.analyze(old_submission())
    service.draft_status_note("SUB-9001")
    record = service._get_store().get("SUB-9001")
    assert record.status_note_refusals == []
    assert "repair_families" not in record.status_note


# --- the digest counts ------------------------------------------------------------


def test_the_digest_counts_ride_the_note_repairs():
    records = [
        _record("A", note=_repaired_note(["forward_motion"])),
        _record(
            "C",
            refusals=[_refusal(steered=False, checks=["no_bind_language"])],
        ),
    ]
    # Pin the clock inside the follow-through window: the digest
    # ages the fixture note (drafted 2026-10-01) against `now`, and
    # unpinned this test time-bombs the day the note ages into an
    # escalation and total_attention picks it up.
    digest = ops_digest(
        records, now=datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    )
    counts = digest["counts"]
    assert counts["status_note_repaired"] == 1
    assert counts["status_note_refusals"] == 1
    assert digest["status_note_repairs"]["repaired_notes"] == 1
    assert counts["total_attention"] == 0
