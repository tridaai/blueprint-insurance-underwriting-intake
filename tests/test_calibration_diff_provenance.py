"""Round 20: keep-current provenance in the calibration diff.

A snapshot kept from the live report is marked on itself, but
a diff that named it as a side did not say so: a debate
reading "march → post-review" could not tell whether the
newer side was the live report at its keep moment or a
separately computed keep. The diff now carries each side's
provenance as data (``from_kept_from`` / ``to_kept_from``) and
as a ``provenance`` prose line naming the keeper and the keep
moment. The line rides *beside* the movement sentence, never
inside it — round 19 pinned the cited diff's movement to the
sentence the operator read, and annotating a side's origin
must not rewrite the sentence that was read.
"""

from __future__ import annotations

from datetime import date, timedelta

from underwriting_agent.proposals import calibration_snapshot_diff
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


def _service() -> SubmissionService:
    store = InMemoryStore()
    store.save(_decided("M-2", "moderate", 2))
    return SubmissionService(store=store, checkpointer=False)


def _report_with_movement(service: SubmissionService) -> dict:
    """Keep 'march', move the live evidence, keep the live side
    too — the two-keep state a cited debate actually holds."""
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service._get_store().save(_decided("M-9", "moderate", 4))
    service.keep_calibration_current("Ops Lead", "post-review")
    return service.sla_calibration_snapshot_diff(
        from_label="march", to_label="post-review"
    )["diff"]


# --- the data ----------------------------------------------------------------------------


def test_the_diff_carries_each_sides_provenance():
    diff = _report_with_movement(_service())
    assert diff["from_kept_from"] == ""
    assert diff["to_kept_from"] == "current"


def test_two_plain_keeps_carry_no_provenance():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service.snapshot_sla_calibration("Ops Lead", label="april")
    diff = service.sla_calibration_snapshot_diff()["diff"]
    assert diff["from_kept_from"] == ""
    assert diff["to_kept_from"] == ""
    assert diff["provenance"] == ""


# --- the prose ----------------------------------------------------------------------------


def test_the_provenance_line_names_the_keeper_and_the_moment():
    service = _service()
    kept = service.keep_calibration_current("Priya Nair", "post-review")
    service.snapshot_sla_calibration("Ops Lead", label="march")
    diff = service.sla_calibration_snapshot_diff(
        from_label="post-review", to_label="march"
    )["diff"]
    # The pair walks in keep order, so the kept-from-current side
    # here is the earlier one — provenance follows the side, not
    # the selector.
    assert diff["from_kept_from"] == "current"
    line = diff["provenance"]
    assert "The earlier side ('post-review')" in line
    assert "kept from the live report by Priya Nair" in line
    assert kept["taken_at"] in line
    assert "not a separately computed report" in line


def test_the_newer_sides_provenance_is_named_too():
    diff = _report_with_movement(_service())
    line = diff["provenance"]
    assert "The newer side ('post-review')" in line
    assert "kept from the live report by Ops Lead" in line


def test_a_diff_against_current_names_the_live_side_as_live():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    diff = service.sla_calibration_snapshot_diff(to_label="current")[
        "diff"
    ]
    assert diff["to_kept_from"] == ""
    assert "the live report itself" in diff["provenance"]


def test_provenance_never_rewrites_the_movement_sentence():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service._get_store().save(_decided("M-9", "moderate", 4))
    shown = service.sla_calibration_snapshot_diff(to_label="current")[
        "diff"
    ]
    kept = service.keep_calibration_current("Ops Lead", "now-cite")
    cited = service.sla_calibration_snapshot_diff(
        from_label="march", to_label="now-cite"
    )["diff"]
    # Round 19's invariant stands: the cited diff is the diff
    # that was read — same tiers, same movement sentence.
    assert cited["tiers"] == shown["tiers"]
    assert cited["movement"] == shown["movement"]
    # What round 20 adds rides beside it: the cited side's
    # origin, which the live side could not have carried.
    assert "post-review" not in cited["provenance"]
    assert "'now-cite'" in cited["provenance"]
    assert kept["taken_at"] in cited["provenance"]
    assert shown["provenance"] != cited["provenance"]


def test_the_pure_diff_reads_provenance_off_the_snapshots():
    older = {
        "snapshot_id": "cal-0001",
        "label": "march",
        "taken_at": "2026-09-05T09:00:00+00:00",
        "taken_by": "Ops Lead",
        "tiers": [],
    }
    newer = {
        "snapshot_id": "cal-0002",
        "label": "post-review",
        "taken_at": "2026-10-02T09:00:00+00:00",
        "taken_by": "Priya Nair",
        "kept_from": "current",
        "tiers": [],
    }
    diff = calibration_snapshot_diff(older, newer)
    assert diff["to_kept_from"] == "current"
    assert "by Priya Nair at 2026-10-02T09:00:00+00:00" in (
        diff["provenance"]
    )
