"""Round 23: pruned renames kept whole, like pruned rejections.

Round 22's rename log was a projection of the kept reports —
a prune took a report's renames with it, and the addressing
archaeology ended at the prune. Round 21 had already settled
this question for the proposals document: a prune keeps the
pruned *rejections* whole in its own entry, because a
rejection is the history the scans act on. A rename is the
history the *citations* act on, and the calibration prune now
keeps it the same way: ``pruned_renames`` on the prune entry
(each pruned report's label history, flattened move by move
with the label the report wore when it left), the preview
naming them before the act, and the rename log reading them
back as a named stratum — marked ``pruned``, the prune itself
named — behind the kept reports' own histories.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import (
    snapshot_prune_entry,
    snapshot_prune_preview,
)
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


def _service_with_renamed_keeps() -> SubmissionService:
    """Three kept reports; the first renamed twice, the
    second once, the third never."""
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service._get_store().save(_decided("M-9", "moderate", 4))
    service.snapshot_sla_calibration("Ops Lead", label="april")
    service._get_store().save(_decided("M-15", "moderate", 6))
    service.snapshot_sla_calibration("Ops Lead", label="may")
    service.rename_sla_calibration_snapshot_label(
        "march", "mid-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "mid-change", "pre-change", "Ops Lead"
    )
    service.rename_sla_calibration_snapshot_label(
        "april", "post-change", "Uma"
    )
    return service


# --- the prune entry keeps the history whole ---------------------------------


def test_the_prune_entry_keeps_a_pruned_reports_renames_whole():
    snapshots = [
        {
            "snapshot_id": "cal-0001",
            "label": "pre-change",
            "label_history": [
                {
                    "from": "march",
                    "to": "mid-change",
                    "renamed_by": "Uma",
                    "renamed_at": "2026-10-03T09:00:00+00:00",
                },
                {
                    "from": "mid-change",
                    "to": "pre-change",
                    "renamed_by": "Ops Lead",
                    "renamed_at": "2026-10-04T09:00:00+00:00",
                },
            ],
        },
        {"snapshot_id": "cal-0002", "label": "may"},
    ]
    entry = snapshot_prune_entry(
        actor="Uma",
        keep=1,
        pruned=snapshots[:1],
        kept=snapshots[1:],
        pruned_at="2026-10-06T09:00:00+00:00",
    )
    assert entry["pruned_renames"] == [
        {
            "snapshot_id": "cal-0001",
            "label_at_prune": "pre-change",
            "from": "march",
            "to": "mid-change",
            "renamed_by": "Uma",
            "renamed_at": "2026-10-03T09:00:00+00:00",
        },
        {
            "snapshot_id": "cal-0001",
            "label_at_prune": "pre-change",
            "from": "mid-change",
            "to": "pre-change",
            "renamed_by": "Ops Lead",
            "renamed_at": "2026-10-04T09:00:00+00:00",
        },
    ]


def test_a_report_never_renamed_contributes_no_pruned_renames():
    snapshots = [{"snapshot_id": "cal-0001", "label": "march"}]
    entry = snapshot_prune_entry(
        actor="Uma",
        keep=1,
        pruned=snapshots,
        kept=[],
        pruned_at="2026-10-06T09:00:00+00:00",
    )
    assert entry["pruned_renames"] == []


def test_the_preview_names_the_renames_the_prune_would_carry():
    service = _service_with_renamed_keeps()
    preview = service.preview_prune_sla_calibration_snapshots(1)
    assert preview["would_prune_count"] == 2
    assert len(preview["would_prune_renames"]) == 3
    report = service.prune_sla_calibration_snapshots("Uma", 1)
    # Preview and prune read the one computation: the renames
    # the preview named are the renames the entry kept.
    assert report["prune"]["pruned_renames"] == preview[
        "would_prune_renames"
    ]


# --- the log reads the pruned stratum back ------------------------------------


def test_the_log_reads_pruned_renames_back_marked():
    service = _service_with_renamed_keeps()
    service.prune_sla_calibration_snapshots("Uma", 1)
    out = service.sla_calibration_rename_log()
    assert out["count"] == 3
    assert all(e["pruned"] for e in out["renames"])
    by_from = {e["from"]: e for e in out["renames"]}
    march = by_from["march"]
    assert march["to"] == "mid-change"
    assert march["snapshot_id"] == "cal-0001"
    # The label the report wore when it left, not a label it
    # wears now — it wears none.
    assert march["current_label"] == "pre-change"
    assert march["pruned_by"] == "Uma"
    assert march["pruned_at"]
    assert by_from["april"]["current_label"] == "post-change"
    # The twice-moved report's chain still walks in its own
    # order inside the stratum.
    chain = [
        e["from"] for e in out["renames"] if e["snapshot_id"] == "cal-0001"
    ]
    assert chain == ["march", "mid-change"]


def test_kept_and_pruned_strata_read_side_by_side():
    service = _service_with_renamed_keeps()
    # Rename the survivor too, then prune the older two: the
    # log holds one kept rename and three pruned ones.
    service.rename_sla_calibration_snapshot_label(
        "may", "june", "Ops Lead"
    )
    service.prune_sla_calibration_snapshots("Uma", 1)
    out = service.sla_calibration_rename_log()
    assert out["count"] == 4
    kept = [e for e in out["renames"] if not e["pruned"]]
    pruned = [e for e in out["renames"] if e["pruned"]]
    assert len(kept) == 1
    assert kept[0]["from"] == "may"
    assert kept[0]["current_label"] == "june"
    assert "pruned_at" not in kept[0]
    assert len(pruned) == 3


def test_pruning_again_does_not_duplicate_the_stratum():
    service = _service_with_renamed_keeps()
    service.prune_sla_calibration_snapshots("Uma", 2)
    first = service.sla_calibration_rename_log()
    assert first["count"] == 3  # two pruned moves + one kept move... see below
    service.prune_sla_calibration_snapshots("Uma", 1)
    second = service.sla_calibration_rename_log()
    # Each prune kept its own removals whole; no rename is
    # carried twice.
    assert second["count"] == 3
    assert all(e["pruned"] for e in second["renames"])


# --- over the API ---------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_the_pruned_stratum_over_the_api(client):
    for label in ("march", "april", "may"):
        res = client.post(
            "/queue/sla/calibration/snapshots",
            json={"actor": "Ops Lead", "label": label},
        )
        assert res.status_code == 200
    client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={
            "actor": "Uma",
            "from_label": "march",
            "to_label": "pre-change",
        },
    )
    client.post(
        "/queue/sla/calibration/snapshots/prune",
        json={"actor": "Uma", "keep": 1},
    )
    body = client.get("/queue/sla/calibration/snapshots/renames").json()
    assert body["count"] == 1
    (entry,) = body["renames"]
    assert entry["pruned"] is True
    assert entry["from"] == "march"
    assert entry["to"] == "pre-change"
    assert entry["pruned_by"] == "Uma"
    # And the prune record itself, on the snapshots read,
    # carries the renames whole.
    listing = client.get("/queue/sla/calibration/snapshots").json()
    (prune,) = listing["prunes"]
    assert prune["pruned_renames"][0]["from"] == "march"
    assert prune["pruned_renames"][0]["label_at_prune"] == "pre-change"
