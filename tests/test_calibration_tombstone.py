"""Round 24: a tombstone read for pruned citations.

Round 23's citation repair answers for a report that still
stands; a report a prune removed had nothing to descend to,
and its labels 404'd exactly like a name that never existed.
The prune records know the difference — round 23 kept each
pruned report's renames whole in the entry, and the entry now
also keeps the label each pruned report wore when it left —
so a by-label read for a pruned report's label answers with
a **tombstone** instead: the label the report wore when it
left, the rename chain that led there, and the prune itself
(who, when). "Existed, pruned on this date by this operator"
is said plainly instead of silence — while still returning
no report. The 404 survives only for a name that never named
anything.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import calibration_snapshot_tombstone
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


def _move(frm: str, to: str, at: str, by: str = "Uma") -> dict:
    return {"from": frm, "to": to, "renamed_by": by, "renamed_at": at}


def _prune_entry(pruned_at: str, pruned_by: str = "Uma", **kw) -> dict:
    entry = {
        "pruned_at": pruned_at,
        "pruned_by": pruned_by,
        "requested_keep": 1,
        "pruned_count": 1,
        "pruned_ids": ["cal-0001"],
        "pruned_renames": [],
        "pruned_labels": [{"snapshot_id": "cal-0001", "label": "may"}],
        "surviving_first": "cal-0002",
        "surviving_last": "cal-0002",
    }
    entry.update(kw)
    return entry


# --- the pure fate read ---------------------------------------------------------


def test_a_label_worn_at_the_prune_answers_with_its_tombstone():
    prunes = [_prune_entry("2026-10-06T09:00:00+00:00")]
    stone = calibration_snapshot_tombstone(prunes, "may")
    assert stone["cited_label"] == "may"
    assert stone["snapshot_id"] == "cal-0001"
    assert stone["label_at_prune"] == "may"
    # Never renamed: the chain is empty, honestly — the name
    # named the report directly until the prune.
    assert stone["rename_chain"] == []
    assert stone["pruned_at"] == "2026-10-06T09:00:00+00:00"
    assert stone["pruned_by"] == "Uma"
    assert stone["pruned"] is True


def test_a_renamed_away_label_on_a_pruned_report_walks_its_chain():
    moves = [
        _move("march", "mid-change", "2026-10-03T09:00:00+00:00"),
        _move(
            "mid-change", "pre-change",
            "2026-10-04T09:00:00+00:00", by="Ops Lead",
        ),
    ]
    prunes = [
        _prune_entry(
            "2026-10-06T09:00:00+00:00",
            pruned_labels=[
                {"snapshot_id": "cal-0001", "label": "pre-change"}
            ],
            pruned_renames=[
                {
                    "snapshot_id": "cal-0001",
                    "label_at_prune": "pre-change",
                    **move,
                }
                for move in moves
            ],
        )
    ]
    stone = calibration_snapshot_tombstone(prunes, "march")
    assert stone["label_at_prune"] == "pre-change"
    assert [m["from"] for m in stone["rename_chain"]] == [
        "march",
        "mid-change",
    ]
    # The middle of the chain answers from its own name.
    stone = calibration_snapshot_tombstone(prunes, "mid-change")
    assert [m["from"] for m in stone["rename_chain"]] == ["mid-change"]
    # And the resting place itself: the whole chain led there.
    stone = calibration_snapshot_tombstone(prunes, "pre-change")
    assert [m["from"] for m in stone["rename_chain"]] == [
        "march",
        "mid-change",
    ]


def test_a_name_no_prune_removed_has_no_tombstone():
    prunes = [_prune_entry("2026-10-06T09:00:00+00:00")]
    assert calibration_snapshot_tombstone(prunes, "nope") is None
    assert calibration_snapshot_tombstone(prunes, "") is None
    assert calibration_snapshot_tombstone([], "may") is None


def test_the_latest_prunes_answer_wins_for_a_retaken_name():
    # "may" was pruned with cal-0001, retaken by a later keep,
    # and pruned again with cal-0007: the name's most recent
    # fate is the tombstone's answer.
    first = _prune_entry("2026-10-06T09:00:00+00:00")
    second = _prune_entry(
        "2026-11-02T09:00:00+00:00",
        pruned_by="Ravi",
        pruned_ids=["cal-0007"],
        pruned_labels=[{"snapshot_id": "cal-0007", "label": "may"}],
    )
    stone = calibration_snapshot_tombstone([first, second], "may")
    assert stone["snapshot_id"] == "cal-0007"
    assert stone["pruned_by"] == "Ravi"
    assert stone["pruned_at"] == "2026-11-02T09:00:00+00:00"


def test_an_old_prune_entry_answers_only_for_labels_its_renames_carry():
    # A prune entry written before pruned_labels existed: a
    # renamed report's names are answerable from its kept
    # renames; an unrenamed report left no label behind, and
    # the tombstone does not invent one.
    old = _prune_entry(
        "2026-10-06T09:00:00+00:00",
        pruned_ids=["cal-0001", "cal-0002"],
        pruned_labels=[],
        pruned_renames=[
            {
                "snapshot_id": "cal-0001",
                "label_at_prune": "pre-change",
                "from": "march",
                "to": "pre-change",
                "renamed_by": "Uma",
                "renamed_at": "2026-10-03T09:00:00+00:00",
            }
        ],
    )
    stone = calibration_snapshot_tombstone([old], "march")
    assert stone["snapshot_id"] == "cal-0001"
    assert stone["label_at_prune"] == "pre-change"
    # cal-0002's label was never recorded anywhere.
    assert calibration_snapshot_tombstone([old], "april") is None


# --- the prune record keeps the resting labels -----------------------------------


def test_the_prune_entry_keeps_each_pruned_reports_label():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service._get_store().save(_decided("M-9", "moderate", 4))
    service.snapshot_sla_calibration("Ops Lead", label="april")
    preview = service.preview_prune_sla_calibration_snapshots(1)
    assert preview["would_prune_labels"] == [
        {"snapshot_id": "cal-0001", "label": "march"}
    ]
    report = service.prune_sla_calibration_snapshots("Uma", 1)
    # Preview and prune read the one computation: the labels
    # the preview named are the labels the entry kept.
    assert report["prune"]["pruned_labels"] == preview[
        "would_prune_labels"
    ]


# --- the service and API -----------------------------------------------------------


def _kept_renamed_pruned() -> SubmissionService:
    """cal-0001 kept as pre-change, renamed to final-baseline,
    then pruned away by a later keep."""
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="pre-change")
    service.rename_sla_calibration_snapshot_label(
        "pre-change", "final-baseline", "Uma"
    )
    service._get_store().save(_decided("M-9", "moderate", 4))
    service.snapshot_sla_calibration("Ops Lead", label="later")
    service.prune_sla_calibration_snapshots("Ravi", 1)
    return service


def test_the_service_reads_a_pruned_reports_tombstone():
    service = _kept_renamed_pruned()
    stone = service.sla_calibration_snapshot_tombstone("pre-change")
    assert stone["snapshot_id"] == "cal-0001"
    assert stone["label_at_prune"] == "final-baseline"
    assert stone["pruned_by"] == "Ravi"
    assert [m["to"] for m in stone["rename_chain"]] == [
        "final-baseline"
    ]
    # The resting label answers too, with the chain that led
    # there — and a name that never named anything does not.
    stone = service.sla_calibration_snapshot_tombstone("final-baseline")
    assert stone["label_at_prune"] == "final-baseline"
    assert len(stone["rename_chain"]) == 1
    assert service.sla_calibration_snapshot_tombstone("nope") is None


def test_a_kept_reports_label_never_reaches_the_tombstone():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="may")
    # The direct read owns a carried label; the tombstone is
    # only for labels whose report is gone.
    assert service.sla_calibration_snapshot_by_label("may") is not None
    assert service.sla_calibration_snapshot_tombstone("may") is None


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_the_by_label_read_answers_a_tombstone(client):
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={
            "actor": "Uma",
            "from_label": "pre-change",
            "to_label": "final-baseline",
        },
    )
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "later"},
    )
    client.post(
        "/queue/sla/calibration/snapshots/prune",
        json={"actor": "Ravi", "keep": 1},
    )
    res = client.get(
        "/queue/sla/calibration/snapshots/by-label/pre-change"
    )
    assert res.status_code == 200
    body = res.json()
    # A tombstone, not a report: no snapshot rides in it, and
    # nothing pretends the report still stands.
    assert "snapshot" not in body
    assert body["pruned"] is True
    assert body["label_at_prune"] == "final-baseline"
    assert body["pruned_by"] == "Ravi"
    # And a name that never named anything still 404s.
    res = client.get("/queue/sla/calibration/snapshots/by-label/nope")
    assert res.status_code == 404
