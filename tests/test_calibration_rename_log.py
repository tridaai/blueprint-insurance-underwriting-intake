"""Round 22: a rename log across kept reports.

Round 21 recorded each rename on its own snapshot
(``label_history``) — the right place to audit one report's
addressing, and the wrong shape for a debate reconstructing
its *own* addressing history: which names has it cited, in
what order did they move, when did each stop naming
anything? That archaeology read the kept reports one at a
time. The rename log is the document-level projection
(``proposals.calibration_rename_log``, served at
``GET /queue/sla/calibration/snapshots/renames``): every
rename across the kept reports, in the order the renames
happened, each entry naming the report it moved on and the
label that report wears now. A projection, not a second
record — prune a report and its renames leave with it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import calibration_rename_log
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


def _service_with_two_keeps() -> SubmissionService:
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service._get_store().save(_decided("M-9", "moderate", 4))
    service.snapshot_sla_calibration("Ops Lead", label="april")
    return service


# --- the projection over fabricated histories --------------------------------------


def _snapshot(snapshot_id: str, label: str, history: list[dict]) -> dict:
    return {
        "snapshot_id": snapshot_id,
        "label": label,
        "label_history": history,
    }


def _move(frm: str, to: str, by: str, at: str) -> dict:
    return {"from": frm, "to": to, "renamed_by": by, "renamed_at": at}


def test_the_log_walks_in_the_order_the_renames_happened():
    snapshots = [
        _snapshot(
            "cal-0002",
            "post-change",
            [_move("april", "post-change", "Uma", "2026-10-05T09:00:00+00:00")],
        ),
        _snapshot(
            "cal-0001",
            "pre-change",
            [
                _move("march", "mid-change", "Uma", "2026-10-03T09:00:00+00:00"),
                _move("mid-change", "pre-change", "Ops Lead", "2026-10-04T09:00:00+00:00"),
            ],
        ),
    ]
    log = calibration_rename_log(snapshots)
    # Chronological across reports — not grouped by report,
    # even though cal-0002 was kept after cal-0001.
    assert [(e["from"], e["to"]) for e in log] == [
        ("march", "mid-change"),
        ("mid-change", "pre-change"),
        ("april", "post-change"),
    ]
    assert log[0]["snapshot_id"] == "cal-0001"
    assert log[0]["current_label"] == "pre-change"
    assert log[0]["renamed_by"] == "Uma"
    assert log[2]["snapshot_id"] == "cal-0002"
    assert log[2]["current_label"] == "post-change"


def test_reports_never_renamed_contribute_nothing():
    snapshots = [
        _snapshot("cal-0001", "march", []),
        {"snapshot_id": "cal-0002", "label": "april"},
    ]
    assert calibration_rename_log(snapshots) == []
    assert calibration_rename_log([]) == []


# --- the service read ----------------------------------------------------------------


def test_the_service_log_collects_every_kept_reports_renames():
    service = _service_with_two_keeps()
    assert service.sla_calibration_rename_log() == {
        "renames": [],
        "count": 0,
    }
    service.rename_sla_calibration_snapshot_label(
        "march", "pre-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "april", "post-change", "Ops Lead"
    )
    out = service.sla_calibration_rename_log()
    assert out["count"] == 2
    by_from = {e["from"]: e for e in out["renames"]}
    march = by_from["march"]
    assert march["to"] == "pre-change"
    assert march["renamed_by"] == "Uma"
    assert march["current_label"] == "pre-change"
    assert march["snapshot_id"] == service.sla_calibration_snapshots()[0][
        "snapshot_id"
    ]
    assert by_from["april"]["renamed_by"] == "Ops Lead"


def test_a_second_rename_of_the_same_report_appends_in_its_history_order():
    service = _service_with_two_keeps()
    service.rename_sla_calibration_snapshot_label(
        "march", "mid-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "mid-change", "pre-change", "Ops Lead"
    )
    out = service.sla_calibration_rename_log()
    assert [(e["from"], e["to"]) for e in out["renames"]] == [
        ("march", "mid-change"),
        ("mid-change", "pre-change"),
    ]
    assert all(e["current_label"] == "pre-change" for e in out["renames"])


# --- over the API ----------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_the_log_over_the_api(client):
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    empty = client.get("/queue/sla/calibration/snapshots/renames")
    assert empty.status_code == 200
    assert empty.json() == {"renames": [], "count": 0}
    client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={
            "actor": "Uma",
            "from_label": "pre-change",
            "to_label": "pre-budget-change",
        },
    )
    res = client.get("/queue/sla/calibration/snapshots/renames")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    (entry,) = body["renames"]
    assert entry["from"] == "pre-change"
    assert entry["to"] == "pre-budget-change"
    assert entry["renamed_by"] == "Uma"
    assert entry["current_label"] == "pre-budget-change"
    assert entry["snapshot_id"]
    assert entry["renamed_at"]
