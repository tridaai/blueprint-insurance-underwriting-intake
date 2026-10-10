"""Round 21: renaming a kept calibration report's label.

Labels are addresses (round 17), and a debate that renames
its milestone used to re-keep the report to rename it —
forking the citation into two kept reports over one evidence
base. The rename now moves the address on the kept snapshot
itself: the label changes in place, the move is recorded in
the snapshot's ``label_history`` (from, to, who, when), the
evidence is untouched, and the old label is freed at once —
citations by it answer 404 honestly. The new label answers
to every keep's rules: not ``current``, not another kept
report's label, not empty, not the label already carried.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture()
def service() -> SubmissionService:
    svc = _service()
    svc.snapshot_sla_calibration("Ops Lead", label="pre-change")
    svc.snapshot_sla_calibration("Ops Lead", label="post-change")
    return svc


# --- the rename itself ------------------------------------------------------------


def test_the_rename_moves_the_address_and_records_the_move(service):
    renamed = service.rename_sla_calibration_snapshot_label(
        "pre-change", "pre-budget-change", "Uma"
    )
    assert renamed["label"] == "pre-budget-change"
    assert renamed["label_history"] == [
        {
            "from": "pre-change",
            "to": "pre-budget-change",
            "renamed_by": "Uma",
            "renamed_at": renamed["label_history"][0]["renamed_at"],
        }
    ]
    assert renamed["label_history"][0]["renamed_at"]
    # The evidence is the same report: same id, keeper, moment.
    original = service.sla_calibration_snapshots()[0]
    assert renamed["snapshot_id"] == original["snapshot_id"]
    assert renamed["taken_by"] == "Ops Lead"
    assert renamed["tiers"] == original["tiers"]


def test_the_old_label_is_freed_and_the_new_one_resolves(service):
    service.rename_sla_calibration_snapshot_label(
        "pre-change", "pre-budget-change", "Uma"
    )
    assert service.sla_calibration_snapshot_by_label("pre-change") is None
    found = service.sla_calibration_snapshot_by_label("pre-budget-change")
    assert found is not None
    assert found["snapshot_id"] == "cal-0001"
    # The freed label may be kept again, by a later report.
    kept = service.snapshot_sla_calibration("Ops Lead", label="pre-change")
    assert kept["snapshot_id"] == "cal-0003"
    # And the diff follows the citation to the new address.
    diff = service.sla_calibration_snapshot_diff(
        from_label="pre-budget-change", to_label="post-change"
    )
    assert diff["diff"]["from_label"] == "pre-budget-change"


def test_serial_renames_accumulate_in_the_history(service):
    service.rename_sla_calibration_snapshot_label(
        "pre-change", "pre-budget-change", "Uma"
    )
    renamed = service.rename_sla_calibration_snapshot_label(
        "pre-budget-change", "march-baseline", "Ravi"
    )
    assert [h["to"] for h in renamed["label_history"]] == [
        "pre-budget-change",
        "march-baseline",
    ]
    assert [h["renamed_by"] for h in renamed["label_history"]] == [
        "Uma",
        "Ravi",
    ]


# --- the refusals --------------------------------------------------------------------


def test_an_unknown_from_label_is_refused_naming_labels_in_use(service):
    with pytest.raises(ValueError) as exc:
        service.rename_sla_calibration_snapshot_label(
            "no-such", "new-name", "Uma"
        )
    assert "no-such" in str(exc.value)
    assert "pre-change" in str(exc.value)
    assert "post-change" in str(exc.value)


def test_renaming_to_a_carried_label_is_refused_naming_the_holder(service):
    with pytest.raises(ValueError) as exc:
        service.rename_sla_calibration_snapshot_label(
            "pre-change", "post-change", "Uma"
        )
    assert "cal-0002" in str(exc.value)


def test_renaming_to_current_is_refused_as_reserved(service):
    with pytest.raises(ValueError) as exc:
        service.rename_sla_calibration_snapshot_label(
            "pre-change", "current", "Uma"
        )
    assert "reserved" in str(exc.value)


def test_renaming_to_nothing_or_to_itself_is_refused(service):
    with pytest.raises(ValueError):
        service.rename_sla_calibration_snapshot_label(
            "pre-change", "", "Uma"
        )
    with pytest.raises(ValueError):
        service.rename_sla_calibration_snapshot_label(
            "pre-change", "pre-change", "Uma"
        )
    with pytest.raises(ValueError):
        service.rename_sla_calibration_snapshot_label(
            "", "new-name", "Uma"
        )
    # Nothing moved under any of the refusals.
    assert service.sla_calibration_snapshots()[0]["label"] == "pre-change"


# --- over the API -----------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_the_rename_over_the_api_and_the_old_citation_404s(client):
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    res = client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={
            "actor": "Uma",
            "from_label": "pre-change",
            "to_label": "pre-budget-change",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "pre-budget-change"
    assert body["label_history"][0]["renamed_by"] == "Uma"

    old = client.get("/queue/sla/calibration/snapshots/by-label/pre-change")
    assert old.status_code == 404
    new = client.get(
        "/queue/sla/calibration/snapshots/by-label/pre-budget-change"
    )
    assert new.status_code == 200
    assert new.json()["snapshot_id"] == body["snapshot_id"]


def test_the_rename_refusals_are_422_over_the_api(client):
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    for payload in (
        {"actor": "Uma", "from_label": "ghost", "to_label": "x"},
        {"actor": "Uma", "from_label": "pre-change", "to_label": "current"},
        {"actor": "Uma", "from_label": "pre-change", "to_label": ""},
        {"actor": "Uma", "from_label": "pre-change", "to_label": "pre-change"},
    ):
        res = client.post(
            "/queue/sla/calibration/snapshots/rename", json=payload
        )
        assert res.status_code == 422, payload
    # An unnamed renamer is refused like every named act.
    res = client.post(
        "/queue/sla/calibration/snapshots/rename",
        json={"from_label": "pre-change", "to_label": "new-name"},
    )
    assert res.status_code == 422
