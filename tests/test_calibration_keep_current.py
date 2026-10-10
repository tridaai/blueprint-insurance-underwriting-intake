"""Round 19: keep the current side from the diff.

Round 18's diff against ``current`` answered "what moved
since 'march'?" without storing anything — but a diff that
shows movement worth citing ended in a clumsy loop: keep the
live report by hand, then re-run the diff to cite it, under a
second computation the citation could not be sure matched
the one the operator read. The keep-from-diff act closes the
loop: the live report is shaped once in the snapshot's own
shape and that very shape is kept — re-stamped with the next
id, the keeper's name, and the label they give at that
moment, and marked ``kept_from: "current"``. The label is
required (the act exists to make a citation) and answers to
every keep's rules: "current" reserved, duplicates refused.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    live_calibration_snapshot,
    sla_budget_calibration,
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


# --- the act ------------------------------------------------------------------------


def test_keep_current_keeps_the_live_shape_under_the_given_label():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    kept = service.keep_calibration_current("Ops Lead", "post-review")
    assert kept["snapshot_id"] == "cal-0002"
    assert kept["taken_by"] == "Ops Lead"
    assert kept["label"] == "post-review"
    assert kept["kept_from"] == "current"
    assert kept["taken_at"]
    # The evidence kept is the live report's, in the shaper's
    # own shape: same budgets, same weighed cases, same tiers.
    report = sla_budget_calibration(
        service._get_store().all_records(),
        service.effective_sla_budgets(),
        budget_sources={"moderate": "test"},
    )
    live = live_calibration_snapshot(
        service.sla_budget_calibration()
    )
    assert kept["budgets"] == live["budgets"]
    assert kept["decided_cases_weighed"] == live["decided_cases_weighed"]
    assert kept["tiers"] == live["tiers"]
    assert report["decided_cases_weighed"] == kept["decided_cases_weighed"]


def test_the_kept_current_side_is_what_the_diff_showed():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    # Decide another case after the keep: the live report moves.
    service._get_store().save(_decided("M-9", "moderate", 4))
    shown = service.sla_calibration_snapshot_diff(to_label="current")[
        "diff"
    ]
    service.keep_calibration_current("Ops Lead", "now-cite")
    cited = service.sla_calibration_snapshot_diff(
        from_label="march", to_label="now-cite"
    )["diff"]
    # The cited diff is the diff the operator read: same tiers,
    # same movement sentence — no second computation in between.
    assert cited["tiers"] == shown["tiers"]
    assert cited["movement"] == shown["movement"]
    assert cited["to_snapshot"] == "cal-0002"
    assert cited["to_label"] == "now-cite"


def test_keep_current_needs_no_prior_keep():
    service = _service()
    kept = service.keep_calibration_current("Ops Lead", "baseline")
    assert kept["snapshot_id"] == "cal-0001"
    assert service.sla_calibration_snapshot_by_label("baseline") == kept


def test_a_plain_keep_carries_no_kept_from_mark():
    service = _service()
    plain = service.snapshot_sla_calibration("Ops Lead", label="march")
    assert "kept_from" not in plain


# --- the label rules --------------------------------------------------------------------


def test_keep_current_requires_a_label():
    service = _service()
    with pytest.raises(ValueError, match="needs the label"):
        service.keep_calibration_current("Ops Lead", "")
    assert service.sla_calibration_snapshots() == []


def test_keep_current_refuses_the_reserved_label():
    service = _service()
    with pytest.raises(ValueError, match="reserved"):
        service.keep_calibration_current("Ops Lead", "current")
    assert service.sla_calibration_snapshots() == []


def test_keep_current_refuses_a_label_already_in_use():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    with pytest.raises(ValueError, match="already carried by cal-0001"):
        service.keep_calibration_current("Ops Lead", "march")
    assert len(service.sla_calibration_snapshots()) == 1


# --- the API ----------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    import underwriting_agent.api as api_module
    from underwriting_agent.api import app

    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(app), service


def test_keep_current_over_the_api(client):
    test_client, service = client
    res = test_client.post(
        "/queue/sla/calibration/snapshots/keep-current",
        json={"actor": "Ops Lead", "label": "post-review"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "post-review"
    assert body["kept_from"] == "current"
    assert (
        service.sla_calibration_snapshot_by_label("post-review") == body
    )


def test_keep_current_over_the_api_refuses_without_a_label(client):
    test_client, _service_unused = client
    res = test_client.post(
        "/queue/sla/calibration/snapshots/keep-current",
        json={"actor": "Ops Lead"},
    )
    assert res.status_code == 422
    assert "needs the label" in res.json()["detail"]


def test_keep_current_over_the_api_refuses_a_duplicate_label(client):
    test_client, _service_unused = client
    res = test_client.post(
        "/queue/sla/calibration/snapshots/keep-current",
        json={"actor": "Ops Lead", "label": "march"},
    )
    assert res.status_code == 422
    assert "already carried" in res.json()["detail"]
