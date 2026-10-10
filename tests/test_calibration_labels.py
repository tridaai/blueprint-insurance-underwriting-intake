"""Round 16: calibration snapshot labels.

A kept calibration report is cited in budget debates, and
nobody cites ``cal-0007`` — they cite "the post-budget-change
report". The taker can now label a snapshot when keeping it;
the label is kept verbatim and rides the snapshot, the list,
and the diff, so the argument ("which report says that?")
reads in business terms. Unlabelled snapshots — every one kept
before this round, and any kept without a label — read as an
empty label: a snapshot predating labels is a fact about the
snapshot.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    build_calibration_snapshot,
    calibration_snapshot_diff,
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


# --- the builder and the diff -------------------------------------------------


def test_a_snapshot_keeps_its_label():
    report = {
        "budgets": {"moderate": 5},
        "decided_cases_weighed": 1,
        "tiers": [],
    }
    labelled = build_calibration_snapshot(
        report, "cal-0001", "Ops Lead", label="post-budget-change"
    )
    assert labelled["label"] == "post-budget-change"
    unlabelled = build_calibration_snapshot(report, "cal-0002", "Ops Lead")
    assert unlabelled["label"] == ""


def test_the_diff_reads_in_business_terms():
    report = {
        "budgets": {"moderate": 5},
        "decided_cases_weighed": 1,
        "tiers": [],
    }
    older = build_calibration_snapshot(
        report, "cal-0001", "Ops Lead", label="pre-budget-change"
    )
    newer = build_calibration_snapshot(
        report, "cal-0002", "Ops Lead", label="post-budget-change"
    )
    diff = calibration_snapshot_diff(older, newer)
    assert diff["from_label"] == "pre-budget-change"
    assert diff["to_label"] == "post-budget-change"


def test_a_snapshot_predating_labels_reads_as_unlabelled():
    report = {
        "budgets": {"moderate": 5},
        "decided_cases_weighed": 1,
        "tiers": [],
    }
    old = build_calibration_snapshot(report, "cal-0001", "Ops Lead")
    old.pop("label")  # a kept snapshot from before labels existed
    newer = build_calibration_snapshot(
        report, "cal-0002", "Ops Lead", label="post-budget-change"
    )
    diff = calibration_snapshot_diff(old, newer)
    assert diff["from_label"] == ""
    assert diff["to_label"] == "post-budget-change"


# --- the service ---------------------------------------------------------------


def test_the_service_keeps_and_lists_the_label():
    service = _service()
    snapshot = service.snapshot_sla_calibration(
        "Ops Lead", label="post-budget-change"
    )
    assert snapshot["label"] == "post-budget-change"
    plain = service.snapshot_sla_calibration("Ops Lead")
    assert plain["label"] == ""
    listed = service.sla_calibration_snapshots()
    assert [s["label"] for s in listed] == ["post-budget-change", ""]
    diff = service.sla_calibration_snapshot_diff()["diff"]
    assert diff["from_label"] == "post-budget-change"
    assert diff["to_label"] == ""


# --- the API -----------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    store.save(_decided("M-2", "moderate", 2))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    return TestClient(api_module.app)


def test_api_keeps_a_label_and_the_diff_names_it(client):
    first = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-budget-change"},
    )
    assert first.status_code == 200
    assert first.json()["label"] == "pre-budget-change"
    second = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "post-budget-change"},
    )
    assert second.json()["label"] == "post-budget-change"
    listed = client.get("/queue/sla/calibration/snapshots").json()
    assert [s["label"] for s in listed["snapshots"]] == [
        "pre-budget-change",
        "post-budget-change",
    ]
    diff = client.get("/queue/sla/calibration/snapshots/diff").json()["diff"]
    assert diff["from_label"] == "pre-budget-change"
    assert diff["to_label"] == "post-budget-change"


def test_api_a_snapshot_without_a_label_reads_empty(client):
    kept = client.post(
        "/queue/sla/calibration/snapshots", json={"actor": "Ops Lead"}
    )
    assert kept.status_code == 200
    assert kept.json()["label"] == ""
    anonymous = client.post(
        "/queue/sla/calibration/snapshots", json={"label": "x"}
    )
    assert anonymous.status_code == 422  # the actor is still required
