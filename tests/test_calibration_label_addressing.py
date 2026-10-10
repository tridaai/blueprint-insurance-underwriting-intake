"""Round 17: addressing kept calibration reports by label.

Round 16 let the taker label a kept report so a budget debate
could cite it ("the post-budget-change report"), but the label
was display only: the list could not be filtered by it, no
read addressed a report by it, and the diff always read the
two latest. Worse, nothing stopped two reports carrying the
same label, so the citation could name two different evidence
bases. Labels are addresses now: keeping refuses a label a
kept snapshot already carries (naming the holder and the
labels in use), the list filters by label, a report reads by
label, and the diff accepts from_label / to_label selectors —
each resolving to the latest kept snapshot carrying it, the
pair walking in keep order.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    calibration_snapshot_by_label,
    calibration_snapshot_labels,
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


def _kept_three(service: SubmissionService) -> None:
    service.snapshot_sla_calibration("Ops Lead", label="pre-change")
    service.snapshot_sla_calibration("Ops Lead", label="mid-review")
    service.snapshot_sla_calibration("Ops Lead", label="post-change")


# --- the resolver -------------------------------------------------------------


def test_by_label_resolves_latest_kept_and_names_labels_in_use():
    snapshots = [
        {"snapshot_id": "cal-0001", "label": "march"},
        {"snapshot_id": "cal-0002", "label": ""},
        {"snapshot_id": "cal-0003", "label": "march"},  # a pre-round-17 repeat
        {"snapshot_id": "cal-0004", "label": "april"},
    ]
    assert calibration_snapshot_labels(snapshots) == [
        "march",
        "march",
        "april",
    ]
    # Repeats from before the discipline resolve to the latest kept.
    assert calibration_snapshot_by_label(snapshots, "march")[
        "snapshot_id"
    ] == "cal-0003"
    assert calibration_snapshot_by_label(snapshots, "april")[
        "snapshot_id"
    ] == "cal-0004"
    assert calibration_snapshot_by_label(snapshots, "may") is None
    # An empty label addresses nothing — unlabelled reports are
    # cited by id, as they always were.
    assert calibration_snapshot_by_label(snapshots, "") is None


# --- the keep -----------------------------------------------------------------


def test_a_repeated_label_is_refused_and_names_what_exists():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="post-budget-change")
    service.snapshot_sla_calibration("Ops Lead", label="q3-review")
    with pytest.raises(ValueError) as excinfo:
        service.snapshot_sla_calibration(
            "Other Lead", label="post-budget-change"
        )
    message = str(excinfo.value)
    assert "post-budget-change" in message  # the colliding label
    assert "cal-0001" in message  # its holder
    assert "q3-review" in message  # the other labels in use
    # The refused keep appended nothing.
    assert len(service.sla_calibration_snapshots()) == 2


def test_unlabelled_keeps_are_unlimited_and_a_prune_frees_a_label():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="cited")
    service.snapshot_sla_calibration("Ops Lead")
    service.snapshot_sla_calibration("Ops Lead")
    assert len(service.sla_calibration_snapshots()) == 3
    # Prune away the holder (the oldest) and the label is free.
    service.prune_sla_calibration_snapshots("Ops Lead", keep=2)
    freed = service.snapshot_sla_calibration("Ops Lead", label="cited")
    assert freed["label"] == "cited"


# --- the reads ------------------------------------------------------------------


def test_the_list_filters_by_label_and_a_report_reads_by_label():
    service = _service()
    _kept_three(service)
    assert [
        s["snapshot_id"]
        for s in service.sla_calibration_snapshots(label="mid-review")
    ] == ["cal-0002"]
    assert service.sla_calibration_snapshots(label="nope") == []
    assert len(service.sla_calibration_snapshots()) == 3
    found = service.sla_calibration_snapshot_by_label("post-change")
    assert found["snapshot_id"] == "cal-0003"
    assert service.sla_calibration_snapshot_by_label("nope") is None


def test_the_diff_resolves_its_sides_by_label():
    service = _service()
    _kept_three(service)
    diff = service.sla_calibration_snapshot_diff(
        from_label="pre-change", to_label="post-change"
    )
    assert diff["snapshot_count"] == 3
    assert diff["diff"]["from_snapshot"] == "cal-0001"
    assert diff["diff"]["to_snapshot"] == "cal-0003"
    # An unnamed side defaults to its positional partner: to
    # alone reads the report kept just before it as the from.
    diff = service.sla_calibration_snapshot_diff(to_label="mid-review")
    assert diff["diff"]["from_snapshot"] == "cal-0001"
    assert diff["diff"]["to_snapshot"] == "cal-0002"
    # Reversed selectors still walk in keep order.
    diff = service.sla_calibration_snapshot_diff(
        from_label="post-change", to_label="pre-change"
    )
    assert diff["diff"]["from_snapshot"] == "cal-0001"
    assert diff["diff"]["to_snapshot"] == "cal-0003"
    # The default read is untouched: the two latest.
    diff = service.sla_calibration_snapshot_diff()
    assert diff["diff"]["from_snapshot"] == "cal-0002"
    assert diff["diff"]["to_snapshot"] == "cal-0003"


def test_the_diff_refuses_unknown_and_self_naming_labels():
    service = _service()
    _kept_three(service)
    with pytest.raises(ValueError) as excinfo:
        service.sla_calibration_snapshot_diff(from_label="nope")
    message = str(excinfo.value)
    assert "nope" in message and "pre-change" in message
    with pytest.raises(ValueError):
        service.sla_calibration_snapshot_diff(
            from_label="mid-review", to_label="mid-review"
        )
    # One selector naming the latest report has no later side:
    # no diff, the same honesty as fewer than two kept.
    diff = service.sla_calibration_snapshot_diff(from_label="post-change")
    assert diff == {"snapshot_count": 3, "diff": None}


# --- the API -----------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_api_addresses_reports_by_label(client):
    kept = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    assert kept.status_code == 200
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "post-change"},
    )
    # The collision is refused, naming the existing labels.
    refused = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    assert refused.status_code == 422
    assert "cal-0001" in refused.json()["detail"]
    assert "post-change" in refused.json()["detail"]
    # The list filters; the report reads by its handle.
    listed = client.get(
        "/queue/sla/calibration/snapshots", params={"label": "post-change"}
    ).json()
    assert [s["snapshot_id"] for s in listed["snapshots"]] == ["cal-0002"]
    by_label = client.get(
        "/queue/sla/calibration/snapshots/by-label/pre-change"
    )
    assert by_label.status_code == 200
    assert by_label.json()["snapshot_id"] == "cal-0001"
    assert (
        client.get("/queue/sla/calibration/snapshots/by-label/nope").status_code
        == 404
    )
    # And the diff follows a citation.
    diff = client.get(
        "/queue/sla/calibration/snapshots/diff",
        params={"from_label": "pre-change", "to_label": "post-change"},
    ).json()
    assert diff["diff"]["from_label"] == "pre-change"
    assert diff["diff"]["to_label"] == "post-change"
    unknown = client.get(
        "/queue/sla/calibration/snapshots/diff",
        params={"from_label": "nope"},
    )
    assert unknown.status_code == 422
    assert "pre-change" in unknown.json()["detail"]
