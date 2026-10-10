"""Round 18: diff against the live report.

Kept calibration reports answer "what moved between march and
june"; the debates' commonest question is "what moved since
'march'?" against the report as it stands *now*, and until now
answering it meant keeping a throwaway snapshot just to diff
against. ``to_label=current`` is a reserved selector: the newer
side is the live calibration report, shaped in memory in the
snapshot's own shape and cited as ``current`` — nothing is
stored, and the movement sentence anchors on the kept side in
the same register. "current" is refused as a keepable label (a
citation must name a kept report, not a moving target) and as
a from_label (the live report is the newest side there is).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import (
    CURRENT_CALIBRATION_LABEL,
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


# --- the shaper ------------------------------------------------------------------


def test_the_live_snapshot_is_shaped_like_a_keep_and_cited_current():
    report = sla_budget_calibration(
        [_decided("M-2", "moderate", 2)],
        {"elevated": 2, "high": 3, "moderate": 5, "low": 7},
    )
    shaped = live_calibration_snapshot(report)
    assert shaped["snapshot_id"] == CURRENT_CALIBRATION_LABEL == "current"
    assert shaped["label"] == "current"
    assert shaped["taken_by"] == ""  # nobody kept it — nobody to name
    assert shaped["decided_cases_weighed"] == 1
    moderate = next(t for t in shaped["tiers"] if t["tier"] == "moderate")
    assert moderate["decided_cases"] == 1
    assert moderate["budget_in_force"] == 5


# --- keeping ---------------------------------------------------------------------


def test_current_is_refused_as_a_keepable_label():
    service = _service()
    with pytest.raises(ValueError, match="reserved"):
        service.snapshot_sla_calibration("Ops Lead", label="current")
    assert service.sla_calibration_snapshots() == []
    # And the by-label read names nothing for it, as for any
    # label no kept report carries.
    assert service.sla_calibration_snapshot_by_label("current") is None


# --- the diff ----------------------------------------------------------------------


def test_a_diff_against_current_cites_the_live_side_and_stores_nothing():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    result = service.sla_calibration_snapshot_diff(to_label="current")
    assert result["snapshot_count"] == 1
    diff = result["diff"]
    assert diff is not None
    assert diff["from_snapshot"] == "cal-0001"
    assert diff["from_label"] == "march"
    assert diff["to_snapshot"] == "current"
    assert diff["to_label"] == "current"
    # Evidence has not moved since the keep: the movement
    # sentence anchors on the kept side and says the rows held.
    assert diff["movement"].startswith("Since 'march':")
    assert "no proposal-ready movement" in diff["movement"]
    # The diff kept nothing: the throwaway snapshot the old
    # workflow needed never exists.
    assert len(service.sla_calibration_snapshots()) == 1


def test_the_current_diff_reads_the_live_evidence_not_the_kept_one():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    # Decide another moderate case after the keep: the live
    # report weighs two cases where the kept one weighed one.
    service._get_store().save(_decided("M-9", "moderate", 4))
    diff = service.sla_calibration_snapshot_diff(to_label="current")[
        "diff"
    ]
    assert diff["tiers"]["moderate"]["decided_cases"] == {
        "from": 1,
        "to": 2,
        "change": 1,
    }
    # The kept report is untouched by having been read.
    kept = service.sla_calibration_snapshots()[0]
    moderate = next(t for t in kept["tiers"] if t["tier"] == "moderate")
    assert moderate["decided_cases"] == 1


def test_the_current_diff_names_its_kept_side_by_label_or_latest():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    service.snapshot_sla_calibration("Ops Lead", label="april")
    by_label = service.sla_calibration_snapshot_diff(
        from_label="march", to_label="current"
    )["diff"]
    assert by_label["from_snapshot"] == "cal-0001"
    # Unnamed, the kept side is the latest kept — current is
    # the newer side, so its partner is the newest keep.
    unnamed = service.sla_calibration_snapshot_diff(to_label="current")[
        "diff"
    ]
    assert unnamed["from_snapshot"] == "cal-0002"
    assert unnamed["from_label"] == "april"


def test_no_kept_report_is_no_diff_even_against_current():
    service = _service()
    assert service.sla_calibration_snapshot_diff(to_label="current") == {
        "snapshot_count": 0,
        "diff": None,
    }


def test_current_as_the_from_side_is_refused():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    with pytest.raises(ValueError, match="only be a diff's 'to' side"):
        service.sla_calibration_snapshot_diff(
            from_label="current", to_label="march"
        )


def test_an_unknown_from_label_against_current_names_the_labels_in_use():
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="march")
    with pytest.raises(ValueError) as excinfo:
        service.sla_calibration_snapshot_diff(
            from_label="february", to_label="current"
        )
    message = str(excinfo.value)
    assert "february" in message
    assert "march" in message


# --- the API ------------------------------------------------------------------------


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def test_api_diffs_against_current_and_refuses_the_reserved_label(client):
    kept = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "march"},
    )
    assert kept.status_code == 200
    response = client.get(
        "/queue/sla/calibration/snapshots/diff",
        params={"to_label": "current"},
    )
    assert response.status_code == 200
    diff = response.json()["diff"]
    assert diff["to_snapshot"] == "current"
    assert diff["to_label"] == "current"
    assert diff["from_label"] == "march"

    refused = client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "current"},
    )
    assert refused.status_code == 422
    assert "reserved" in refused.json()["detail"]

    backwards = client.get(
        "/queue/sla/calibration/snapshots/diff",
        params={"from_label": "current", "to_label": "march"},
    )
    assert backwards.status_code == 422

    by_label = client.get(
        "/queue/sla/calibration/snapshots/by-label/current"
    )
    assert by_label.status_code == 404
