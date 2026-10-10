"""Round 7: diff-trigger calibration — replay the stored renewals.

The referral-trigger thresholds (limit-jump %, new-claim floor,
development floor) decide how much renewal movement reaches an
underwriter. The calibration surface rebuilds every stored
renewal pair, replays the trigger computation at candidate
thresholds one knob at a time, and sets the what-if next to what
actually fired at analysis time and what underwriters decided.
Reading it changes nothing.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.calibration import (
    renewal_pairs,
    trigger_calibration,
    trigger_family,
)
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


def developed_renewal() -> dict:
    """A renewal of SUB-1002 whose open claim developed adversely:
    the 2025-09-30 claim's incurred rises $12,750 -> $80,000
    (+$67,250, past the $50k development floor). Everything else
    about the account is unchanged."""
    sub = copy.deepcopy(SAMPLES["SUB-1002"])
    sub["submission_id"] = "SUB-2002"
    sub["submitted_at"] = "2026-10-09"
    for loss in sub["losses"]:
        if loss["date"] == "2025-09-30":
            loss["amount_incurred"] = 80000
    return sub


@pytest.fixture()
def service() -> SubmissionService:
    svc = SubmissionService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["SUB-1001"])   # Northgate cyber, original
    svc.analyze(SAMPLES["SUB-1013"])   # its renewal: limit +50%, one $18.5k new claim
    svc.analyze(SAMPLES["SUB-1002"])   # Bluepeak property, original
    svc.analyze(developed_renewal())   # its renewal: adverse development
    return svc


def _sweep(report: dict, dimension: str) -> dict:
    return {row["value"]: row for row in next(
        s["rows"] for s in report["sweeps"] if s["dimension"] == dimension
    )}


def test_trigger_family_names_the_four_families():
    assert trigger_family("x (referral trigger: renewal diff — new claims)") == "new_claims"
    assert trigger_family("x (referral trigger: renewal diff — adverse loss development)") == "adverse_development"
    assert trigger_family("x (referral trigger: renewal diff — limit jump)") == "limit_jump"
    assert trigger_family("x (referral trigger: renewal diff — class change)") == "class_change"
    assert trigger_family("some other referral reason") is None


def test_renewal_pairs_chain_per_insured_and_line(service):
    pairs = renewal_pairs(service._get_store().all_records())
    assert {p["submission_id"] for p in pairs} == {"SUB-1013", "SUB-2002"}
    by_id = {p["submission_id"]: p for p in pairs}
    assert by_id["SUB-1013"]["prior"]["submission_id"] == "SUB-1001"
    assert by_id["SUB-2002"]["prior"]["submission_id"] == "SUB-1002"


def test_calibration_at_current_thresholds(service):
    report = trigger_calibration(service._get_store().all_records())
    assert report["renewals"] == 2
    assert report["current_thresholds"] == {
        "limit_jump_pct": 50.0,
        "new_claim_min_incurred": 50000.0,
        "development_min_increase": 50000.0,
    }
    # At the configured thresholds only the developed renewal refers:
    # SUB-1013's +50% limit jump is not > 50%, and its $18.5k new
    # claim is under the $50k floor — by design (round 4).
    assert report["at_current"] == {
        "referrals": 1,
        "family_counts": {"adverse_development": 1},
    }
    # The replay agrees with what the analyses themselves recorded.
    recorded = report["recorded_at_analysis"]
    assert recorded["referrals"] == 1
    assert recorded["family_counts"] == {"adverse_development": 1}
    assert recorded["submission_ids"] == ["SUB-2002"]
    assert recorded["by_status"] == {"awaiting_underwriter": 1}


def test_calibration_sweeps_move_one_knob_at_a_time(service):
    report = trigger_calibration(service._get_store().all_records())
    # Loosening the limit-jump knob to 25% catches SUB-1013's +50%.
    limit_rows = _sweep(report, "limit_jump_pct")
    assert limit_rows[25.0]["referrals"] == 2
    assert limit_rows[25.0]["family_counts"].get("limit_jump") == 1
    assert limit_rows[25.0]["delta_vs_current"] == 1
    assert limit_rows[50.0]["is_current"] is True
    assert limit_rows[100.0]["referrals"] == 1  # the development case still refers
    # Dropping the new-claim floor to $10k catches the $18.5k claim.
    claim_rows = _sweep(report, "new_claim_min_incurred")
    assert claim_rows[10000.0]["referrals"] == 2
    assert claim_rows[10000.0]["family_counts"].get("new_claims") == 1
    # Raising the development floor past the +$67,250 development
    # silences the only current referral.
    dev_rows = _sweep(report, "development_min_increase")
    assert dev_rows[100000.0]["referrals"] == 0
    assert dev_rows[100000.0]["delta_vs_current"] == -1


def test_calibration_is_read_only(service):
    store = service._get_store()
    before = [
        (r.result.submission_id, list(r.result.triage.referral_triggers), r.result.status)
        for r in store.all_records()
    ]
    service.diff_trigger_calibration()
    after = [
        (r.result.submission_id, list(r.result.triage.referral_triggers), r.result.status)
        for r in store.all_records()
    ]
    assert before == after


def test_calibration_empty_store():
    report = trigger_calibration([])
    assert report["renewals"] == 0
    assert report["at_current"]["referrals"] == 0
    assert report["recorded_at_analysis"]["referrals"] == 0
    for sweep in report["sweeps"]:
        assert all(row["referrals"] == 0 for row in sweep["rows"])


def test_recorded_decisions_follow_the_moved_cases(service):
    service.approve("SUB-2002", approver="U. Writer", reason="development explained")
    report = service.diff_trigger_calibration()
    assert report["recorded_at_analysis"]["by_status"] == {"approved": 1}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    import underwriting_agent.api as api_module
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    return TestClient(api_module.app)


def test_calibration_endpoint(client):
    for sid in ("SUB-1001", "SUB-1013"):
        client.post("/submissions/analyze", json=SAMPLES[sid])
    response = client.get("/guidelines/diff-trigger-calibration")
    assert response.status_code == 200
    body = response.json()
    assert body["renewals"] == 1
    assert body["at_current"]["referrals"] == 0  # SUB-1013 sits under the thresholds
    assert {s["dimension"] for s in body["sweeps"]} == {
        "limit_jump_pct",
        "new_claim_min_incurred",
        "development_min_increase",
    }
