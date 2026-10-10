"""Multi-line submissions: one arrival, per-line analyses, one gate.

A broker bundles lines (property + cyber) on one account. Intake
splits the bundle into per-line sub-analyses sharing the insured
context — each with its own triage and readiness — and rolls them
into one combined packet with per-line sections. There is one
underwriter gate, on the bundle; renewal diffs are keyed per line.
"""

from __future__ import annotations

import copy

import pytest

from underwriting_agent.bundles import is_bundle, split_bundle
from underwriting_agent.samples import load_sample_submissions, sample_submission_models
from underwriting_agent.schemas import SubmissionInput
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
MODELS = {s.submission_id: s for s in sample_submission_models()}


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_split_shares_context_and_keeps_line_specifics():
    model = MODELS["SUB-1014"]
    assert is_bundle(model)
    subs = split_bundle(model)
    assert [s.submission_id for s in subs] == ["SUB-1014-PROPERTY", "SUB-1014-CYBER"]
    prop, cyber = subs
    for sub in subs:  # the insured context is shared
        assert sub.insured_name == "Willow Creek Hospitality Group"
        assert sub.broker_name == "Meridian Wholesale Brokers"
        assert sub.revenue == 42_000_000  # bundle-level fallback
        assert sub.prior_carrier == "Sentinel Mutual"
    # Line specifics stay with their line.
    assert prop.requested_limit == "4500000" and prop.losses == []
    assert cyber.requested_limit == "3000000" and len(cyber.losses) == 2
    # The shared application reaches every line; line documents do not leak.
    assert any(d.document_id == "sub1014-app" for d in prop.documents)
    assert any(d.document_id == "sub1014-app" for d in cyber.documents)
    assert any(d.document_id == "sub1014-prop-sov" for d in prop.documents)
    assert not any(d.document_id == "sub1014-prop-sov" for d in cyber.documents)
    assert not any(d.document_id == "sub1014-cyb-lr" for d in prop.documents)


def test_single_line_submission_is_not_a_bundle():
    assert not is_bundle(MODELS["SUB-1001"])


# ---------------------------------------------------------------------------
# The bundle run: divergent lines, one roll-up, one record, one gate
# ---------------------------------------------------------------------------


def test_bundle_lines_triage_independently_and_roll_up_to_the_worst():
    result = make_service().analyze(SAMPLES["SUB-1014"])
    assert result.multi_line is True
    assert result.line_of_business == "multi-line (property + cyber)"
    by_line = {a.line_of_business: a.result for a in result.line_analyses}
    # Divergent outcomes — the whole point of splitting.
    assert by_line["property"].triage.outcome == "in_appetite"
    assert by_line["property"].triage.risk_tier == "low"
    assert by_line["property"].facts.readiness == "ready"
    assert by_line["cyber"].triage.outcome == "borderline"
    assert by_line["cyber"].triage.risk_tier == "high"
    assert any(
        "ransomware" in t for t in by_line["cyber"].triage.referral_triggers
    )
    # The bundle follows its most severe line.
    assert result.triage.outcome == "borderline"
    assert result.triage.risk_tier == "high"
    assert any(
        t.startswith("[cyber]") for t in result.triage.referral_triggers
    )
    # The combined drafts passed the same guardrails as any draft.
    assert result.validation is not None and result.validation.passed


def test_bundle_packet_has_a_section_per_line_and_the_letter_stays_clean():
    result = make_service().analyze(SAMPLES["SUB-1014"])
    packet = result.drafts.underwriter_packet
    assert "multi-line bundle" in packet
    assert "===== Line: property" in packet and "===== Line: cyber" in packet
    assert "Per-line outcomes:" in packet
    letter = result.drafts.broker_letter
    assert "SUB-1014" in letter and "property" in letter and "cyber" in letter
    assert "Next step:" in letter


def test_bundle_is_one_record_with_one_gate():
    service = make_service()
    service.analyze(SAMPLES["SUB-1014"])
    records = service._get_store().all_records()
    assert [r.result.submission_id for r in records] == ["SUB-1014"]
    # Sub-analyses are packet sections, not separately decidable records.
    assert service.get("SUB-1014-PROPERTY") is None
    with pytest.raises(KeyError):
        service.approve("SUB-1014-CYBER", approver="U")
    decided = service.approve("SUB-1014", approver="U", reason="bundle accepted")
    assert decided.status == "approved"
    # And the bundle appears once in the inbox/scorecards, as itself.
    inbox = service.inbox()
    group = next(
        g for g in inbox["insureds"] if g["name"] == "Willow Creek Hospitality Group"
    )
    assert group["submission_count"] == 1


def test_bundle_via_api_and_approve(client):
    posted = client.post("/submissions/analyze", json=SAMPLES["SUB-1014"])
    assert posted.status_code == 200
    body = posted.json()
    assert body["multi_line"] is True
    assert len(body["line_analyses"]) == 2
    decided = client.post(
        "/submissions/SUB-1014/approve", json={"actor": "U. Writer"}
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"


def test_change_impact_report_retriages_bundles_per_line():
    # Round 5 counted bundles as skipped; round 6 re-triages them
    # per line instead — still never a flattened pass over the
    # bundle payload, and still nothing stored is rewritten.
    service = make_service()
    service.analyze(SAMPLES["SUB-1014"])
    report = service.guideline_change_impact()
    assert report["checked"] == 1 and report["skipped"] == 0
    assert report["lines_checked"] == 2
    assert report["moved"] == []


# ---------------------------------------------------------------------------
# Diffs keyed per line
# ---------------------------------------------------------------------------


def _bundle_renewal() -> dict:
    renewal = copy.deepcopy(SAMPLES["SUB-1014"])
    renewal["submission_id"] = "SUB-1015"
    renewal["submitted_at"] = "2026-10-09"
    renewal["lines"][0]["requested_limit"] = "5400000"  # property +20%
    return renewal


def test_renewal_diffs_are_keyed_per_line():
    service = make_service()
    service.analyze(SAMPLES["SUB-1014"])
    result = service.analyze(_bundle_renewal())
    by_line = {a.line_of_business: a.result for a in result.line_analyses}

    prop_diff = by_line["property"].submission_diff
    assert prop_diff is not None
    assert prop_diff.prior_submission_id == "SUB-1014-PROPERTY"
    changed = {c.field for c in prop_diff.changes}
    assert changed == {"requested_limit"}

    cyber_diff = by_line["cyber"].submission_diff
    assert cyber_diff is not None
    assert cyber_diff.prior_submission_id == "SUB-1014-CYBER"
    assert cyber_diff.changes == []  # the cyber line did not move


def test_single_line_renewal_diffs_against_the_bundles_matching_line():
    service = make_service()
    service.analyze(SAMPLES["SUB-1014"])
    single = {
        "submission_id": "SUB-1016",
        "insured_name": "Willow Creek Hospitality Group",
        "broker_name": "Meridian Wholesale Brokers",
        "line_of_business": "property",
        "class_of_business": "hotels and lodging",
        "submitted_at": "2026-10-10",
        "requested_limit": "4500000",
        "requested_deductible": "25000",
        "documents": SAMPLES["SUB-1014"]["lines"][0]["documents"],
        "losses": [],
    }
    result = service.analyze(single)
    assert result.submission_diff is not None
    assert result.submission_diff.prior_submission_id == "SUB-1014-PROPERTY"


def test_bundle_result_survives_a_store_round_trip():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1014"])
    stored = service.get("SUB-1014")
    assert stored is not None and stored.multi_line is True
    assert [a.line_of_business for a in stored.line_analyses] == ["property", "cyber"]
    cyber = stored.line_analyses[1].result
    assert cyber.triage.outcome == "borderline"
    # A validated result object: the nested results re-validate cleanly.
    assert SubmissionInput.model_validate(
        store.get("SUB-1014") and service._get_store().get("SUB-1014").submission
    )
