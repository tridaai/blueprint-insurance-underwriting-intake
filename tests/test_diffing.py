"""Renewal / resubmission diffing: what changed since last time.

Underwriters price the change, not the snapshot. The diff is
computed in code from the insured's most recent stored submission
and both runs' computed facts, and must land in three places: the
result, the diagnosis evidence, and the packet's changes section.
"""

from __future__ import annotations

from underwriting_agent.diffing import (
    compute_submission_diff,
    diff_evidence_lines,
    diff_packet_lines,
)
from underwriting_agent.extractor import extract_documents
from underwriting_agent.facts import compute_facts
from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.samples import load_sample_submissions, sample_submission_models
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
MODELS = {s.submission_id: s for s in sample_submission_models()}


def prior_bundle(submission_id: str, outcome: str = "in_appetite", tier: str = "low") -> dict:
    """The service's prior bundle, rebuilt the way the service builds it."""
    model = MODELS[submission_id]
    facts = compute_facts(model, extract_documents(model, MockModelBackend()))
    return {
        "submission_id": submission_id,
        "submitted_at": model.submitted_at,
        "outcome": outcome,
        "risk_tier": tier,
        "submission": model.model_dump(mode="json"),
        "facts": facts.model_dump(mode="json"),
    }


def fields_of(diff) -> set[str]:
    return {c.field for c in diff.changes}


def test_diff_reports_the_renewal_changes():
    current = MODELS["SUB-1013"]
    facts = compute_facts(current, extract_documents(current, MockModelBackend()))
    diff = compute_submission_diff(current, facts, prior_bundle("SUB-1001"))
    assert diff is not None
    assert diff.prior_submission_id == "SUB-1001"
    assert fields_of(diff) == {
        "requested_limit",
        "revenue",
        "years_in_business",
        "effective_date_requested",
        "loss_history",
    }
    by_field = {c.field: c for c in diff.changes}
    assert by_field["requested_limit"].previous == "$2,000,000"
    assert by_field["requested_limit"].current == "$3,000,000"
    assert "increased" in by_field["requested_limit"].note
    assert "requested_deductible" not in fields_of(diff)  # unchanged -> absent
    assert "+29.2%" in by_field["revenue"].note
    assert "1 new claim(s)" in by_field["loss_history"].note


def test_diff_of_identical_resubmission_is_empty_but_real():
    current = MODELS["SUB-1001"]
    facts = compute_facts(current, extract_documents(current, MockModelBackend()))
    diff = compute_submission_diff(current, facts, prior_bundle("SUB-1001"))
    assert diff is not None
    assert diff.changes == []
    assert any("no material changes" in line for line in diff_packet_lines(diff))
    assert any("no material changes" in line for line in diff_evidence_lines(diff))


def test_diff_handles_unusable_prior_payload():
    current = MODELS["SUB-1013"]
    facts = compute_facts(current, extract_documents(current, MockModelBackend()))
    assert compute_submission_diff(current, facts, {}) is None
    assert compute_submission_diff(current, facts, {"submission": {"bogus": 1}}) is None


def test_diff_flags_class_and_line_changes():
    changed = MODELS["SUB-1013"].model_copy(
        update={"class_of_business": "law firm — cyber consulting", "line_of_business": "property"}
    )
    facts = compute_facts(changed, extract_documents(changed, MockModelBackend()))
    diff = compute_submission_diff(changed, facts, prior_bundle("SUB-1001"))
    assert {"class_of_business", "line_of_business"} <= fields_of(diff)


def test_service_diff_lands_on_result_packet_and_evidence():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    first = service.analyze(SAMPLES["SUB-1001"])
    assert first.submission_diff is None  # nothing to diff against yet

    renewal = service.analyze(SAMPLES["SUB-1013"])
    diff = renewal.submission_diff
    assert diff is not None and diff.prior_submission_id == "SUB-1001"
    assert "requested_limit" in fields_of(diff)

    packet = renewal.drafts.underwriter_packet
    assert "Changes since last submission (SUB-1001" in packet
    assert "Requested limit: $2,000,000 -> $3,000,000" in packet
    evidence = "\n".join(renewal.diagnosis.evidence)
    assert "renewal diff:" in evidence
    assert "requested limit $2,000,000 -> $3,000,000" in evidence


def test_diff_uses_the_most_recent_prior_not_the_first():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1013"])
    third = dict(SAMPLES["SUB-1013"])
    third["submission_id"] = "SUB-1014"
    third["revenue"] = 17000000
    result = service.analyze(third)
    assert result.submission_diff is not None
    assert result.submission_diff.prior_submission_id == "SUB-1013"
    revenue = next(c for c in result.submission_diff.changes if c.field == "revenue")
    assert revenue.previous == "$15,500,000" and revenue.current == "$17,000,000"


def test_first_submission_for_an_insured_has_no_diff():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(SAMPLES["SUB-1009"])  # different insured entirely
    assert result.submission_diff is None
    assert "Changes since last submission" not in result.drafts.underwriter_packet


def test_standalone_run_has_no_diff():
    from underwriting_agent.graph import run_submission

    result = run_submission(MODELS["SUB-1013"])
    assert result.submission_diff is None


def test_diff_survives_store_round_trip():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1013"])
    stored = service.get("SUB-1013")
    assert stored is not None and stored.submission_diff is not None
    assert stored.submission_diff.prior_submission_id == "SUB-1001"


# ---------------------------------------------------------------------------
# Diff-driven referral triggers: the diff moves triage
# ---------------------------------------------------------------------------

from underwriting_agent.diffing import apply_diff_triggers, diff_referral_triggers
from underwriting_agent.schemas import LossRecord
from underwriting_agent.triage import classify_submission


def _facts_for(model):
    return compute_facts(model, extract_documents(model, MockModelBackend()))


def test_new_claims_over_the_severity_floor_trigger():
    prior_model = MODELS["SUB-1001"]  # no losses
    current = prior_model.model_copy(
        update={
            "submission_id": "SUB-9001",
            "losses": [
                LossRecord(
                    date="2026-08-15",
                    amount_incurred=120000,
                    description="Business email compromise — wire fraud, funds recovered in part",
                    status="closed",
                )
            ],
        }
    )
    triggers = diff_referral_triggers(
        current, _facts_for(current), prior_bundle("SUB-1001")
    )
    assert len(triggers) == 1
    assert "1 new claim(s)" in triggers[0] and "$120,000" in triggers[0]
    assert "renewal diff" in triggers[0]


def test_limit_jump_and_class_change_trigger():
    prior_model = MODELS["SUB-1001"]  # limit $2,000,000, class "law firm"
    current = prior_model.model_copy(
        update={
            "submission_id": "SUB-9002",
            "requested_limit": "5000000",
            "class_of_business": "legal technology services",
        }
    )
    triggers = diff_referral_triggers(
        current, _facts_for(current), prior_bundle("SUB-1001")
    )
    assert any("limit increased 150%" in t for t in triggers)
    assert any("class of business changed" in t for t in triggers)


def test_sample_renewal_pair_stays_under_the_thresholds():
    # SUB-1013 vs SUB-1001: one new claim at $18,500 (under the
    # severity floor) and a limit move of exactly +50% (not beyond
    # it) — real movement, deliberately not a diff referral.
    current = MODELS["SUB-1013"]
    triggers = diff_referral_triggers(
        current, _facts_for(current), prior_bundle("SUB-1001")
    )
    assert triggers == []


def test_apply_diff_triggers_escalates_a_clean_triage():
    sub = MODELS["SUB-1001"]
    triage = classify_submission(sub, _facts_for(sub))
    assert triage.outcome == "in_appetite"
    adjusted = apply_diff_triggers(triage, ["synthetic diff trigger"])
    assert adjusted.outcome == "borderline"
    assert adjusted.risk_tier == "moderate"
    assert "synthetic diff trigger" in adjusted.referral_triggers
    assert "Renewal diff referral" in adjusted.rationale
    assert triage.outcome == "in_appetite"  # the input is not mutated


def test_apply_diff_triggers_leaves_incomplete_alone():
    sub = MODELS["SUB-1003"]  # D&O, missing financials -> incomplete
    triage = classify_submission(sub, _facts_for(sub))
    assert triage.outcome == "incomplete"
    assert apply_diff_triggers(triage, ["synthetic diff trigger"]) is triage


def test_renewal_is_referred_on_the_change_end_to_end():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    renewal = dict(SAMPLES["SUB-1001"])
    renewal["submission_id"] = "SUB-9003"
    renewal["submitted_at"] = "2026-10-09"
    renewal["losses"] = [
        {
            "date": "2026-08-15",
            "amount_incurred": 120000,
            "description": "Business email compromise — wire fraud, funds recovered in part",
            "status": "closed",
        }
    ]
    result = service.analyze(renewal)
    # The static rules alone see one moderate claim and no trigger;
    # the referral exists because the loss is NEW since the prior.
    assert result.triage.outcome == "borderline"
    assert any("renewal diff" in t for t in result.triage.referral_triggers)
    assert result.submission_diff is not None
