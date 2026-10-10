"""End-to-end pipeline behaviour in the default (offline fallback) mode."""

from __future__ import annotations

import pytest

from underwriting_agent.events import CollectingSink
from underwriting_agent.graph import run_submission
from underwriting_agent.samples import sample_submission_models

SAMPLES = {s.submission_id: s for s in sample_submission_models()}

EXPECTED = {
    "SUB-1001": ("in_appetite", "low"),
    "SUB-1002": ("borderline", "high"),
    "SUB-1003": ("incomplete", "moderate"),
    "SUB-1004": ("out_of_appetite", "elevated"),
    "SUB-1005": ("borderline", "moderate"),
    "SUB-1006": ("borderline", "moderate"),
    "SUB-1007": ("incomplete", "moderate"),
    "SUB-1008": ("borderline", "moderate"),
    "SUB-1009": ("in_appetite", "low"),
    "SUB-1010": ("borderline", "moderate"),
    "SUB-1011": ("in_appetite", "low"),
    "SUB-1012": ("borderline", "high"),
}


@pytest.mark.parametrize("submission_id", sorted(EXPECTED))
def test_sample_triage_matrix(submission_id):
    result = run_submission(SAMPLES[submission_id])
    assert (result.triage.outcome, result.triage.risk_tier) == EXPECTED[submission_id]
    assert result.status == "awaiting_underwriter"
    assert result.external_action_taken is False
    assert result.drafts is not None
    assert result.trace  # the step trace is always populated


def test_every_sample_cites_retrieved_guidelines():
    for sub in SAMPLES.values():
        result = run_submission(sub)
        assert result.guidelines, sub.submission_id
        ids = {g.guideline_id for g in result.guidelines}
        assert ids  # citations in the packet come from retrieval


def test_recommended_options_follow_the_triage():
    assert _recommended("SUB-1001") == "proceed_to_quote_prep"
    assert _recommended("SUB-1002") == "refer_to_senior_underwriter"
    assert _recommended("SUB-1003") == "request_information"
    assert _recommended("SUB-1004") == "draft_decline_rationale"
    assert _recommended("SUB-1007") == "request_information"


def _recommended(submission_id: str) -> str:
    result = run_submission(SAMPLES[submission_id])
    return next(o.kind for o in result.options if o.recommended)


def test_incomplete_samples_compose_information_requests():
    for sid in ("SUB-1003", "SUB-1007"):
        result = run_submission(SAMPLES[sid])
        assert result.needs_information
        assert result.information_request is not None
        assert result.information_request.missing_items
        # The request itself never binds or quotes.
        assert "in force" not in result.information_request.message


def test_injection_sample_is_flagged_and_never_obeyed():
    result = run_submission(SAMPLES["SUB-1012"])
    assert result.injection_flags  # the SYSTEM: sentence was screened
    assert result.triage.outcome == "borderline"  # facts decided, not the injection
    assert _recommended("SUB-1012") == "refer_to_senior_underwriter"
    # The flagged sentence never reaches the drafts.
    assert "SYSTEM:" not in result.drafts.broker_letter
    assert "SYSTEM:" not in result.drafts.underwriter_packet
    assert result.validation.passed


def test_mock_telemetry_is_honest_about_no_model():
    result = run_submission(SAMPLES["SUB-1001"])
    telemetry = result.telemetry
    assert telemetry is not None
    assert telemetry.backend == "mock"
    assert telemetry.input_tokens is None  # no model ran
    assert telemetry.estimated_cost_usd is None
    assert telemetry.model_calls >= 1  # the template render is counted as a call


def test_autonomy_only_for_clean_in_appetite_runs():
    clean = run_submission(SAMPLES["SUB-1001"])
    assert clean.autonomy.eligible_for_auto_triage
    borderline = run_submission(SAMPLES["SUB-1002"])
    assert not borderline.autonomy.eligible_for_auto_triage
    trapped = run_submission(SAMPLES["SUB-1011"])
    assert not trapped.autonomy.eligible_for_auto_triage  # guardrails failed


def test_events_stream_covers_every_node():
    sink = CollectingSink()
    run_submission(SAMPLES["SUB-1001"], event_sink=sink)
    types = [e.type for e in sink.events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    nodes = {e.node for e in sink.events if e.type == "node_completed"}
    assert {"extract", "classify", "diagnose", "draft", "validate", "gate"} <= nodes
    assert all(e.duration_ms is not None for e in sink.events if e.type == "node_completed")


def test_trace_has_a_step_per_node_with_durations():
    result = run_submission(SAMPLES["SUB-1001"])
    names = [s.name for s in result.trace]
    assert names[0] == "extract"
    assert "approval_gate" in names
    assert all(s.duration_ms >= 0 for s in result.trace)


def test_readiness_flags_are_computed():
    assert run_submission(SAMPLES["SUB-1001"]).facts.readiness == "ready"
    assert run_submission(SAMPLES["SUB-1003"]).facts.readiness == "ready_with_gaps"
    assert run_submission(SAMPLES["SUB-1007"]).facts.readiness == "not_ready"
