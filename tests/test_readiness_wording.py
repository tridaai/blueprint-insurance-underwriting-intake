"""Readiness wording: the gaps must say WHERE they come from.

Regression coverage for the round-1 nit: with completeness 1.0 and
readiness ``ready_with_gaps`` (gaps = extraction cross-check
discrepancies, nothing missing), a live reviewer read the pair as
"incomplete documentation". The facts now carry ``gap_sources`` and
one canonical explanation sentence, and every surface that mentions
readiness — evidence, prompts, packet — quotes it.
"""

from __future__ import annotations

from underwriting_agent.diagnosis import build_evidence
from underwriting_agent.facts import compute_facts, readiness_explanation
from underwriting_agent.model_backends import DraftContext, MockModelBackend
from underwriting_agent.schemas import DocumentExtraction, FieldExtraction, SubmissionInput


def make_submission(**overrides) -> SubmissionInput:
    base = {
        "submission_id": "SUB-T9",
        "insured_name": "Test Insured",
        "broker_name": "Test Broker",
        "line_of_business": "cyber",
        "class_of_business": "consulting",
        "broker_email_text": "Submission attached. MFA is enforced.",
        "documents": [
            {
                "document_id": "app",
                "doc_type": "application",
                "fields": {"insured_name": "Test Insured", "revenue": "5000000"},
                "raw_text": "Application. Multi-factor authentication enforced.",
            },
            {
                "document_id": "lr",
                "doc_type": "loss runs",
                "fields": {"loss_history_present": "yes"},
                "raw_text": "No losses in the period.",
            },
        ],
        "revenue": 5_000_000,
        "years_in_business": 8,
        "losses": [],
    }
    base.update(overrides)
    return SubmissionInput.model_validate(base)


def _mismatch_extraction() -> DocumentExtraction:
    return DocumentExtraction(
        document_id="app",
        doc_type="acord_form",
        source="llm",
        fields=[
            FieldExtraction(field="revenue", value="6000000", status="mismatch")
        ],
    )


def test_gap_sources_name_discrepancies_when_nothing_is_missing():
    facts = compute_facts(make_submission(), [_mismatch_extraction()])
    assert facts.completeness_score == 1.0
    assert facts.missing_elements == []
    assert facts.readiness == "ready_with_gaps"
    assert len(facts.gap_sources) == 1
    assert facts.gap_sources[0].startswith("extraction cross-check discrepancy:")
    assert "revenue" in facts.gap_sources[0]
    assert not any(s.startswith("missing required element") for s in facts.gap_sources)


def test_gap_sources_name_missing_elements():
    facts = compute_facts(
        make_submission(documents=[], revenue=None, broker_email_text=""), []
    )
    assert facts.readiness == "not_ready"
    assert facts.gap_sources
    assert all(s.startswith("missing required element:") for s in facts.gap_sources)


def test_explanation_never_implies_missing_when_there_are_none():
    text = readiness_explanation(
        readiness="ready_with_gaps",
        completeness=1.0,
        missing=[],
        discrepancies=["app: field 'revenue' differs from the document text"],
    )
    assert "no required element is missing" in text
    assert "extraction cross-check discrepancy" in text
    assert "missing required element(s):" not in text


def test_explanation_names_missing_elements_when_present():
    text = readiness_explanation(
        readiness="not_ready",
        completeness=0.25,
        missing=["application", "revenue"],
        discrepancies=[],
    )
    assert "missing required element(s): application, revenue" in text
    assert "no required element is missing" not in text


def test_explanation_ready_case():
    text = readiness_explanation(
        readiness="ready", completeness=1.0, missing=[], discrepancies=[]
    )
    assert "all required elements are present" in text


def _draft_context(**overrides) -> DraftContext:
    context = DraftContext(
        submission_id="SUB-T9",
        insured_name="Test Insured",
        broker_name="Test Broker",
        line_of_business="cyber",
        class_of_business="consulting",
        outcome="borderline",
        risk_tier="moderate",
        confidence=0.9,
        rationale="test",
        referral_triggers=[],
        completeness=1.0,
        readiness="ready_with_gaps",
        missing=[],
        loss_summary="no losses",
        diagnosis_summary="summary",
        diagnosis_risk_factors=[],
        citations=["UW-GEN-01"],
        options=[],
        recommended_option_kind="proceed_to_quote_prep",
        broker_text="",
        discrepancies=["app: field 'revenue' differs from the document text"],
    )
    context.update(overrides)
    return context


def test_mock_packet_states_gap_sources():
    _subject, _letter, packet = MockModelBackend().draft_documents(_draft_context())
    assert "Missing: none" in packet
    assert "Readiness gaps:" in packet
    assert "no required element is missing" in packet
    assert "extraction cross-check discrepancy" in packet


def test_mock_packet_omits_gap_line_when_ready():
    _subject, _letter, packet = MockModelBackend().draft_documents(
        _draft_context(readiness="ready", discrepancies=[])
    )
    assert "Readiness gaps:" not in packet


def test_diagnosis_evidence_states_gap_sources():
    evidence = build_evidence(
        triage={"outcome": "borderline", "risk_tier": "moderate", "signals": [], "referral_triggers": []},
        facts={
            "completeness_score": 1.0,
            "readiness": "ready_with_gaps",
            "missing_elements": [],
            "discrepancies": ["app: field 'revenue' differs from the document text"],
            "loss_summary": {},
        },
        extractions=[],
        guidelines=[],
    )
    gap_lines = [line for line in evidence if line.startswith("readiness gaps (computed):")]
    assert len(gap_lines) == 1
    assert "no required element is missing" in gap_lines[0]
