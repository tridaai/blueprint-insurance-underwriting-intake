"""AI layer: extraction cross-check, triage cross-check policy, retrieval."""

from __future__ import annotations

import pytest

from underwriting_agent.crosscheck import resolve_triage
from underwriting_agent.extractor import extract_documents, extraction_discrepancies
from underwriting_agent.retriever import KeywordRetriever, get_retriever, rerank_fused
from underwriting_agent.schemas import SubmissionInput, Triage


def make_submission() -> SubmissionInput:
    return SubmissionInput.model_validate(
        {
            "submission_id": "SUB-T2",
            "insured_name": "Acme Clinics",
            "broker_name": "Broker",
            "line_of_business": "cyber",
            "documents": [
                {
                    "document_id": "app",
                    "doc_type": "application",
                    "fields": {"insured_name": "Acme Clinics", "revenue": "9000000"},
                    "raw_text": "Application for Acme Clinics. Revenue is 7500000.",
                }
            ],
        }
    )


class ExtractBackend:
    name = "fake-llm"

    def extract_document_fields(self, doc_type, document_id, raw_text, fields):
        return {
            "insured_name": {"value": "Acme Clinics", "confidence": 0.97},
            "revenue": {"value": "7500000", "confidence": 0.88},
            "prior_carrier": {"value": "Old Carrier", "confidence": 0.7},
            "requested_limit": {"value": None, "confidence": 0.0},
        }


def test_extraction_cross_check_statuses():
    extractions = extract_documents(make_submission(), ExtractBackend())
    by_field = {f.field: f for f in extractions[0].fields}
    assert by_field["insured_name"].status == "match"
    assert by_field["revenue"].status == "mismatch"  # 9000000 provided vs 7500000 in text
    assert by_field["prior_carrier"].status == "extracted_only"
    assert extractions[0].source == "llm"
    assert extraction_discrepancies(extractions)  # the mismatch is reported


def test_extraction_missing_in_text():
    class PartialBackend:
        name = "fake-llm"

        def extract_document_fields(self, doc_type, document_id, raw_text, fields):
            return {f: {"value": None, "confidence": 0.0} for f in fields}

    extractions = extract_documents(make_submission(), PartialBackend())
    statuses = {f.field: f.status for f in extractions[0].fields}
    assert statuses["insured_name"] == "missing_in_text"
    assert statuses["revenue"] == "missing_in_text"


def test_extraction_without_llm_is_provided_passthrough():
    class NoExtract:
        name = "mock"

    extractions = extract_documents(make_submission(), NoExtract())
    assert extractions[0].source == "provided"
    assert all(f.status == "source_provided" for f in extractions[0].fields)


def test_extraction_failure_falls_back_with_note():
    class BoomBackend:
        name = "fake-llm"

        def extract_document_fields(self, *args):
            raise RuntimeError("provider down")

    extractions = extract_documents(make_submission(), BoomBackend())
    assert extractions[0].source == "provided"
    assert "failed" in extractions[0].note


def _rule(outcome="in_appetite", tier="low", confidence=0.94) -> Triage:
    return Triage(outcome=outcome, risk_tier=tier, confidence=confidence, rationale="rules")


def test_crosscheck_agree_keeps_rules():
    final, cross = resolve_triage(
        _rule(), {"outcome": "in_appetite", "risk_tier": "moderate", "confidence": 0.9, "rationale": "llm"}
    )
    assert final.outcome == "in_appetite" and final.risk_tier == "low"
    assert cross.resolution == "agree"
    assert "tier differs" in cross.note


def test_crosscheck_disagreement_rules_authoritative():
    final, cross = resolve_triage(
        _rule(confidence=0.94),
        {"outcome": "borderline", "risk_tier": "high", "confidence": 0.99, "rationale": "llm"},
    )
    assert final.outcome == "in_appetite"
    assert cross.resolution == "rules_authoritative"


def test_crosscheck_llm_adopted_when_rules_weak():
    final, cross = resolve_triage(
        _rule(outcome="incomplete", confidence=0.6),
        {"outcome": "borderline", "risk_tier": "high", "confidence": 0.9, "rationale": "litigation in text"},
    )
    assert final.outcome == "borderline"
    assert cross.resolution == "llm_adopted"
    assert any("llm-adopted" in s for s in final.signals)


def test_crosscheck_no_llm_is_rules_only():
    final, cross = resolve_triage(_rule(), None)
    assert final.outcome == "in_appetite"
    assert cross.resolution == "rules_only"


def test_keyword_retriever_ranks_relevant_guideline_first():
    results = KeywordRetriever().retrieve(
        "cyber ransomware referral senior underwriter multi-factor", top_k=3
    )
    assert results
    assert results[0].guideline_id == "UW-CYB-02"


def test_hybrid_rerank_fuses_sources():
    kw = KeywordRetriever().retrieve("authority revenue limits referral", top_k=4)
    fused = rerank_fused(kw, kw, top_k=2)
    assert len(fused) == 2
    assert all(g.sources for g in fused)


def test_get_retriever_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown RETRIEVER"):
        get_retriever("bogus")
