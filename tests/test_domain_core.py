"""Domain core: intake normalisation, computed facts, triage rules."""

from __future__ import annotations

from underwriting_agent.facts import compute_facts, normalize_line
from underwriting_agent.schemas import SubmissionInput
from underwriting_agent.triage import classify_submission


def make_submission(**overrides) -> SubmissionInput:
    base = {
        "submission_id": "SUB-T1",
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


def test_doc_type_aliases_fold_to_vocabulary():
    sub = make_submission(
        documents=[
            {"document_id": "a", "doc_type": "ACORD Form", "fields": {}, "raw_text": ""},
            {"document_id": "b", "doc_type": "claims history", "fields": {}, "raw_text": ""},
            {"document_id": "c", "doc_type": "statement of values", "fields": {}, "raw_text": ""},
            {"document_id": "d", "doc_type": "financial statements", "fields": {}, "raw_text": ""},
        ]
    )
    assert [d.doc_type for d in sub.documents] == [
        "acord_form",
        "loss_run",
        "sov",
        "financials",
    ]


def test_unknown_doc_type_is_kept_and_flagged():
    sub = make_submission(
        documents=[{"document_id": "x", "doc_type": "drone survey", "fields": {}, "raw_text": ""}]
    )
    assert sub.documents[0].doc_type == "drone_survey"
    assert sub.documents[0].doc_type_flagged is True


def test_numeric_document_fields_coerce_to_strings():
    sub = make_submission(
        documents=[
            {
                "document_id": "a",
                "doc_type": "application",
                "fields": {"revenue": 5_000_000, "years_in_business": 8.0},
                "raw_text": "",
            }
        ]
    )
    assert sub.documents[0].fields == {"revenue": "5000000", "years_in_business": "8"}


def test_string_amounts_coerce_on_submission_and_losses():
    sub = make_submission(
        revenue="$12,500,000",
        losses=[{"date": "2024-01-01", "amount_incurred": "$45,000", "description": "x"}],
    )
    assert sub.revenue == 12_500_000
    assert sub.losses[0].amount_incurred == 45_000


def test_line_aliases():
    assert normalize_line("D&O") == "directors_and_officers"
    assert normalize_line("e&o") == "professional_liability"
    assert normalize_line("commercial property") == "property"


def test_facts_complete_clean_submission():
    sub = make_submission()
    facts = compute_facts(sub, [])
    assert facts.completeness_score == 1.0
    assert facts.readiness == "ready"
    assert facts.missing_elements == []
    assert facts.loss_summary.claim_count == 0
    assert facts.document_pair_warning == ""


def test_facts_loss_summary_math():
    sub = make_submission(
        losses=[
            {"date": "2023-02-01", "amount_incurred": 100_000, "description": "a"},
            {"date": "2025-02-01", "amount_incurred": 50_000, "description": "b"},
            {"date": "2025-06-01", "amount_incurred": 25_000, "description": "c"},
        ]
    )
    facts = compute_facts(sub, [])
    ls = facts.loss_summary
    assert ls.claim_count == 3
    assert ls.total_incurred == 175_000
    assert ls.largest_claim == 100_000
    assert ls.loss_years == [2023, 2025]
    assert ls.frequency_per_year == 1.0  # 3 claims over a 3-year span


def test_facts_missing_elements_and_readiness_bands():
    sub = make_submission(documents=[], revenue=None, broker_email_text="")
    facts = compute_facts(sub, [])
    # cyber requires application, revenue, loss_runs_or_no_loss, mfa_attestation
    assert set(facts.missing_elements) == {
        "application",
        "revenue",
        "loss_runs_or_no_loss",
        "mfa_attestation",
    }
    assert facts.readiness == "not_ready"
    assert facts.document_pair_warning != ""


def test_triage_out_of_appetite_class():
    sub = make_submission(class_of_business="hazardous waste transport")
    triage = classify_submission(sub, compute_facts(sub, []))
    assert triage.outcome == "out_of_appetite"
    assert triage.risk_tier == "elevated"


def test_triage_incomplete_when_not_ready():
    sub = make_submission(documents=[], revenue=None, broker_email_text="")
    triage = classify_submission(sub, compute_facts(sub, []))
    assert triage.outcome == "incomplete"


def test_triage_do_missing_financials_is_incomplete_even_with_gaps_only():
    sub = make_submission(
        line_of_business="directors_and_officers",
        documents=[
            {"document_id": "a", "doc_type": "application", "fields": {}, "raw_text": ""},
            {"document_id": "l", "doc_type": "loss runs", "fields": {}, "raw_text": "no losses"},
        ],
    )
    facts = compute_facts(sub, [])
    assert facts.readiness == "ready_with_gaps"  # 2 of 3 elements
    triage = classify_submission(sub, facts)
    assert triage.outcome == "incomplete"


def test_triage_referral_triggers_make_borderline():
    sub = make_submission(
        losses=[
            {"date": "2024-01-01", "amount_incurred": 300_000, "description": "big"},
            {"date": "2024-05-01", "amount_incurred": 10_000, "description": "x"},
            {"date": "2025-01-01", "amount_incurred": 10_000, "description": "y"},
        ]
    )
    triage = classify_submission(sub, compute_facts(sub, []))
    assert triage.outcome == "borderline"
    assert triage.risk_tier == "high"  # >= 3 claims
    assert len(triage.referral_triggers) >= 2


def test_triage_authority_triggers():
    sub = make_submission(revenue=150_000_000)
    triage = classify_submission(sub, compute_facts(sub, []))
    assert triage.outcome == "borderline"
    assert any("authority" in t for t in triage.referral_triggers)


def test_triage_clean_is_in_appetite_low():
    sub = make_submission()
    triage = classify_submission(sub, compute_facts(sub, []))
    assert (triage.outcome, triage.risk_tier) == ("in_appetite", "low")
