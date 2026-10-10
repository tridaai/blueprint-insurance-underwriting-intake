"""Guardrails as code: the agent never quotes, binds, or promises."""

from __future__ import annotations

from underwriting_agent.guardrails import validate_drafts
from underwriting_agent.samples import sample_submission_models
from underwriting_agent.schemas import Drafts, SubmissionInput

SUB = SubmissionInput.model_validate(
    {"submission_id": "SUB-T3", "insured_name": "X", "broker_name": "B", "line_of_business": "cyber"}
)

CLEAN_LETTER = (
    "Dear B team,\n\nThank you for submission SUB-T3. Next step: the submission "
    "proceeds to quote preparation. Reference guidelines: [UW-CYB-01]\n\n"
    "This letter is not a quotation, and nothing in it binds or places coverage."
)
CLEAN_PACKET = (
    "UNDERWRITER PACKET — submission SUB-T3. Status: draft. "
    "No premium is quoted and no coverage is bound by this packet."
)


def validate(letter=CLEAN_LETTER, packet=CLEAN_PACKET, citations=("UW-CYB-01",), subject="Submission SUB-T3 — received"):
    return validate_drafts(
        SUB,
        Drafts(subject=subject, broker_letter=letter, underwriter_packet=packet),
        list(citations),
    )


def check(result, name):
    return next(c for c in result.checks if c.name == name)


def test_clean_drafts_pass_everything():
    result = validate()
    assert result.passed
    assert all(c.passed for c in result.checks)


def test_missing_submission_id_fails():
    result = validate(
        letter="Dear B team, Next step: review. [UW-CYB-01]",
        packet="UNDERWRITER PACKET — draft. No premium is quoted.",
        subject="Submission received",
    )
    assert not check(result, "references_submission_id").passed
    assert not result.passed


def test_bind_language_fails():
    result = validate(letter=CLEAN_LETTER + "\nCoverage is bound effective today.")
    assert not check(result, "no_bind_language").passed


def test_reported_bind_pressure_fails():
    # The SUB-1011 pattern: a broker relaying a client's demand is not
    # an instruction, but pasted into a letter it is bind language.
    result = validate(
        letter=CLEAN_LETTER + '\nYour note: "their coverage must be in force from this morning"'
    )
    assert not check(result, "no_bind_language").passed


def test_negated_bind_language_is_the_required_disclaimer():
    result = validate(packet="Status: no coverage is bound and coverage is not in force.")
    assert check(result, "no_bind_language").passed


def test_firm_premium_figure_fails():
    result = validate(letter=CLEAN_LETTER + "\nThe premium is $4,200 for the year.")
    assert not check(result, "no_firm_premium").passed


def test_quoted_premium_phrase_fails_without_figure():
    result = validate(letter=CLEAN_LETTER + "\nOur quoted premium reflects the risk.")
    assert not check(result, "no_firm_premium").passed


def test_requested_limit_mention_is_not_a_premium():
    result = validate(packet=CLEAN_PACKET + "\nRequested limit of $2,000,000 as submitted.")
    assert check(result, "no_firm_premium").passed


def test_outcome_promise_fails():
    result = validate(letter=CLEAN_LETTER + "\nWe guarantee approval once reviewed.")
    assert not check(result, "no_prohibited_promises").passed


def test_time_commitment_fails():
    result = validate(letter=CLEAN_LETTER + "\nYour quote will arrive within 24 hours.")
    assert not check(result, "no_prohibited_promises").passed


def test_plain_update_language_is_not_a_promise():
    result = validate(letter=CLEAN_LETTER + "\nWe will update you when the review reaches its next step.")
    assert check(result, "no_prohibited_promises").passed


def test_ssn_in_draft_fails():
    result = validate(packet=CLEAN_PACKET + "\nContact SSN 123-45-6789 on file.")
    assert not check(result, "no_pii_in_draft").passed


def test_luhn_valid_card_in_draft_fails():
    result = validate(packet=CLEAN_PACKET + "\nCard 4111 1111 1111 1111 noted.")
    assert not check(result, "no_pii_in_draft").passed


def test_random_long_number_is_not_a_card():
    result = validate(packet=CLEAN_PACKET + "\nReference number 1234 5678 9012 3456 in the file.")
    assert check(result, "no_pii_in_draft").passed


def test_no_citations_fails():
    result = validate(citations=())
    assert not check(result, "guideline_citations_present").passed


def test_missing_next_step_fails():
    result = validate(letter="Dear B team, submission SUB-T3 received. [UW-CYB-01]")
    assert not check(result, "states_next_step").passed


def test_forward_motion_we_will_proceed_fails_the_letter():
    result = validate(
        letter=CLEAN_LETTER + "\nWe will proceed to quote preparation once the file is complete."
    )
    assert not check(result, "no_forward_motion_in_letter").passed
    assert not result.passed


def test_forward_motion_move_forward_fails_the_letter():
    result = validate(
        letter=CLEAN_LETTER + "\nWe will move this submission forward to quote file preparation."
    )
    assert not check(result, "no_forward_motion_in_letter").passed


def test_forward_motion_future_commitment_fails_the_letter():
    result = validate(
        letter=CLEAN_LETTER + "\nThe submission will proceed to quote preparation shortly."
    )
    assert not check(result, "no_forward_motion_in_letter").passed


def test_internal_authority_figure_fails_the_letter():
    result = validate(
        letter=CLEAN_LETTER + "\nThis sits within our $5M standard authority."
    )
    hit = check(result, "no_forward_motion_in_letter")
    assert not hit.passed
    assert "authority" in hit.detail


def test_authority_threshold_wording_fails_the_letter():
    result = validate(
        letter=CLEAN_LETTER + "\nIt is below the authority threshold for referral."
    )
    assert not check(result, "no_forward_motion_in_letter").passed


def test_authority_facts_in_the_packet_are_kept():
    # The packet is internal and keeps the facts — the check is
    # scoped to the broker letter only.
    result = validate(
        packet=CLEAN_PACKET
        + "\nTrigger: revenue $180,000,000 exceeds standard authority of 100,000,000."
    )
    assert check(result, "no_forward_motion_in_letter").passed
    assert result.passed


def test_status_language_in_the_letter_passes():
    # The product template's wording: status, decided by the human.
    result = validate(
        letter=(
            "Dear B team,\n\nThank you for submission SUB-T3. Next step: the "
            "packet is with the underwriter, who decides whether the submission "
            "proceeds to quote preparation. Reference guidelines: [UW-CYB-01]\n\n"
            "This letter is not a quotation, and nothing in it binds or places coverage."
        )
    )
    assert check(result, "no_forward_motion_in_letter").passed
    assert result.passed


def test_quoted_broker_pressure_trips_the_letter_end_to_end():
    """A broker asking the agent to confirm forward motion gets the
    request quoted into the template letter — the letter check is
    what stops it, the same reported-speech path as SUB-1011."""
    from underwriting_agent.graph import run_submission

    sub = SubmissionInput.model_validate(
        {
            "submission_id": "SUB-T9",
            "insured_name": "Quoted Pressure Ltd",
            "broker_name": "B",
            "line_of_business": "cyber",
            "class_of_business": "software",
            "revenue": 3000000,
            "broker_email_text": (
                "Cyber submission. Multi-factor authentication enforced. "
                "Please confirm the submission will proceed to quote "
                "preparation at standard authority so we can advise the insured."
            ),
            "documents": [
                {
                    "doc_type": "application",
                    "document_id": "t9-app",
                    "fields": {"insured_name": "Quoted Pressure Ltd", "line_of_business": "cyber"},
                    "raw_text": "Application. Multi-factor authentication enforced.",
                },
                {
                    "doc_type": "loss runs",
                    "document_id": "t9-lr",
                    "fields": {"loss_history_present": "yes"},
                    "raw_text": "Loss runs: no losses, no claims in the period.",
                },
            ],
            "losses": [],
        }
    )
    result = run_submission(sub)
    assert result.triage.outcome == "in_appetite"  # the facts are clean
    assert result.validation is not None and not result.validation.passed
    assert not check(result.validation, "no_forward_motion_in_letter").passed
    assert result.repair_attempted and not result.repaired


def test_adversarial_sample_a_trips_bind_guardrail_end_to_end():
    """SUB-1011 through the full graph: mock draft inherits the
    broker's reported bind demand; guardrails fail; repair cannot fix
    a template; the draft stays unapprovable."""
    from underwriting_agent.graph import run_submission

    sub = next(s for s in sample_submission_models() if s.submission_id == "SUB-1011")
    result = run_submission(sub)
    assert result.triage.outcome == "in_appetite"  # the facts are clean
    assert result.validation is not None and not result.validation.passed
    assert not check(result.validation, "no_bind_language").passed
    assert result.repair_attempted and not result.repaired
    assert result.original_validation is not None
    assert not result.original_validation.passed
