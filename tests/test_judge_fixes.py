"""Fixes from the round-2 LLM-judge pack (see the self-QA log).

The pack flagged two real classes in provider-mode drafts:
forward-motion phrasing in broker letters ("we will move this
submission forward to quote preparation") judged as a prohibited
promise, and internal guideline figures (authority limits) leaking
into broker letters. The deterministic surfaces are pinned here:
the template next-step language reports status instead of motion,
the drafting prompt carries both hard lines, and the judge template
receives the full verified facts the drafter saw.
"""

from __future__ import annotations

from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.prompts import DRAFT_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore
from underwriting_agent.samples import load_sample_submissions

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}

FORWARD_MOTION = [
    "we will proceed",
    "we will move",
    "we will advance",
    "proceeds to quote preparation at standard authority",
    "moves to quote preparation",
    "standard authority",
]


def test_mock_broker_letter_reports_status_not_motion():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(SAMPLES["SUB-1001"])  # clean → proceed_to_quote_prep
    letter = result.drafts.broker_letter.lower()
    for phrase in FORWARD_MOTION:
        assert phrase not in letter, f"forward-motion/internal phrase leaked: {phrase}"
    assert "underwriter" in letter and "decides" in letter


def test_mock_letter_next_step_keeps_decision_with_the_underwriter():
    from underwriting_agent.model_backends import DraftContext

    context = DraftContext(
        submission_id="SUB-T1",
        insured_name="Insured",
        broker_name="Broker",
        line_of_business="cyber",
        class_of_business="consulting",
        outcome="in_appetite",
        risk_tier="low",
        confidence=0.95,
        rationale="clean",
        referral_triggers=[],
        completeness=1.0,
        readiness="ready",
        missing=[],
        loss_summary="no losses",
        diagnosis_summary="summary",
        diagnosis_risk_factors=[],
        citations=["UW-CYB-01"],
        options=[],
        recommended_option_kind="proceed_to_quote_prep",
        broker_text="",
        discrepancies=[],
    )
    _subject, letter, _packet = MockModelBackend().draft_documents(context)
    lowered = letter.lower()
    for phrase in FORWARD_MOTION:
        assert phrase not in lowered
    assert "who decides whether the submission proceeds to quote preparation" in lowered


def test_draft_prompt_carries_the_figure_and_motion_rules():
    assert "Guideline authority limits are internal" in DRAFT_SYSTEM_PROMPT
    assert "forward-motion" in DRAFT_SYSTEM_PROMPT
    assert "underwriter decides the next step" in DRAFT_SYSTEM_PROMPT


def test_judge_template_takes_the_full_verified_facts():
    rendered = JUDGE_USER_TEMPLATE.format(
        submission_id="SUB-1",
        insured_name="Insured",
        broker_name="Broker",
        line_of_business="cyber",
        class_of_business="consulting",
        requested_limit="5000000",
        requested_deductible="25000",
        revenue="20000000",
        years_in_business="10",
        outcome="in_appetite",
        risk_tier="low",
        completeness=1.0,
        readiness="ready",
        gap_sources="none",
        missing="none",
        discrepancies="none",
        loss_summary="none",
        citations="UW-CYB-01",
        guidelines="[UW-CYB-01] Appetite: ...",
        recommended_option="proceed",
        subject="s",
        broker_letter="l",
        underwriter_packet="p",
    )
    assert "Revenue: 20000000" in rendered
    assert "[UW-CYB-01] Appetite" in rendered
