"""Options node logic: next-step proposals, scored deterministically.

The four option kinds are the only paths a submission can take from
triage (``schemas.RecoveryOption.kind``):

- ``proceed_to_quote_prep`` — in appetite and complete: the packet
  goes to quote preparation under standard authority.
- ``refer_to_senior_underwriter`` — a referral trigger or tier demands
  senior eyes before anything proceeds.
- ``request_information`` — the submission cannot be assessed until
  the broker supplies what is missing.
- ``draft_decline_rationale`` — outside appetite: draft the decline
  rationale FOR the underwriter. Drafting a rationale never declines
  anything; the underwriter decides.

In provider mode the LLM may propose the wording of the options; in
every mode their *kinds* are validated here and their scores are
computed by deterministic code (the model never does this arithmetic):

    score = 60 * fit + 40 * readiness_fit - tier_penalty - loss_penalty

where ``fit`` ∈ [0, 1] is how well the kind matches the triage
outcome, ``readiness_fit`` how well it matches the computed
data-readiness flag, ``tier_penalty`` grows with the risk tier for
proceeding without referral, and ``loss_penalty`` grows with claim
frequency. The recommended option is the argmax.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import RecoveryOption

OPTION_KINDS = {
    "proceed_to_quote_prep",
    "refer_to_senior_underwriter",
    "request_information",
    "draft_decline_rationale",
}

_TITLES = {
    "proceed_to_quote_prep": "Proceed to quote preparation",
    "refer_to_senior_underwriter": "Refer to a senior underwriter",
    "request_information": "Request the missing information from the broker",
    "draft_decline_rationale": "Draft a decline rationale for underwriter review",
}

_DESCRIPTIONS = {
    "proceed_to_quote_prep": (
        "The submission is within appetite and complete enough to move to "
        "quote preparation under standard authority. The underwriter still "
        "decides; nothing is quoted by this step."
    ),
    "refer_to_senior_underwriter": (
        "Referral triggers or the risk tier put this submission above "
        "standard authority; a senior underwriter reviews the packet first."
    ),
    "request_information": (
        "The submission cannot be assessed until the broker supplies the "
        "missing elements listed in the packet."
    ),
    "draft_decline_rationale": (
        "The class as submitted sits outside appetite; a decline rationale "
        "is drafted for the underwriter, who makes the decline decision."
    ),
}

# How well each kind fits each triage outcome (0..1).
_FIT: dict[str, dict[str, float]] = {
    "in_appetite": {
        "proceed_to_quote_prep": 1.0,
        "refer_to_senior_underwriter": 0.35,
        "request_information": 0.3,
        "draft_decline_rationale": 0.0,
    },
    "borderline": {
        "proceed_to_quote_prep": 0.45,
        "refer_to_senior_underwriter": 1.0,
        "request_information": 0.45,
        "draft_decline_rationale": 0.1,
    },
    "out_of_appetite": {
        "proceed_to_quote_prep": 0.0,
        "refer_to_senior_underwriter": 0.5,
        "request_information": 0.2,
        "draft_decline_rationale": 1.0,
    },
    "incomplete": {
        "proceed_to_quote_prep": 0.1,
        "refer_to_senior_underwriter": 0.4,
        "request_information": 1.0,
        "draft_decline_rationale": 0.05,
    },
}

# How well each kind fits the computed data-readiness flag (0..1).
_READINESS_FIT: dict[str, dict[str, float]] = {
    "ready": {
        "proceed_to_quote_prep": 1.0,
        "refer_to_senior_underwriter": 0.7,
        "request_information": 0.2,
        "draft_decline_rationale": 0.8,
    },
    "ready_with_gaps": {
        "proceed_to_quote_prep": 0.6,
        "refer_to_senior_underwriter": 0.8,
        "request_information": 0.8,
        "draft_decline_rationale": 0.6,
    },
    "not_ready": {
        "proceed_to_quote_prep": 0.1,
        "refer_to_senior_underwriter": 0.5,
        "request_information": 1.0,
        "draft_decline_rationale": 0.4,
    },
}

_TIER_ORDER = {"low": 0, "moderate": 1, "high": 2, "elevated": 3}


def template_proposals(triage: dict, facts: dict) -> list[dict]:
    """The deterministic proposal set for a triage (all four kinds are
    always proposed — scoring, not presence, expresses the fit)."""
    return [
        {"kind": kind, "title": _TITLES[kind], "description": _DESCRIPTIONS[kind]}
        for kind in (
            "proceed_to_quote_prep",
            "refer_to_senior_underwriter",
            "request_information",
            "draft_decline_rationale",
        )
    ]


def score_option(kind: str, triage: dict, facts: dict) -> float:
    """The deterministic score for one option kind (see module docstring)."""
    outcome = triage.get("outcome", "incomplete")
    readiness = facts.get("readiness", "not_ready")
    fit = _FIT.get(outcome, _FIT["incomplete"]).get(kind, 0.0)
    readiness_fit = _READINESS_FIT.get(readiness, _READINESS_FIT["not_ready"]).get(kind, 0.0)
    tier = _TIER_ORDER.get(triage.get("risk_tier", "moderate"), 1)
    tier_penalty = 0.0
    if kind == "proceed_to_quote_prep":
        tier_penalty = 6.0 * tier
    loss = facts.get("loss_summary") or {}
    frequency = float(loss.get("frequency_per_year") or 0.0)
    loss_penalty = 0.0
    if kind == "proceed_to_quote_prep":
        loss_penalty = min(frequency, 2.0) * 4.0
    return round(60.0 * fit + 40.0 * readiness_fit - tier_penalty - loss_penalty, 2)


def score_options(proposals: list[dict], triage: dict, facts: dict) -> list[RecoveryOption]:
    """Validate kinds, score every proposal, mark the argmax recommended."""
    scored: list[RecoveryOption] = []
    for proposal in proposals:
        kind = str(proposal.get("kind", "")).strip().lower()
        if kind not in OPTION_KINDS:
            continue
        scored.append(
            RecoveryOption(
                kind=kind,
                title=str(proposal.get("title") or _TITLES[kind])[:120],
                description=str(proposal.get("description") or _DESCRIPTIONS[kind])[:300],
                score=score_option(kind, triage, facts),
            )
        )
    if not scored:
        scored = score_options(template_proposals(triage, facts), triage, facts)
    best = max(scored, key=lambda o: o.score)
    for option in scored:
        option.recommended = option is best
    scored.sort(key=lambda o: o.score, reverse=True)
    return scored


def build_options(
    *,
    submission,
    triage: dict,
    facts: dict,
    diagnosis_summary: str,
    backend,
) -> tuple[list[RecoveryOption], str]:
    """Propose + score the next-step options for a triage.

    In provider mode the LLM proposes the option wording (kinds
    validated, scores computed here); otherwise the template proposal
    set runs. Returns (options, note) — the note records a degraded
    LLM path.
    """
    propose = getattr(backend, "propose_options", None)
    if propose is None:
        return score_options(template_proposals(triage, facts), triage, facts), ""
    context = DraftContext(
        submission_id=submission.submission_id,
        insured_name=submission.insured_name,
        broker_name=submission.broker_name,
        line_of_business=submission.line_of_business,
        class_of_business=submission.class_of_business,
        outcome=triage["outcome"],
        risk_tier=triage["risk_tier"],
        rationale=triage["rationale"],
        referral_triggers=triage.get("referral_triggers") or [],
        completeness=facts.get("completeness_score"),
        readiness=facts.get("readiness"),
        missing=facts.get("missing_elements") or [],
        loss_summary="",
        diagnosis_summary=diagnosis_summary,
        guideline_details=[],
    )
    note = ""
    try:
        proposals = propose(context)
    except Exception as exc:  # options never fail the run
        proposals = None
        note = f"LLM option proposals failed ({exc}) — template proposals used"
    if not proposals:
        proposals = template_proposals(triage, facts)
        if not note:
            note = "LLM option proposals were unusable — template proposals used"
    return score_options(proposals, triage, facts), note
