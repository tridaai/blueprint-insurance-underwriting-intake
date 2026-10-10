"""Autonomy recommendation: should cases like this one ever skip the gate?

After every run the pipeline prints a recommendation: whether this
run *would have been* eligible for auto-triage (no human gate) under
a conservative policy — and the reasons either way. It is a
recommendation only: nothing in this blueprint acts on it. The human
gate is unconditional; this exists so an adopting team can see, case
by case, what a future autonomy policy would have to be comfortable
with before they ever consider one.

Eligibility is deliberately narrow — ALL of:

- triage outcome ``in_appetite``;
- risk tier ``low`` or ``moderate``;
- data readiness ``ready`` (complete, no discrepancies);
- guardrails passed on the first draft (no repair was needed);
- the rules/LLM cross-check did not disagree (when it ran);
- the independent reviewer did not block.
"""

from __future__ import annotations

from .schemas import (
    AutonomyRecommendation,
    CrossCheck,
    ReviewResult,
    SubmissionFacts,
    Triage,
    Validation,
)


def recommend_autonomy(
    *,
    triage: Triage,
    facts: SubmissionFacts,
    validation: Validation,
    repair_attempted: bool,
    cross_check: CrossCheck | None,
    review: ReviewResult | None,
    reviewer_blocked: bool,
) -> AutonomyRecommendation:
    reasons: list[str] = []
    blockers: list[str] = []

    if triage.outcome == "in_appetite":
        reasons.append("triage outcome is in_appetite")
    else:
        blockers.append(f"triage outcome is {triage.outcome}, not in_appetite")

    if triage.risk_tier in ("low", "moderate"):
        reasons.append(f"risk tier is {triage.risk_tier}")
    else:
        blockers.append(f"risk tier is {triage.risk_tier}")

    if facts.readiness == "ready":
        reasons.append("data readiness is ready (complete, no discrepancies)")
    else:
        blockers.append(f"data readiness is {facts.readiness}")

    if validation.passed:
        reasons.append("guardrails passed")
    else:
        blockers.append("guardrails failed")

    if repair_attempted:
        blockers.append("the draft needed a repair pass")
    else:
        reasons.append("no repair pass was needed")

    if cross_check is not None and cross_check.resolution not in ("agree", "rules_only"):
        blockers.append(
            f"the rules/LLM cross-check did not agree ({cross_check.resolution})"
        )
    elif cross_check is not None:
        reasons.append("the rules/LLM cross-check raised no disagreement")

    if reviewer_blocked:
        blockers.append("the independent reviewer blocked the draft")
    elif review is not None and review.enabled:
        reasons.append("the independent reviewer did not block")

    return AutonomyRecommendation(
        eligible_for_auto_triage=not blockers,
        reasons=reasons + [f"not eligible: {b}" for b in blockers],
    )
