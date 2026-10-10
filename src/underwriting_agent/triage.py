"""Deterministic triage rules over a synthetic appetite matrix.

Rules run first and independently. They read only computed facts and
provided submission fields — never model output — so the same
submission always triages the same way. The LLM cross-check (see
``crosscheck.py``) can only change the outcome under the one explicit
resolution policy.
"""

from __future__ import annotations

from .facts import normalize_line
from .schemas import SubmissionFacts, SubmissionInput, Triage

OUT_OF_APPETITE_CLASSES = [
    "hazardous waste transport",
    "hazardous waste",
    "fireworks manufacturing",
    "fireworks",
    "cryptocurrency mining",
    "crypto mining",
]

STANDARD_REVENUE_AUTHORITY = 100_000_000
STANDARD_LIMIT_AUTHORITY = 5_000_000


def _limit_value(limit: str) -> float:
    if not limit:
        return 0.0
    s = str(limit).lower().replace(",", "").replace("$", "")
    mult = 1.0
    if "mm" in s or "m" in s and "month" not in s:
        mult = 1_000_000
    digits = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    try:
        return float(digits or 0) * (mult if mult != 1.0 or "m" in s else 1.0)
    except ValueError:
        return 0.0


def classify_submission(sub: SubmissionInput, facts: SubmissionFacts) -> Triage:
    line = normalize_line(sub.line_of_business)
    haystack = f"{sub.class_of_business} {sub.line_of_business}".lower()
    signals: list[str] = []
    triggers: list[str] = []
    ls = facts.loss_summary

    for cls in OUT_OF_APPETITE_CLASSES:
        if cls in haystack:
            return Triage(
                outcome="out_of_appetite",
                risk_tier="elevated",
                confidence=0.95,
                rationale=f"Class '{sub.class_of_business or sub.line_of_business}' is outside the synthetic appetite matrix (UW-GEN-01).",
                signals=[f"out-of-appetite class: {cls}"],
                referral_triggers=["out-of-appetite class"],
            )

    if facts.readiness == "not_ready" or (
        line == "directors_and_officers" and "financials" in facts.missing_elements
    ) or (
        line == "property" and "values_schedule" in facts.missing_elements and facts.completeness_score < 0.75
    ):
        return Triage(
            outcome="incomplete",
            risk_tier="moderate",
            confidence=0.9,
            rationale=(
                "Required elements are missing for this line "
                f"({', '.join(facts.missing_elements) or 'insufficient documentation'}); "
                "the submission cannot be assessed until they are received."
            ),
            signals=[f"missing: {m}" for m in facts.missing_elements] or ["readiness: not_ready"],
            referral_triggers=[],
        )

    # Referral triggers (borderline)
    if ls.claim_count >= 3:
        triggers.append(f"{ls.claim_count} claims in the loss history (referral trigger: frequency)")
    if ls.largest_claim > 250_000:
        triggers.append(f"largest claim ${ls.largest_claim:,.0f} exceeds 250,000 (referral trigger: severity)")
    if sub.revenue is not None and sub.revenue > STANDARD_REVENUE_AUTHORITY:
        triggers.append(f"revenue ${sub.revenue:,.0f} exceeds standard authority of 100,000,000")
    if _limit_value(sub.requested_limit) > STANDARD_LIMIT_AUTHORITY:
        triggers.append(f"requested limit {sub.requested_limit} exceeds standard authority of 5,000,000")
    text = f"{sub.notes} {sub.broker_email_text}".lower()
    if any(p in text for p in ("pending litigation", "threatened litigation", "going concern", "lawsuit pending")):
        triggers.append("pending/threatened litigation or going-concern language in the submission")
    if line == "cyber" and any("ransomware" in (l.description or "").lower() for l in sub.losses):
        triggers.append("ransomware loss in the loss history (cyber referral trigger)")
    if line == "professional_liability" and sub.years_in_business is not None and sub.years_in_business < 3:
        triggers.append("fewer than three years in business (E&O referral trigger)")

    # Risk tier
    if ls.largest_claim > 500_000 or ls.claim_count >= 5:
        tier = "elevated"
    elif ls.claim_count >= 3 or ls.largest_claim > 250_000:
        tier = "high"
    elif ls.claim_count >= 1 or triggers:
        tier = "moderate"
    else:
        tier = "low"
    signals.append(f"loss history: {ls.claim_count} claim(s), total incurred ${ls.total_incurred:,.0f}")
    signals.append(f"readiness: {facts.readiness} (completeness {facts.completeness_score:.2f})")

    if triggers:
        return Triage(
            outcome="borderline",
            risk_tier=tier if tier != "low" else "moderate",
            confidence=0.88,
            rationale="One or more referral triggers are present; a senior underwriter should review before quote preparation.",
            signals=signals,
            referral_triggers=triggers,
        )

    return Triage(
        outcome="in_appetite",
        risk_tier=tier,
        confidence=0.94,
        rationale="No referral triggers, documentation complete for the line, and the class is within the synthetic appetite matrix.",
        signals=signals,
        referral_triggers=[],
    )
