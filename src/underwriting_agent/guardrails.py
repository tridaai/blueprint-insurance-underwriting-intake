"""Guardrails as code: deterministic checks on every draft.

These run on the broker letter and the underwriter packet in every
mode — template, LLM, and repaired drafts alike — and a failed
blocking check makes the draft unapprovable (the service layer
refuses approval and the API returns 422).

The rules encode the blueprint's core promise: **the agent never
quotes a firm premium, never states or implies coverage is bound,
and never promises an outcome.** A broker's own words can violate
them — a broker relaying a client's demand ("coverage must be in
force from this morning") is reported speech, not an instruction,
so the injection screen leaves it alone; when a drafter pastes it
into a letter, these checks are what stop it. See sample SUB-1011.
"""

from __future__ import annotations

import re

from .schemas import Drafts, GuardrailCheck, SubmissionInput, Validation

# Affirmative bind language. Each match is checked against a negation
# window (see _affirmative): "no coverage is bound" is the required
# disclaimer, not a violation.
_BIND_PATTERNS = [
    r"\bcoverage is (now )?(bound|in force|placed|effected)\b",
    r"\b(policy|coverage) (is|will be) (bound|in force|placed)\b",
    r"\bbound (effective|today|from)\b",
    r"\bmust be in force\b",
    r"\bwe (have|'ve) bound\b",
    r"\bconfirm(s|ing|ed)? (that )?(the )?coverage\b",
    r"\bpleased to confirm\b[^.]{0,40}\bcoverage\b",
    r"\byou are (now )?covered\b",
    r"\bcoverage (has been|is hereby) (bound|placed|effected)\b",
]

_NEGATION = re.compile(r"\b(not|never|no|n't|nor)\b", re.IGNORECASE)


def _affirmative(pattern: str, text: str) -> str | None:
    """First match of ``pattern`` not preceded by a negation within
    30 characters — returns the matched phrase, or None."""
    for match in re.finditer(pattern, text, re.IGNORECASE):
        window = text[max(0, match.start() - 30):match.start()]
        if _NEGATION.search(window):
            continue
        return match.group(0)
    return None


_PREMIUM_FIGURE = [
    r"\bpremium\b[^.]{0,30}\$\s?[\d,]+",
    r"\$\s?[\d,]+[^.]{0,30}\bpremium\b",
    r"\b(quoted|firm|final|indicated) premium\b",
    r"\bprice (is|of|:)\s*\$\s?[\d,]+",
    r"\bwe can offer\b[^.]{0,40}\$\s?[\d,]+",
    r"\bannual premium\b[^.]{0,20}\d",
]

_PROMISE_PATTERNS = [
    # Outcome promises — the agent may never promise a decision.
    r"\bwe (promise|guarantee)\b",
    r"\bguaranteed (approval|acceptance|premium|coverage|terms)\b",
    r"\bwill be (approved|accepted|bound|covered|quoted)\b",
    r"\bapproval is (guaranteed|assured|certain)\b",
    r"\byou will (be|get) (approved|covered|a quote)\b",
    r"\bwe will (bind|place|approve|quote)\b",
    # Time commitments: a promise about WHEN something will happen,
    # without naming who does it ("your quote will arrive within 24
    # hours") — the same failure class as an explicit promise.
    r"\bwill (arrive|be sent|be issued|be ready|be provided|be delivered|reach you)\b[^.]{0,40}\b(within|in|by)\b[^.]{0,20}\b(hours?|days?|weeks?|tomorrow|eod|end of day|end of week)\b",
    r"\bwithin (the next )?\d+ (hours?|days?|weeks?)\b",
    r"\bby (tomorrow|end of day|eod|end of week|monday|tuesday|wednesday|thursday|friday)\b[^.]{0,30}\b(will|you|quote|decision|terms|premium)\b",
    r"\b(quote|decision|terms|premium|answer) (within|in|by)\b[^.]{0,20}\b(\d+|tomorrow|end of)",
]

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PASSPORT = re.compile(r"\b[A-Z]\d{8}\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_card(text: str) -> str | None:
    for match in re.finditer(r"\b(?:\d[ -]?){13,19}\b", text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return match.group(0)
    return None


def validate_drafts(
    submission: SubmissionInput,
    drafts: Drafts,
    citations: list[str],
) -> Validation:
    """Run all guardrail checks over the drafts. ``passed`` is the
    conjunction — every check here is blocking."""
    letter = drafts.broker_letter or ""
    packet = drafts.underwriter_packet or ""
    combined = f"{drafts.subject}\n{letter}\n{packet}"
    checks: list[GuardrailCheck] = []

    present = submission.submission_id in combined
    checks.append(
        GuardrailCheck(
            name="references_submission_id",
            passed=present,
            detail=(
                f"submission id {submission.submission_id} referenced"
                if present
                else f"drafts never reference submission {submission.submission_id}"
            ),
        )
    )

    bind_hit = next(
        (hit for hit in (_affirmative(p, combined) for p in _BIND_PATTERNS) if hit),
        None,
    )
    checks.append(
        GuardrailCheck(
            name="no_bind_language",
            passed=bind_hit is None,
            detail=(
                "no language states or implies coverage is bound"
                if bind_hit is None
                else f"bind language found: '{bind_hit}' — the agent never binds coverage"
            ),
        )
    )

    premium_hit = next(
        (m.group(0) for p in _PREMIUM_FIGURE if (m := re.search(p, combined, re.IGNORECASE))),
        None,
    )
    checks.append(
        GuardrailCheck(
            name="no_firm_premium",
            passed=premium_hit is None,
            detail=(
                "no premium figure is stated or quoted"
                if premium_hit is None
                else f"premium figure found: '{premium_hit}' — the agent never quotes a premium"
            ),
        )
    )

    promise_hit = next(
        (m.group(0) for p in _PROMISE_PATTERNS if (m := re.search(p, combined, re.IGNORECASE))),
        None,
    )
    checks.append(
        GuardrailCheck(
            name="no_prohibited_promises",
            passed=promise_hit is None,
            detail=(
                "no outcome or timing promises"
                if promise_hit is None
                else f"prohibited promise found: '{promise_hit}'"
            ),
        )
    )

    pii_hit = None
    if _SSN.search(combined):
        pii_hit = "what looks like a US Social Security number"
    elif _find_card(combined):
        pii_hit = "what looks like a payment card number (Luhn-valid)"
    elif _PASSPORT.search(combined):
        pii_hit = "what looks like a passport number"
    checks.append(
        GuardrailCheck(
            name="no_pii_in_draft",
            passed=pii_hit is None,
            detail=(
                "no SSN / card / passport patterns in the drafts"
                if pii_hit is None
                else f"PII in draft: {pii_hit} — personal identifiers never belong in drafts"
            ),
        )
    )

    checks.append(
        GuardrailCheck(
            name="guideline_citations_present",
            passed=bool(citations),
            detail=(
                f"guideline citations present: {citations}"
                if citations
                else "no guideline citations — every triage must cite its guidelines"
            ),
        )
    )

    next_step = "next step" in combined.lower()
    checks.append(
        GuardrailCheck(
            name="states_next_step",
            passed=next_step,
            detail=(
                "the drafts state the next step"
                if next_step
                else "the drafts never state the next step for the broker"
            ),
        )
    )

    return Validation(passed=all(c.passed for c in checks), checks=checks)
