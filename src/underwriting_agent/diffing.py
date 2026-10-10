"""Renewal / resubmission diffing — computed in code, never the model.

When an insured submits again, the underwriter's real question is
not "what does this look like" but "what changed". This module
diffs the new submission against the insured's most recent stored
one — requested limit and deductible, revenue, years in business,
line/class, prior carrier, effective date, and the computed loss
summary — deterministically, from the stored payload and both
sides' computed facts. The diff rides on the result, joins the
diagnosis evidence, and gets its own section in the packet.

The diff also MOVES triage: material adverse deltas (new claims
clearing a severity floor, a limit jump beyond a configured
percentage, a class change) become referral triggers on the rule
triage before the cross-check — see ``diff_referral_triggers``.

Matching is by insured name (the same key the memory features use).
Only the most recent prior counts: a renewal is priced against the
last submission, not the whole history.
"""

from __future__ import annotations

from .config import env_float
from .facts import normalize_line
from .schemas import (
    DiffFieldChange,
    SubmissionDiff,
    SubmissionFacts,
    SubmissionInput,
    Triage,
)
from .triage import _limit_value


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _amount_text(raw: str) -> str:
    """A requested limit/deductible as display text: normalised to a
    money figure when it parses, the submitted text when it does not."""
    parsed = _limit_value(raw)
    return _money(parsed) if parsed else (raw or "not stated")


def _loss_text(facts: SubmissionFacts) -> str:
    ls = facts.loss_summary
    return (
        f"{ls.claim_count} claim(s), {_money(ls.total_incurred)} incurred, "
        f"largest {_money(ls.largest_claim)}"
    )


def _pct_change(old: float, new: float) -> str:
    if not old:
        return ""
    pct = (new - old) / old * 100
    return f" ({pct:+.1f}%)" if abs(pct) >= 0.05 else ""


def compute_submission_diff(
    current: SubmissionInput,
    current_facts: SubmissionFacts,
    prior: dict,
) -> SubmissionDiff | None:
    """Diff ``current`` against one prior submission bundle.

    ``prior`` is the service layer's projection of the prior stored
    record: {submission_id, submitted_at, outcome, risk_tier,
    submission (the stored payload), facts (the prior result's
    computed facts)}. Returns None when the bundle carries no usable
    prior submission. A diff with an empty ``changes`` list is a real
    answer — the insured resubmitted and nothing material moved.
    """
    payload = prior.get("submission") or {}
    if not payload:
        return None
    try:
        previous = SubmissionInput.model_validate(payload)
    except Exception:
        return None
    prior_facts_raw = prior.get("facts") or {}
    try:
        prior_facts = SubmissionFacts.model_validate(prior_facts_raw)
    except Exception:
        prior_facts = SubmissionFacts()

    changes: list[DiffFieldChange] = []

    def _amount_change(field: str, label: str, old_raw: str, new_raw: str) -> None:
        if (old_raw or "").strip() == (new_raw or "").strip():
            return
        old_v, new_v = _limit_value(old_raw), _limit_value(new_raw)
        note = ""
        if old_v and new_v and old_v != new_v:
            note = "increased" if new_v > old_v else "decreased"
            note += _pct_change(old_v, new_v)
        changes.append(
            DiffFieldChange(
                field=field,
                label=label,
                previous=_amount_text(old_raw),
                current=_amount_text(new_raw),
                note=note,
            )
        )

    _amount_change(
        "requested_limit", "Requested limit",
        previous.requested_limit, current.requested_limit,
    )
    _amount_change(
        "requested_deductible", "Requested deductible",
        previous.requested_deductible, current.requested_deductible,
    )

    if (
        previous.revenue is not None
        and current.revenue is not None
        and previous.revenue != current.revenue
    ):
        changes.append(
            DiffFieldChange(
                field="revenue",
                label="Revenue",
                previous=_money(previous.revenue),
                current=_money(current.revenue),
                note=("increased" if current.revenue > previous.revenue else "decreased")
                + _pct_change(previous.revenue, current.revenue),
            )
        )
    if (
        previous.years_in_business is not None
        and current.years_in_business is not None
        and previous.years_in_business != current.years_in_business
    ):
        delta = current.years_in_business - previous.years_in_business
        changes.append(
            DiffFieldChange(
                field="years_in_business",
                label="Years in business",
                previous=f"{previous.years_in_business:g}",
                current=f"{current.years_in_business:g}",
                note=f"{delta:+g} year(s)",
            )
        )
    if normalize_line(previous.line_of_business) != normalize_line(current.line_of_business):
        changes.append(
            DiffFieldChange(
                field="line_of_business",
                label="Line of business",
                previous=previous.line_of_business or "not stated",
                current=current.line_of_business or "not stated",
            )
        )
    if (previous.class_of_business or "").strip().lower() != (
        current.class_of_business or ""
    ).strip().lower():
        changes.append(
            DiffFieldChange(
                field="class_of_business",
                label="Class of business",
                previous=previous.class_of_business or "not stated",
                current=current.class_of_business or "not stated",
            )
        )
    if (previous.prior_carrier or "") != (current.prior_carrier or ""):
        changes.append(
            DiffFieldChange(
                field="prior_carrier",
                label="Prior carrier",
                previous=previous.prior_carrier or "not stated",
                current=current.prior_carrier or "not stated",
            )
        )
    if (previous.effective_date_requested or "") != (current.effective_date_requested or ""):
        changes.append(
            DiffFieldChange(
                field="effective_date_requested",
                label="Requested effective date",
                previous=previous.effective_date_requested or "not stated",
                current=current.effective_date_requested or "not stated",
            )
        )

    old_loss, new_loss = prior_facts.loss_summary, current_facts.loss_summary
    if (old_loss.claim_count, old_loss.total_incurred, old_loss.largest_claim) != (
        new_loss.claim_count,
        new_loss.total_incurred,
        new_loss.largest_claim,
    ):
        note = ""
        if new_loss.claim_count > old_loss.claim_count:
            note = f"{new_loss.claim_count - old_loss.claim_count} new claim(s) since the prior submission"
        elif new_loss.claim_count < old_loss.claim_count:
            note = "fewer claims than the prior loss history showed"
        changes.append(
            DiffFieldChange(
                field="loss_history",
                label="Loss history (computed)",
                previous=_loss_text(prior_facts),
                current=_loss_text(current_facts),
                note=note,
            )
        )

    return SubmissionDiff(
        prior_submission_id=prior.get("submission_id") or previous.submission_id,
        prior_submitted_at=prior.get("submitted_at") or previous.submitted_at,
        prior_outcome=prior.get("outcome") or "",
        prior_risk_tier=prior.get("risk_tier") or "",
        changes=changes,
    )


def diff_evidence_lines(diff: SubmissionDiff) -> list[str]:
    """The diff as diagnosis evidence, in the evidence list's style."""
    header = (
        f"renewal diff: prior submission {diff.prior_submission_id}"
        + (f" (submitted {diff.prior_submitted_at})" if diff.prior_submitted_at else "")
        + (
            f" triaged {diff.prior_outcome} / {diff.prior_risk_tier}"
            if diff.prior_outcome
            else ""
        )
    )
    lines = [header]
    if not diff.changes:
        lines.append("renewal diff: no material changes from the prior submission")
    for change in diff.changes:
        line = (
            f"renewal diff: {change.label.lower()} "
            f"{change.previous} -> {change.current}"
        )
        if change.note:
            line += f" ({change.note})"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Diff-driven referral triggers — the diff moves triage, not just prose
# ---------------------------------------------------------------------------

# Thresholds for "materially adverse": deliberately conservative
# defaults, both configurable — a deployment tunes them to its own
# appetite without touching code.
DEFAULT_LIMIT_JUMP_PCT = 50.0  # DIFF_LIMIT_JUMP_PCT
DEFAULT_NEW_CLAIM_MIN_INCURRED = 50_000.0  # DIFF_NEW_CLAIM_MIN_INCURRED


def diff_referral_triggers(
    current: SubmissionInput,
    current_facts: SubmissionFacts,
    prior: dict,
    *,
    limit_jump_pct: float | None = None,
    new_claim_min_incurred: float | None = None,
) -> list[str]:
    """Referral triggers computed from the renewal diff itself.

    Three material adverse deltas, all computed in code from the
    prior bundle (the same inputs :func:`compute_submission_diff`
    reads): new claims whose incurred total clears a severity floor,
    a requested-limit jump beyond a configured percentage, and a
    class-of-business change. Benign movement — a small new claim, a
    limit keeping pace with the book, an unchanged class — produces
    no trigger: the diff describes many renewals and refers few.
    """
    payload = (prior or {}).get("submission") or {}
    if not payload:
        return []
    try:
        previous = SubmissionInput.model_validate(payload)
    except Exception:
        return []
    try:
        prior_facts = SubmissionFacts.model_validate((prior or {}).get("facts") or {})
    except Exception:
        prior_facts = SubmissionFacts()
    if limit_jump_pct is None:
        limit_jump_pct = env_float("DIFF_LIMIT_JUMP_PCT", DEFAULT_LIMIT_JUMP_PCT)
    if new_claim_min_incurred is None:
        new_claim_min_incurred = env_float(
            "DIFF_NEW_CLAIM_MIN_INCURRED", DEFAULT_NEW_CLAIM_MIN_INCURRED
        )

    triggers: list[str] = []

    old_loss, new_loss = prior_facts.loss_summary, current_facts.loss_summary
    new_claims = new_loss.claim_count - old_loss.claim_count
    new_incurred = new_loss.total_incurred - old_loss.total_incurred
    if new_claims >= 1 and new_incurred >= new_claim_min_incurred:
        triggers.append(
            f"{new_claims} new claim(s) since the prior submission, "
            f"{_money(new_incurred)} incurred "
            "(referral trigger: renewal diff — new claims)"
        )

    old_limit = _limit_value(previous.requested_limit)
    new_limit = _limit_value(current.requested_limit)
    if old_limit > 0 and new_limit > old_limit:
        pct = (new_limit - old_limit) / old_limit * 100
        if pct > limit_jump_pct:
            text = (
                f"requested limit increased {pct:.0f}% "
                f"({_money(old_limit)} -> {_money(new_limit)}) "
                "since the prior submission"
            )
            if previous.revenue and current.revenue is not None:
                rev_pct = (current.revenue - previous.revenue) / previous.revenue * 100
                if rev_pct > 0:
                    text += f", while revenue rose {rev_pct:.0f}%"
                elif rev_pct < 0:
                    text += f", while revenue fell {abs(rev_pct):.0f}%"
                else:
                    text += ", with revenue flat"
            triggers.append(
                text + " (referral trigger: renewal diff — limit jump)"
            )

    old_class = (previous.class_of_business or "").strip()
    new_class = (current.class_of_business or "").strip()
    if old_class and new_class and old_class.lower() != new_class.lower():
        triggers.append(
            f"class of business changed from '{old_class}' to '{new_class}' "
            "since the prior submission "
            "(referral trigger: renewal diff — class change)"
        )

    return triggers


def apply_diff_triggers(triage: Triage, triggers: list[str]) -> Triage:
    """Fold diff triggers into a rule triage.

    Only an otherwise-clean or already-referred triage can move: an
    ``in_appetite`` outcome escalates to ``borderline`` (tier floored
    at moderate, confidence capped at the borderline rules' 0.88)
    with the rationale saying the referral is on the change; a
    ``borderline`` outcome just gains the triggers. ``incomplete``
    and ``out_of_appetite`` are returned untouched — a submission
    that cannot be assessed, or is outside appetite, is not made
    more or less so by its history.
    """
    if not triggers or triage.outcome not in ("in_appetite", "borderline"):
        return triage
    adjusted = triage.model_copy(deep=True)
    for trigger in triggers:
        if trigger not in adjusted.referral_triggers:
            adjusted.referral_triggers.append(trigger)
    adjusted.signals.append(
        "renewal diff: material adverse change since the prior submission"
    )
    if adjusted.outcome == "in_appetite":
        adjusted.outcome = "borderline"
        if adjusted.risk_tier == "low":
            adjusted.risk_tier = "moderate"
        adjusted.confidence = min(adjusted.confidence, 0.88)
        adjusted.rationale += (
            " Renewal diff referral: " + "; ".join(triggers)
            + " — the change since the last submission is itself a referral reason."
        )
    return adjusted


def diff_packet_lines(diff: SubmissionDiff) -> list[str]:
    """The diff as the packet's "Changes since last submission"
    section — header plus one line per changed field."""
    header = f"Changes since last submission ({diff.prior_submission_id}"
    if diff.prior_submitted_at:
        header += f", submitted {diff.prior_submitted_at}"
    if diff.prior_outcome:
        header += f", triaged {diff.prior_outcome} / {diff.prior_risk_tier}"
    header += "):"
    lines = [header]
    if not diff.changes:
        lines.append("- no material changes from the prior submission")
    for change in diff.changes:
        line = f"- {change.label}: {change.previous} -> {change.current}"
        if change.note:
            line += f" ({change.note})"
        lines.append(line)
    return lines
