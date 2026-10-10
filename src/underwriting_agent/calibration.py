"""Diff-trigger calibration — what-if over the stored renewals.

The renewal-diff referral triggers (``diffing.diff_referral_triggers``)
run on three thresholds: how big a limit jump refers, and the
incurred floors for new claims and adverse development. Those
numbers decide how much renewal movement reaches an underwriter,
and until now tuning them was a guess. This module makes it
evidence: it rebuilds every stored renewal pair (the same chain
the service diffs against — per insured, per line, most recent
prior), then **replays** the trigger computation at candidate
thresholds and reports how many referrals each setting would
have produced, family by family.

Read-only by construction: the replay recomputes triggers from
the stored payloads and facts and writes nothing. The recorded
side of the report — the triggers that actually fired at analysis
time, and what underwriters decided on those cases — comes from
the stored results, so a threshold change can be weighed against
both the what-if and the record.
"""

from __future__ import annotations

from collections import Counter

from .bundles import is_bundle, split_bundle
from .config import env_float
from .diffing import (
    DEFAULT_DEVELOPMENT_MIN_INCREASE,
    DEFAULT_LIMIT_JUMP_PCT,
    DEFAULT_NEW_CLAIM_MIN_INCURRED,
    diff_referral_triggers,
)
from .facts import normalize_line
from .schemas import SubmissionInput
from .store import ApprovalRecord

# The four trigger families, named by the marker the trigger text
# itself carries ("(referral trigger: renewal diff — limit jump)").
TRIGGER_FAMILIES = (
    "new_claims",
    "adverse_development",
    "limit_jump",
    "class_change",
)

_FAMILY_MARKERS = {
    "new_claims": "— new claims)",
    "adverse_development": "— adverse loss development)",
    "limit_jump": "— limit jump)",
    "class_change": "— class change)",
}


def trigger_family(trigger_text: str) -> str | None:
    """The family of one referral-trigger string, or None when the
    trigger did not come from the renewal diff."""
    if "referral trigger: renewal diff" not in trigger_text:
        return None
    for family, marker in _FAMILY_MARKERS.items():
        if marker in trigger_text:
            return family
    return None


def current_thresholds() -> dict:
    """The thresholds the live pipeline is configured with."""
    return {
        "limit_jump_pct": env_float("DIFF_LIMIT_JUMP_PCT", DEFAULT_LIMIT_JUMP_PCT),
        "new_claim_min_incurred": env_float(
            "DIFF_NEW_CLAIM_MIN_INCURRED", DEFAULT_NEW_CLAIM_MIN_INCURRED
        ),
        "development_min_increase": env_float(
            "DIFF_DEVELOPMENT_MIN_INCREASE", DEFAULT_DEVELOPMENT_MIN_INCREASE
        ),
    }


def _prior_bundle_for(
    prior_records: list[ApprovalRecord], insured_name: str, line: str
) -> dict | None:
    """The prior bundle for one (insured, line) over an explicit
    record prefix — the same projection
    ``service._prior_submission_for`` builds at analysis time,
    re-derived here so the replay reads the store the same way the
    pipeline did."""
    if not insured_name:
        return None
    target = normalize_line(line)
    for record in reversed(prior_records):
        insured = record.result.insured_name or record.submission.get(
            "insured_name", ""
        )
        if insured != insured_name or not record.submission:
            continue
        payload = record.submission
        if payload.get("lines"):
            try:
                prior_model = SubmissionInput.model_validate(payload)
            except Exception:
                continue
            for sub, analysis in zip(
                split_bundle(prior_model), record.result.line_analyses
            ):
                if normalize_line(sub.line_of_business) != target:
                    continue
                return {
                    "submission_id": analysis.submission_id or sub.submission_id,
                    "submitted_at": sub.submitted_at,
                    "outcome": analysis.result.triage.outcome,
                    "risk_tier": analysis.result.triage.risk_tier,
                    "submission": sub.model_dump(mode="json"),
                    "facts": analysis.result.facts.model_dump(mode="json"),
                }
            continue
        if normalize_line(payload.get("line_of_business", "")) != target:
            continue
        return {
            "submission_id": record.result.submission_id,
            "submitted_at": payload.get("submitted_at", ""),
            "outcome": record.result.triage.outcome,
            "risk_tier": record.result.triage.risk_tier,
            "submission": payload,
            "facts": record.result.facts.model_dump(mode="json"),
        }
    return None


def renewal_pairs(records: list[ApprovalRecord]) -> list[dict]:
    """Every stored renewal as a replayable pair, in store order.

    One pair per analysed unit — a single-line submission, or one
    line of a bundle — that had a prior for its (insured, line).
    Each pair carries the current submission + its computed facts
    (what the trigger computation reads), the prior bundle (what it
    diffs against), and the triggers the analysis itself recorded.
    """
    pairs: list[dict] = []
    for index, record in enumerate(records):
        if not record.submission:
            continue
        result = record.result
        insured = result.insured_name or record.submission.get("insured_name", "")
        try:
            model = SubmissionInput.model_validate(record.submission)
        except Exception:
            continue
        if is_bundle(model):
            units = [
                (sub, analysis.result)
                for sub, analysis in zip(split_bundle(model), result.line_analyses)
            ]
        else:
            units = [(model, result)]
        for sub, unit_result in units:
            prior = _prior_bundle_for(records[:index], insured, sub.line_of_business)
            if prior is None:
                continue
            pairs.append(
                {
                    "submission_id": unit_result.submission_id,
                    "record_submission_id": result.submission_id,
                    "insured_name": insured,
                    "line_of_business": sub.line_of_business,
                    "submitted_at": sub.submitted_at,
                    "status": result.status,
                    "outcome": unit_result.triage.outcome,
                    "current": sub,
                    "current_facts": unit_result.facts,
                    "prior": prior,
                    "recorded_triggers": [
                        t
                        for t in unit_result.triage.referral_triggers
                        if trigger_family(t) is not None
                    ],
                }
            )
    return pairs


# Candidate values swept per dimension (the configured value is
# always added, so the current setting is a row in its own sweep).
DEFAULT_CANDIDATES = {
    "limit_jump_pct": [25.0, 50.0, 75.0, 100.0],
    "new_claim_min_incurred": [10_000.0, 25_000.0, 50_000.0, 100_000.0],
    "development_min_increase": [10_000.0, 25_000.0, 50_000.0, 100_000.0],
}


def _evaluate(pairs: list[dict], thresholds: dict) -> tuple[int, dict]:
    """Replay the triggers at one threshold set: (pairs with at
    least one trigger, per-family pair counts)."""
    referrals = 0
    families: Counter = Counter()
    for pair in pairs:
        triggers = diff_referral_triggers(
            pair["current"], pair["current_facts"], pair["prior"], **thresholds
        )
        fired = {trigger_family(t) for t in triggers} - {None}
        if fired:
            referrals += 1
            families.update(fired)
    return referrals, dict(sorted(families.items()))


def trigger_calibration(
    records: list[ApprovalRecord],
    *,
    candidates: dict | None = None,
) -> dict:
    """The calibration report over the whole store.

    ``at_current`` replays the stored renewals at the configured
    thresholds; ``recorded_at_analysis`` reports what actually
    fired when those renewals were analysed (and what underwriters
    decided on the moved cases). ``sweeps`` varies one threshold at
    a time — the others held at the configured values — so each
    knob's effect reads on its own. Nothing is written.
    """
    pairs = renewal_pairs(records)
    current = current_thresholds()
    base_referrals, base_families = _evaluate(pairs, current)

    recorded = [p for p in pairs if p["recorded_triggers"]]
    recorded_families: Counter = Counter()
    for pair in recorded:
        recorded_families.update(
            {trigger_family(t) for t in pair["recorded_triggers"]} - {None}
        )

    sweeps = []
    grid = candidates or DEFAULT_CANDIDATES
    for dimension in (
        "limit_jump_pct",
        "new_claim_min_incurred",
        "development_min_increase",
    ):
        values = sorted({float(v) for v in grid.get(dimension, [])} | {current[dimension]})
        rows = []
        for value in values:
            referrals, families = _evaluate(pairs, {**current, dimension: value})
            rows.append(
                {
                    "value": value,
                    "is_current": value == current[dimension],
                    "referrals": referrals,
                    "delta_vs_current": referrals - base_referrals,
                    "family_counts": families,
                }
            )
        sweeps.append({"dimension": dimension, "rows": rows})

    return {
        "renewals": len(pairs),
        "records": len(records),
        "current_thresholds": current,
        "at_current": {"referrals": base_referrals, "family_counts": base_families},
        "recorded_at_analysis": {
            "referrals": len(recorded),
            "family_counts": dict(sorted(recorded_families.items())),
            "by_status": dict(
                sorted(Counter(p["status"] for p in recorded).items())
            ),
            "by_outcome": dict(
                sorted(Counter(p["outcome"] for p in recorded).items())
            ),
            "submission_ids": [p["submission_id"] for p in recorded],
        },
        "sweeps": sweeps,
        "note": (
            "Read-only what-if: triggers are recomputed from the stored "
            "renewal pairs at each candidate setting; no record, triage, "
            "or threshold is changed by reading this report."
        ),
    }
