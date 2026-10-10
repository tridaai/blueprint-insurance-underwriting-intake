"""Guideline change-impact re-triage — a report, never a rewrite.

Guidelines change. When they do, every submission already sitting
in the queue was triaged under the old appetite, and nothing in the
old flow said which ones the change would move. This report closes
that gap: it re-runs **rules-only** triage over the submissions
still awaiting an underwriter, using each stored submission payload
and its stored extractions to recompute the facts, and lists the
ones whose outcome or risk tier would differ under the current
rules and corpus version.

A move is reported with its **reasons**: the guideline document and
the specific line in it the fresh outcome rests on — the referral
trigger that newly fires (or no longer fires), the documentation
standard a missing element belongs to, the loss-history line behind
a tier move. "It moved" is an alert; "it moved because UW-PROP-01's
severity line now catches its largest claim" is something an
underwriter can act on. The attribution is computed from the same
facts and trigger strings the rules produced — never paraphrased
from the guideline text at large.

Multi-line bundles are re-triaged **per line**: each line's stored
sub-analysis is recomputed on its own facts and compared against
its recorded outcome, and the bundle entry reports the affected
lines with their per-line movement (plus what the bundle roll-up
would become). A bundle is never flattened into a single fiction
pass over the bundle payload.

The report mutates nothing — no stored result, status, or decision
is touched. Acting on it (re-running the full pipeline on a moved
submission, or leaving it to the underwriter with the note) is a
human decision, like everything downstream in this system.

One honest nuance: the comparison is *rules, now* against the
*recorded* outcome. A recorded outcome can have come from the LLM
cross-check in provider mode, so a "moved" entry can also mean
"the recorded outcome was the cross-check's, and the bare rules
disagree" — the report names both sides and lets the underwriter
judge which kind of move it is.
"""

from __future__ import annotations

import re
from typing import Callable

from .bundles import rollup_triage, split_bundle
from .facts import compute_facts, normalize_line
from .guidelines_data import GUIDELINES, GUIDELINES_VERSION
from .schemas import SubmissionInput, Triage
from .store import ApprovalRecord
from .triage import classify_submission

Classifier = Callable[..., Triage]

_GUIDELINE_BY_ID = {g["id"]: g for g in GUIDELINES}


# ---------------------------------------------------------------------------
# Why a submission moved — guideline attribution
# ---------------------------------------------------------------------------


def _sentence(text: str, *needles: str) -> str:
    """The sentence of a guideline text that states a rule — the
    first sentence containing a needle, else the opening sentence."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    for needle in needles:
        for sentence in sentences:
            if needle.lower() in sentence.lower():
                return sentence
    return sentences[0] if sentences else text


def _reason(guideline_id: str, needles: tuple[str, ...], driver: str) -> dict:
    guideline = _GUIDELINE_BY_ID.get(guideline_id) or {}
    return {
        "guideline_id": guideline_id,
        "guideline_title": guideline.get("title", ""),
        "guideline_line": _sentence(guideline.get("text", ""), *needles),
        "driver": driver,
    }


def _trigger_guideline(trigger: str) -> tuple[str, tuple[str, ...]] | None:
    """The guideline line a rule-classifier referral trigger rests
    on, keyed by the trigger strings ``triage.py`` itself emits."""
    text = trigger.lower()
    if "claims in the loss history" in text:
        return "UW-PROP-01", ("three or more claims",)
    if "exceeds 250,000" in text:
        return "UW-PROP-01", ("single claim above 250,000",)
    if "standard authority" in text and "revenue" in text:
        return "UW-GEN-03", ("revenue up to 100 million",)
    if "standard authority" in text:
        return "UW-GEN-03", ("requested limits up to 5 million",)
    if "litigation" in text or "going concern" in text:
        return "UW-DNO-02", ("pending or threatened litigation",)
    if "ransomware" in text:
        return "UW-CYB-02", ("ransomware",)
    if "years in business" in text:
        return "UW-EO-01", ("at least three years in business",)
    return None


def _missing_guideline(
    sub: SubmissionInput, element: str
) -> tuple[str, tuple[str, ...]] | None:
    """The line-specific documentation standard a missing element
    belongs to, when one exists beyond the general standard."""
    line = normalize_line(sub.line_of_business)
    if element == "values_schedule" and line == "property":
        return "UW-PROP-02", ("statement of values is required",)
    if element == "financials" and line in ("directors_and_officers", "d&o"):
        return "UW-DNO-01", ("require two years of financial statements",)
    if element == "equipment_schedule" and line in ("marine", "inland_marine"):
        return "UW-MAR-01", ("equipment schedule",)
    return None


def _primary_guideline_id(sub: SubmissionInput) -> str:
    line = normalize_line(sub.line_of_business)
    for guideline in GUIDELINES:
        if line and line in guideline.get("lines", []):
            return guideline["id"]
    return "UW-GEN-02"


def explain_move(
    sub: SubmissionInput, facts, recorded: Triage, fresh: Triage
) -> list[dict]:
    """Why the fresh rules outcome differs from the recorded one,
    as guideline citations: each reason names the guideline
    document, quotes the specific line in it, and states the
    computed driver (the trigger, the missing element, the loss
    figures) that puts this submission on that line."""
    reasons: list[dict] = []

    def add(guideline_id: str, needles: tuple[str, ...], driver: str) -> None:
        entry = _reason(guideline_id, needles, driver)
        if entry not in reasons:
            reasons.append(entry)

    if fresh.outcome == "out_of_appetite" and recorded.outcome != "out_of_appetite":
        add(
            "UW-GEN-01",
            ("outside appetite",),
            f"class '{sub.class_of_business or sub.line_of_business}' is "
            "outside the appetite classes",
        )

    recorded_triggers = set(recorded.referral_triggers)
    for trigger in fresh.referral_triggers:
        if trigger in recorded_triggers:
            continue
        mapped = _trigger_guideline(trigger)
        if mapped:
            add(mapped[0], mapped[1], f"newly triggered: {trigger}")
    for trigger in recorded.referral_triggers:
        if trigger in fresh.referral_triggers:
            continue
        mapped = _trigger_guideline(trigger)
        if mapped:
            add(mapped[0], mapped[1], f"no longer triggered: {trigger}")

    if fresh.outcome == "incomplete" and recorded.outcome != "incomplete":
        for element in facts.missing_elements:
            mapped = _missing_guideline(sub, element)
            if mapped:
                add(mapped[0], mapped[1], f"missing for this line: {element}")
        add(
            "UW-GEN-02",
            ("missing required elements",),
            "required elements are missing: "
            + (", ".join(facts.missing_elements) or "insufficient documentation"),
        )

    if not reasons and fresh.risk_tier != recorded.risk_tier:
        summary = facts.loss_summary
        add(
            "UW-PROP-01",
            ("three or more claims",),
            "the risk tier follows the loss history: "
            f"{summary.claim_count} claim(s), largest "
            f"${summary.largest_claim:,.0f} incurred",
        )

    if not reasons:
        guideline_id = _primary_guideline_id(sub)
        guideline = _GUIDELINE_BY_ID.get(guideline_id) or {}
        reasons.append(
            {
                "guideline_id": guideline_id,
                "guideline_title": guideline.get("title", ""),
                "guideline_line": _sentence(guideline.get("text", "")),
                "driver": (
                    "no single rule line accounts for the move; this is "
                    "the governing guideline for the line under the "
                    "current corpus"
                ),
            }
        )
    return reasons


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _bundle_moved_entry(record: ApprovalRecord, classify: Classifier) -> dict | None:
    """Re-triage one awaiting bundle line by line.

    Returns the moved entry, or ``None`` when no line moved and the
    roll-up stands. Raises ``ValueError`` when the bundle payload
    cannot be re-read — the caller counts that as skipped, exactly
    like an unreadable single-line payload."""
    result = record.result
    try:
        model = SubmissionInput.model_validate(record.submission)
    except Exception as exc:
        raise ValueError("bundle payload cannot be re-read") from exc
    subs = {sub.submission_id: sub for sub in split_bundle(model)}
    fresh_analyses = []
    moved_lines: list[dict] = []
    for analysis in result.line_analyses:
        sub = subs.get(analysis.submission_id)
        if sub is None:
            raise ValueError(f"line {analysis.submission_id} cannot be re-read")
        facts = compute_facts(sub, analysis.result.extractions)
        fresh = classify(sub, facts)
        fresh_analyses.append(
            analysis.model_copy(
                update={"result": analysis.result.model_copy(update={"triage": fresh})}
            )
        )
        recorded_line = analysis.result.triage
        if fresh.outcome == recorded_line.outcome and fresh.risk_tier == recorded_line.risk_tier:
            continue
        moved_lines.append(
            {
                "line_of_business": analysis.line_of_business,
                "submission_id": analysis.submission_id,
                "from_outcome": recorded_line.outcome,
                "to_outcome": fresh.outcome,
                "from_tier": recorded_line.risk_tier,
                "to_tier": fresh.risk_tier,
                "reasons": explain_move(sub, facts, recorded_line, fresh),
            }
        )
    rollup = rollup_triage(fresh_analyses) if fresh_analyses else result.triage
    if not moved_lines and (
        rollup.outcome == result.triage.outcome
        and rollup.risk_tier == result.triage.risk_tier
    ):
        return None
    return {
        "submission_id": result.submission_id,
        "insured_name": result.insured_name,
        "multi_line": True,
        "triaged_under": result.guidelines_version or "unversioned",
        "from_outcome": result.triage.outcome,
        "to_outcome": rollup.outcome,
        "from_tier": result.triage.risk_tier,
        "to_tier": rollup.risk_tier,
        "lines_checked": len(fresh_analyses),
        "lines_moved": len(moved_lines),
        "lines": moved_lines,
    }


def guideline_change_impact(
    records: list[ApprovalRecord],
    *,
    classifier: Classifier | None = None,
    current_version: str | None = None,
) -> dict:
    """Rules-only re-triage over the awaiting queue.

    ``classifier`` defaults to the live triage rules; a caller (or a
    test simulating a corpus/rules change) may inject another rules
    function with the same signature. ``current_version`` defaults
    to the corpus version this code ships.
    """
    classify = classifier or classify_submission
    version = current_version or GUIDELINES_VERSION
    checked = 0
    unchanged = 0
    skipped = 0
    lines_checked = 0
    lines_moved = 0
    moved: list[dict] = []
    for record in records:
        result = record.result
        if result.status != "awaiting_underwriter":
            continue  # decided work is history; the queue is the scope
        if (record.submission or {}).get("lines"):
            # A multi-line bundle triages per line and gates as a
            # bundle; re-triage it per line too — each stored
            # sub-analysis against its own recomputed facts — and
            # report the affected lines, never a flattened pass.
            try:
                entry = _bundle_moved_entry(record, classify)
            except ValueError:
                skipped += 1
                continue
            checked += 1
            lines_checked += len(result.line_analyses)
            if entry is None:
                unchanged += 1
            else:
                lines_moved += entry["lines_moved"]
                moved.append(entry)
            continue
        try:
            sub = SubmissionInput.model_validate(record.submission)
        except Exception:
            skipped += 1  # a payload we cannot re-read cannot be re-triaged
            continue
        facts = compute_facts(sub, result.extractions)
        fresh = classify(sub, facts)
        checked += 1
        if fresh.outcome == result.triage.outcome and fresh.risk_tier == result.triage.risk_tier:
            unchanged += 1
            continue
        moved.append(
            {
                "submission_id": result.submission_id,
                "insured_name": result.insured_name,
                "multi_line": False,
                "triaged_under": result.guidelines_version or "unversioned",
                "from_outcome": result.triage.outcome,
                "to_outcome": fresh.outcome,
                "from_tier": result.triage.risk_tier,
                "to_tier": fresh.risk_tier,
                "reasons": explain_move(sub, facts, result.triage, fresh),
            }
        )
    return {
        "guidelines_version": version,
        "checked": checked,
        "unchanged_count": unchanged,
        "moved_count": len(moved),
        "moved": moved,
        "skipped": skipped,
        "lines_checked": lines_checked,
        "lines_moved": lines_moved,
        "note": (
            "Report only: rules-only re-triage against the recorded "
            "outcomes, bundles re-triaged per line. Each move names "
            "the guideline line it rests on. No stored result was changed."
        ),
    }
