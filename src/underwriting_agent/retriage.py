"""Guideline change-impact re-triage — a report, never a rewrite.

Guidelines change. When they do, every submission already sitting
in the queue was triaged under the old appetite, and nothing in the
old flow said which ones the change would move. This report closes
that gap: it re-runs **rules-only** triage over the submissions
still awaiting an underwriter, using each stored submission payload
and its stored extractions to recompute the facts, and lists the
ones whose outcome or risk tier would differ under the current
rules and corpus version.

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

from typing import Callable

from .facts import compute_facts
from .guidelines_data import GUIDELINES_VERSION
from .schemas import SubmissionInput, Triage
from .store import ApprovalRecord
from .triage import classify_submission

Classifier = Callable[..., Triage]


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
    moved: list[dict] = []
    for record in records:
        result = record.result
        if result.status != "awaiting_underwriter":
            continue  # decided work is history; the queue is the scope
        if (record.submission or {}).get("lines"):
            # A multi-line bundle triages per line and gates as a
            # bundle; a single rules pass over the bundle payload
            # would compare fiction against the roll-up. Per-line
            # change impact is a report of its own — until it
            # exists, bundles are counted as skipped, never misread.
            skipped += 1
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
                "triaged_under": result.guidelines_version or "unversioned",
                "from_outcome": result.triage.outcome,
                "to_outcome": fresh.outcome,
                "from_tier": result.triage.risk_tier,
                "to_tier": fresh.risk_tier,
            }
        )
    return {
        "guidelines_version": version,
        "checked": checked,
        "unchanged_count": unchanged,
        "moved_count": len(moved),
        "moved": moved,
        "skipped": skipped,
        "note": (
            "Report only: rules-only re-triage against the recorded "
            "outcomes. No stored result was changed."
        ),
    }
