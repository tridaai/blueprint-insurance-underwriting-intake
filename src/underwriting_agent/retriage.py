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

Attribution quotes the guideline line as it reads now; when a
corpus snapshot is kept for the version a submission triaged under
(see ``corpus.py``), each reason also quotes the line **as it read
then**, so an edit to the corpus never silently rewrites the
evidence for a move.

The provider pass (:func:`provider_change_impact`) is the opt-in
companion: it re-runs only the model cross-check on the moved
lines — bounded, costed, and reported separately — for deployments
that want the model's second opinion on exactly the submissions a
change touches. The rules report is never rewritten by it.
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


def _reason(
    guideline_id: str,
    needles: tuple[str, ...],
    driver: str,
    then_by_id: dict | None = None,
) -> dict:
    guideline = _GUIDELINE_BY_ID.get(guideline_id) or {}
    entry = {
        "guideline_id": guideline_id,
        "guideline_title": guideline.get("title", ""),
        "guideline_line": _sentence(guideline.get("text", ""), *needles),
        "driver": driver,
    }
    if then_by_id:
        # The same line as it read under the corpus version this
        # submission triaged under (from that version's kept
        # snapshot) — quoted beside today's line so an auditor sees
        # both wordings, never today's text dressed as history.
        then = then_by_id.get(guideline_id)
        if then is not None:
            entry["guideline_line_then"] = _sentence(
                then.get("text", ""), *needles
            )
    return entry


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
    sub: SubmissionInput,
    facts,
    recorded: Triage,
    fresh: Triage,
    then_by_id: dict | None = None,
) -> list[dict]:
    """Why the fresh rules outcome differs from the recorded one,
    as guideline citations: each reason names the guideline
    document, quotes the specific line in it, and states the
    computed driver (the trigger, the missing element, the loss
    figures) that puts this submission on that line.

    ``then_by_id`` is the guideline corpus as of the version the
    recorded outcome triaged under (from that version's kept
    snapshot, see ``corpus.py``); when given, each reason also
    carries ``guideline_line_then`` — the line as it read then.
    """
    reasons: list[dict] = []

    def add(guideline_id: str, needles: tuple[str, ...], driver: str) -> None:
        entry = _reason(guideline_id, needles, driver, then_by_id)
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
        fallback = {
            "guideline_id": guideline_id,
            "guideline_title": guideline.get("title", ""),
            "guideline_line": _sentence(guideline.get("text", "")),
            "driver": (
                "no single rule line accounts for the move; this is "
                "the governing guideline for the line under the "
                "current corpus"
            ),
        }
        if then_by_id:
            then = then_by_id.get(guideline_id)
            if then is not None:
                fallback["guideline_line_then"] = _sentence(then.get("text", ""))
        reasons.append(fallback)
    return reasons


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _then_by_id(
    snapshots: dict[str, list[dict]] | None, version: str, current_version: str
) -> dict | None:
    """The corpus as of ``version`` from its kept snapshot, as an
    id-keyed map — only when a snapshot is kept for a version other
    than the current one (for the current version, today's line IS
    the line as it read then, and reasons stay as they always were).
    """
    if not snapshots or not version or version == current_version:
        return None
    guidelines = snapshots.get(version)
    if not guidelines:
        return None
    return {g["id"]: g for g in guidelines}


def _bundle_moved_entry(
    record: ApprovalRecord,
    classify: Classifier,
    snapshots: dict[str, list[dict]] | None = None,
    current_version: str = "",
) -> dict | None:
    """Re-triage one awaiting bundle line by line.

    Returns the moved entry, or ``None`` when no line moved and the
    roll-up stands. Raises ``ValueError`` when the bundle payload
    cannot be re-read — the caller counts that as skipped, exactly
    like an unreadable single-line payload."""
    result = record.result
    then_by_id = _then_by_id(
        snapshots, result.guidelines_version or "", current_version
    )
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
                "reasons": explain_move(sub, facts, recorded_line, fresh, then_by_id),
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
    snapshots: dict[str, list[dict]] | None = None,
) -> dict:
    """Rules-only re-triage over the awaiting queue.

    ``classifier`` defaults to the live triage rules; a caller (or a
    test simulating a corpus/rules change) may inject another rules
    function with the same signature. ``current_version`` defaults
    to the corpus version this code ships. ``snapshots`` maps corpus
    versions to their kept snapshot text (see ``corpus.py``); a move
    on a submission that triaged under an older kept version gains
    the as-it-read-then guideline line beside today's.
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
                entry = _bundle_moved_entry(record, classify, snapshots, version)
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
                "reasons": explain_move(
                    sub,
                    facts,
                    result.triage,
                    fresh,
                    _then_by_id(
                        snapshots, result.guidelines_version or "", version
                    ),
                ),
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


# ---------------------------------------------------------------------------
# Provider-mode change impact — the model's second opinion on moved lines
# ---------------------------------------------------------------------------

# Default bound on the provider pass: at most this many moved lines
# get a model cross-check per call (``PROVIDER_IMPACT_MAX_LINES``
# overrides). The pass spends real money on a real provider, so the
# bound is part of the report, and the lines it leaves out are named,
# never silently dropped.
PROVIDER_IMPACT_DEFAULT_CAP = 25


def _moved_line_targets(
    records: list[ApprovalRecord], rules_report: dict
) -> list[dict]:
    """Flatten a rules report's moved set to per-line targets.

    Each target carries everything the provider pass needs to rebuild
    the line's facts: its record, its submission id (the line's own
    id for a bundle), the recorded (outcome, tier) pair, and the
    fresh rules pair the rules report itself published — the pass
    reports against those numbers, it does not recompute its own.
    """
    by_id = {r.result.submission_id: r for r in records}
    targets: list[dict] = []
    for entry in rules_report["moved"]:
        record = by_id.get(entry["submission_id"])
        if record is None:
            continue
        if entry.get("multi_line"):
            recorded_by_line = {
                a.submission_id: a.result.triage for a in record.result.line_analyses
            }
            for line in entry["lines"]:
                recorded = recorded_by_line.get(line["submission_id"])
                if recorded is None:
                    continue
                targets.append(
                    {
                        "record": record,
                        "submission_id": entry["submission_id"],
                        "line_submission_id": line["submission_id"],
                        "line_of_business": line["line_of_business"],
                        "recorded": recorded,
                        "fresh_pair": (line["to_outcome"], line["to_tier"]),
                    }
                )
        else:
            targets.append(
                {
                    "record": record,
                    "submission_id": entry["submission_id"],
                    "line_submission_id": entry["submission_id"],
                    "line_of_business": record.result.line_of_business,
                    "recorded": record.result.triage,
                    "fresh_pair": (entry["to_outcome"], entry["to_tier"]),
                }
            )
    return targets


def _line_facts(target: dict):
    """Rebuild one moved line's submission + computed facts — the
    same recomputation the rules pass itself performed."""
    record = target["record"]
    if target["line_submission_id"] == target["submission_id"]:
        sub = SubmissionInput.model_validate(record.submission)
        return sub, compute_facts(sub, record.result.extractions)
    model = SubmissionInput.model_validate(record.submission)
    subs = {s.submission_id: s for s in split_bundle(model)}
    sub = subs[target["line_submission_id"]]
    analysis = next(
        a
        for a in record.result.line_analyses
        if a.submission_id == target["line_submission_id"]
    )
    return sub, compute_facts(sub, analysis.result.extractions)


def _agreement_side(value: str, recorded_value: str, fresh_value: str) -> str:
    """Which side(s) a value agrees with: ``recorded``, ``rules``,
    ``both`` (a tier-only move, where the two sides share an
    outcome), or ``neither``."""
    hits = []
    if value == recorded_value:
        hits.append("recorded")
    if value == fresh_value:
        hits.append("rules")
    if len(hits) == 2:
        return "both"
    return hits[0] if hits else "neither"


def provider_change_impact(
    records: list[ApprovalRecord],
    backend,
    *,
    classifier: Classifier | None = None,
    current_version: str | None = None,
    snapshots: dict[str, list[dict]] | None = None,
    cap: int | None = None,
) -> dict:
    """The rules change-impact report, plus an opt-in provider pass.

    The rules report is computed exactly as
    :func:`guideline_change_impact` computes it and returned
    untouched under ``rules``. The provider pass then re-runs **only
    the model cross-check** — the same independent classify call the
    graph's cross-check node makes, over the same computed facts —
    on the lines that report moves, and no others: an unmoved line
    costs no call, ever. Each checked line reports the model's
    verdict beside the recorded outcome and the fresh rules outcome,
    which side it agrees with, and how the cross-check resolution
    policy would settle the pair — a second opinion, in its own
    section. It never rewrites the rules report, and it mutates no
    stored record.

    The pass is bounded: at most ``cap`` moved lines are checked
    (``PROVIDER_IMPACT_MAX_LINES``, default
    :data:`PROVIDER_IMPACT_DEFAULT_CAP`); the lines the cap leaves
    out are listed by id. Usage is accounted from the backend's own
    totals (delta over the pass) and costed with the telemetry
    price table — a model outside the table reports cost as None,
    never a guessed number.

    With no provider cross-check available — the offline mock
    backend, which has no classify capability — the pass reports
    itself **unavailable** and makes no calls, exactly as the LLM
    eval pack refuses offline. Nothing is substituted: the mock is
    never dressed up as a second opinion.
    """
    from .config import env_int
    from .crosscheck import resolve_triage
    from .graph import _classify_context
    from .model_backends import estimate_cost_usd

    classify = classifier or classify_submission
    rules = guideline_change_impact(
        records,
        classifier=classifier,
        current_version=current_version,
        snapshots=snapshots,
    )
    targets = _moved_line_targets(records, rules)
    backend_name = getattr(backend, "name", "") if backend is not None else ""
    classify_fn = (
        getattr(backend, "classify_with_llm", None) if backend is not None else None
    )
    if classify_fn is None:
        return {
            "rules": rules,
            "provider": {
                "available": False,
                "backend": backend_name,
                "reason": (
                    "no provider cross-check is available: the configured "
                    f"backend ({backend_name or 'none'}) has no LLM classify "
                    "capability — the offline fallback cannot second-guess "
                    "the rules, and nothing is substituted for a model that "
                    "did not run. Configure a real provider (MODEL_BACKEND, "
                    "see .env.example) to run this pass."
                ),
                "lines_moved": len(targets),
                "lines_cross_checked": 0,
                "checks": [],
                "lines_over_cap": [],
                "usage": None,
                "note": (
                    "Report only: the rules section above stands as the "
                    "change-impact report. The provider pass did not run."
                ),
            },
        }

    limit = (
        cap
        if cap is not None
        else env_int("PROVIDER_IMPACT_MAX_LINES", PROVIDER_IMPACT_DEFAULT_CAP)
    )
    limit = max(0, limit)
    usage_totals = getattr(backend, "usage_totals", None)
    before = (
        usage_totals()
        if callable(usage_totals)
        else {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    )
    checks: list[dict] = []
    for target in targets[:limit]:
        recorded = target["recorded"]
        check: dict = {
            "submission_id": target["submission_id"],
            "line_submission_id": target["line_submission_id"],
            "line_of_business": target["line_of_business"],
            "recorded_outcome": recorded.outcome,
            "recorded_tier": recorded.risk_tier,
            "rules_outcome": target["fresh_pair"][0],
            "rules_tier": target["fresh_pair"][1],
        }
        try:
            sub, facts = _line_facts(target)
            llm = classify_fn(_classify_context(sub, facts))
        except Exception as exc:  # a failed call is reported, never fatal
            check["error"] = f"{type(exc).__name__}: {exc}"
            checks.append(check)
            continue
        if llm is None:
            check["llm"] = None
            check["note"] = "the provider reply was unusable; no second opinion"
            checks.append(check)
            continue
        fresh = classify(sub, facts)
        _final, cross = resolve_triage(fresh, llm, backend_name)
        # Agreement is judged the way the cross-check itself judges:
        # on outcome (a tier difference is noted, not a disagreement),
        # with the tier's own agreement reported beside it.
        check.update(
            {
                "llm_outcome": llm["outcome"],
                "llm_tier": llm["risk_tier"],
                "llm_confidence": llm["confidence"],
                "agrees_with": _agreement_side(
                    llm["outcome"], recorded.outcome, target["fresh_pair"][0]
                ),
                "tier_agrees_with": _agreement_side(
                    llm["risk_tier"], recorded.risk_tier, target["fresh_pair"][1]
                ),
                "cross_check_resolution": cross.resolution,
            }
        )
        checks.append(check)
    after = (
        usage_totals()
        if callable(usage_totals)
        else {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    )
    delta = {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "calls")
    }
    model = getattr(backend, "_model", None) or ""
    return {
        "rules": rules,
        "provider": {
            "available": True,
            "backend": backend_name,
            "model": model,
            "cap": limit,
            "lines_moved": len(targets),
            "lines_cross_checked": len(checks),
            "checks": checks,
            "lines_over_cap": [
                {
                    "submission_id": t["submission_id"],
                    "line_submission_id": t["line_submission_id"],
                    "line_of_business": t["line_of_business"],
                }
                for t in targets[limit:]
            ],
            "usage": {
                **delta,
                "estimated_cost_usd": estimate_cost_usd(
                    model, delta["input_tokens"], delta["output_tokens"]
                ),
            },
            "note": (
                "Report only: the provider pass re-ran only the model "
                "cross-check on the lines the rules report moves, and "
                "is reported separately from it — the rules section was "
                "not recomputed or rewritten by the model, and no "
                "stored result was changed."
            ),
        },
    }
