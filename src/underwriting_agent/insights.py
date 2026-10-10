"""Inbox grouping and history scorecards — computed from the store.

The approval store is the system of record; everything in this
module is a deterministic projection of it, computed in code (never
by the model):

- **Inbox groups** — the intake queue grouped by insured and by
  broker: how many submissions each party has in the system, and
  the latest one's outcome, risk tier, and readiness. This is what
  an intake lead scans to see repeat insureds and busy brokers.
- **Scorecards** — per-broker and per-insured aggregates over their
  whole stored history: submission count, outcome mix, decision
  counts and approval rate, average computed completeness, and the
  referral triggers that fire most often for them. The scorecard
  for the parties on a new submission joins its diagnosis evidence
  (see ``diagnosis.memory_evidence_lines``), so the underwriter's
  packet shows the track record, not just this one case.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timedelta, timezone

from .config import env_float, env_int
from .diffing import _norm_desc
from .facts import _parse_date, normalize_line
from .store import ApprovalRecord, history_entry


def _party_name(record: ApprovalRecord, kind: str) -> str:
    entry = history_entry(record)
    return entry["insured_name"] if kind == "insured" else entry["broker_name"]


def inbox_groups(records: list[ApprovalRecord], kind: str) -> list[dict]:
    """Group stored submissions by insured (``kind="insured"``) or
    broker (``kind="broker"``).

    Records arrive in store order (oldest first); "latest" is the
    most recently analysed submission for the party. Groups are
    returned most-recent-activity first. Each group also carries
    its ``submissions`` — the drill-down list behind the counts, in
    store order, one entry per submission with its status.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for record in records:
        name = _party_name(record, kind)
        if not name:
            continue
        result = record.result
        group = groups.get(name)
        if group is None:
            group = {
                "name": name,
                "submission_count": 0,
                "outcome_counts": {},
                "awaiting_count": 0,
                "decided_count": 0,
                "submissions": [],
            }
            groups[name] = group
            order.append(name)
        group["submission_count"] += 1
        outcome = result.triage.outcome
        group["outcome_counts"][outcome] = group["outcome_counts"].get(outcome, 0) + 1
        if result.status == "awaiting_underwriter":
            group["awaiting_count"] += 1
        elif result.status in ("approved", "rejected"):
            group["decided_count"] += 1
        group["submissions"].append(
            {
                "submission_id": result.submission_id,
                "outcome": outcome,
                "risk_tier": result.triage.risk_tier,
                "readiness": result.facts.readiness,
                "completeness": result.facts.completeness_score,
                "status": result.status,
                "submitted_at": record.submission.get("submitted_at", ""),
            }
        )
        group.update(
            latest_submission_id=result.submission_id,
            latest_outcome=outcome,
            latest_risk_tier=result.triage.risk_tier,
            latest_readiness=result.facts.readiness,
            latest_completeness=result.facts.completeness_score,
            latest_status=result.status,
        )
        order.remove(name)
        order.append(name)  # most recent activity last
    return [groups[name] for name in reversed(order)]


def _approval_trend(decided_entries: list[dict]) -> dict:
    """The direction of a group's approval rate over its own
    decision history.

    The scorecard's approval rate is one number over everything;
    the trend reads the same decisions in the order they were made
    (store order): the approval rate over the earlier half of the
    decisions against the recent half. A delta beyond ±0.05 is a
    direction; inside it, the rate is flat. Fewer than two
    decisions cannot have a direction and say so instead of
    inventing one.
    """
    sequence = [
        1 if e["status"] == "approved" else 0 for e in decided_entries
    ]
    if len(sequence) < 2:
        return {
            "decisions": len(sequence),
            "early_rate": None,
            "recent_rate": None,
            "delta": None,
            "direction": "single_decision" if sequence else "insufficient_history",
        }
    mid = len(sequence) // 2
    early = sum(sequence[:mid]) / mid
    recent = sum(sequence[mid:]) / (len(sequence) - mid)
    delta = round(recent - early, 3)
    direction = (
        "improving" if delta > 0.05 else "declining" if delta < -0.05 else "flat"
    )
    return {
        "decisions": len(sequence),
        "early_rate": round(early, 3),
        "recent_rate": round(recent, 3),
        "delta": delta,
        "direction": direction,
    }


def scorecard(records: list[ApprovalRecord], kind: str, name: str) -> dict | None:
    """Aggregate one party's stored history, or None when they have none."""
    hits = [r for r in records if name and _party_name(r, kind) == name]
    if not hits:
        return None
    outcome_mix: Counter = Counter()
    triggers: Counter = Counter()
    approvals = rejections = 0
    completeness_total = 0.0
    for record in hits:
        result = record.result
        outcome_mix[result.triage.outcome] += 1
        triggers.update(result.triage.referral_triggers)
        if result.status == "approved":
            approvals += 1
        elif result.status == "rejected":
            rejections += 1
        completeness_total += result.facts.completeness_score
    decisions = approvals + rejections
    decided_entries = [
        {"status": record.result.status}
        for record in hits
        if record.result.status in ("approved", "rejected")
    ]
    return {
        "kind": kind,
        "name": name,
        "submissions": len(hits),
        "outcome_mix": dict(sorted(outcome_mix.items())),
        "decisions": decisions,
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": round(approvals / decisions, 3) if decisions else None,
        "approval_trend": _approval_trend(decided_entries),
        "avg_completeness": round(completeness_total / len(hits), 3),
        "top_referral_reasons": [
            {"reason": reason, "count": count}
            for reason, count in triggers.most_common(3)
        ],
    }


def all_scorecards(records: list[ApprovalRecord]) -> dict:
    """Scorecards for every insured and broker in the store,
    busiest first."""
    cards = {"insureds": [], "brokers": []}
    for kind, key in (("insured", "insureds"), ("broker", "brokers")):
        names = {_party_name(r, kind) for r in records}
        built = [scorecard(records, kind, name) for name in names if name]
        cards[key] = sorted(
            (c for c in built if c is not None),
            key=lambda c: (-c["submissions"], c["name"]),
        )
    return cards


def scorecards_for(
    records: list[ApprovalRecord], *, insured_name: str, broker_name: str
) -> dict:
    """The scorecards for one submission's two parties (either may
    be absent — a first submission from a party has no card)."""
    cards = {}
    insured_card = scorecard(records, "insured", insured_name)
    if insured_card is not None:
        cards["insured"] = insured_card
    broker_card = scorecard(records, "broker", broker_name)
    if broker_card is not None:
        cards["broker"] = broker_card
    return cards


def format_scorecard_line(card: dict) -> str:
    """One evidence-style line for a scorecard."""
    mix = ", ".join(
        f"{count} {outcome}" for outcome, count in card["outcome_mix"].items()
    )
    line = (
        f"{card['kind']} scorecard: {card['submissions']} prior submission(s) "
        f"for {card['name']} (outcomes: {mix}; avg completeness "
        f"{card['avg_completeness']}"
    )
    if card["decisions"]:
        line += (
            f"; {card['approvals']} approved / {card['rejections']} rejected "
            f"— approval rate {card['approval_rate']}"
        )
    else:
        line += "; no underwriter decisions yet"
    if card["top_referral_reasons"]:
        reasons = ", ".join(
            f"{r['reason']} ×{r['count']}" for r in card["top_referral_reasons"]
        )
        line += f"; top referral reasons: {reasons}"
    return line + ")"


# ---------------------------------------------------------------------------
# Broker submission-quality report
# ---------------------------------------------------------------------------


def broker_quality_report(records: list[ApprovalRecord], broker_name: str) -> dict | None:
    """A per-broker quality report, composed from the scorecard data.

    Where the scorecard is the aggregate, this is the feedback view:
    is the broker's submission quality moving (completeness of their
    recent submissions against their earlier ones), how often do
    their documents disagree with their forms (the extraction
    discrepancy rate), and which required elements do they most
    often leave out. Returns None for an unknown broker. The report
    carries its own drafted text (``draft``) — composed here, and,
    like every draft in this system, never sent by the agent. The
    draft answers to the same drafting discipline as the letter
    and the status note (see :func:`_discipline_broker_draft`):
    guardrail-checked, steered once on a wording-only failure,
    refused on substance — a refused report serves its computed
    sections with ``draft`` None and the refusal named, never a
    draft that failed.
    """
    hits = [r for r in records if broker_name and _party_name(r, "broker") == broker_name]
    if not hits:
        return None
    card = scorecard(records, "broker", broker_name)
    completeness = [r.result.facts.completeness_score for r in hits]
    count = len(hits)
    if count >= 2:
        split = count // 2
        earlier_avg = round(sum(completeness[:split]) / split, 3)
        recent_avg = round(sum(completeness[split:]) / (count - split), 3)
        delta = round(recent_avg - earlier_avg, 3)
        direction = (
            "improving" if delta > 0.02 else "declining" if delta < -0.02 else "flat"
        )
        trend = {
            "direction": direction,
            "delta": delta,
            "earlier_avg": earlier_avg,
            "recent_avg": recent_avg,
        }
    else:
        trend = {
            "direction": "single_submission",
            "delta": None,
            "earlier_avg": None,
            "recent_avg": None,
        }
    discrepancy_counts = [r.result.facts.discrepancy_count for r in hits]
    missing: Counter = Counter()
    for record in hits:
        missing.update(record.result.facts.missing_elements)
    report = {
        "broker": broker_name,
        "submissions": count,
        "avg_completeness": card["avg_completeness"] if card else None,
        "approval_rate": card["approval_rate"] if card else None,
        "outcome_mix": card["outcome_mix"] if card else {},
        "completeness_trend": trend,
        "discrepancy_total": sum(discrepancy_counts),
        "discrepancy_rate": round(sum(discrepancy_counts) / count, 3),
        "discrepancy_share": round(
            sum(1 for c in discrepancy_counts if c > 0) / count, 3
        ),
        "top_missing_elements": [
            {"element": element, "count": hits_count}
            for element, hits_count in missing.most_common(5)
        ],
    }
    draft, drafting = _discipline_broker_draft(
        compose_broker_quality_draft(report), broker_name
    )
    report["draft"] = draft
    report.update(drafting)
    return report


#: Store document key for the broker-quality drafting events
#: (see :func:`_broker_quality_repair_aggregate`): one event per
#: report the service drafts, so the drafting discipline's
#: outcomes on this surface are countable in /metrics.
BROKER_QUALITY_DRAFTING_KEY = "broker_quality_drafting"


def _discipline_broker_draft(
    text: str, broker_name: str
) -> tuple[str | None, dict]:
    """One broker-quality draft through the drafting discipline.

    The report is composed in code from record prose — the
    broker's name, the missing-element labels its submissions
    carried — so, like the status note, it is guardrail-checked
    before it is served (``guardrails.validate_broker_quality_draft``)
    and a **wording-only** failure earns one steered redraft in
    the report's own shapes (``letter_repair.steer_broker_quality_draft``),
    the repair recorded with the original validation, the
    guidance, and the families rewritten. Anything else — bind
    language, a premium figure, a promise, PII — is refused:
    the report serves with ``draft`` None and a
    ``draft_refusal`` naming the failed checks, because the
    report's computed sections are facts an operator may still
    read, but a draft that failed on substance is never served
    as a draft. Returns the served draft (or None) and the
    drafting block the report carries: the final
    ``draft_validation``, and the ``draft_repair`` or
    ``draft_refusal`` when either happened — each stamped
    ``source: "broker_quality"``, the third drafting surface
    in the repair aggregate's vocabulary.
    """
    from .guardrails import (
        letter_wording_hits,
        validate_broker_quality_draft,
    )
    from .letter_repair import (
        broker_quality_repair_guidance,
        steer_broker_quality_draft,
    )

    validation = validate_broker_quality_draft(text, broker_name)
    drafting: dict = {
        "draft_validation": validation.model_dump(mode="json")
    }
    if validation.passed:
        return text, drafting
    guidance = broker_quality_repair_guidance(validation, text)
    steered = (
        steer_broker_quality_draft(text, validation) if guidance else None
    )
    if steered is not None:
        steered_validation = validate_broker_quality_draft(
            steered, broker_name
        )
        if steered_validation.passed:
            drafting["draft_repair"] = {
                "repair_attempted": True,
                "repaired": True,
                "source": "broker_quality",
                "original_validation": validation.model_dump(mode="json"),
                "repair_guidance": guidance,
                "repair_families": sorted(
                    family
                    for family, hit in letter_wording_hits(text).items()
                    if hit
                ),
            }
            drafting["draft_validation"] = steered_validation.model_dump(
                mode="json"
            )
            return steered, drafting
    failed_checks = [c.name for c in validation.checks if not c.passed]
    drafting["draft_refusal"] = {
        "refused_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source": "broker_quality",
        "failed_checks": failed_checks,
        "steered": steered is not None,
        "repair_families": sorted(
            family
            for family, hit in letter_wording_hits(text).items()
            if hit
        )
        if "no_forward_motion_in_letter" in failed_checks
        else [],
    }
    return None, drafting


def compose_broker_quality_draft(report: dict) -> str:
    """The quality report as a drafted document.

    Written the way the packet is written — facts from the record,
    no promises, no timelines — and headed as what it is: an
    internal draft a human may choose to share. It is never sent.
    """
    trend = report["completeness_trend"]
    if trend["direction"] == "single_submission":
        trend_line = (
            "Completeness trend: only one submission on record — "
            "no trend can be read yet."
        )
    else:
        trend_line = (
            f"Completeness trend: {trend['direction']} — earlier submissions "
            f"averaged {trend['earlier_avg']}, recent submissions "
            f"{trend['recent_avg']} (delta {trend['delta']:+.3f})."
        )
    if report["top_missing_elements"]:
        missing_line = "Most often missing: " + ", ".join(
            f"{entry['element']} (×{entry['count']})"
            for entry in report["top_missing_elements"]
        ) + "."
    else:
        missing_line = "No required elements were missing across these submissions."
    lines = [
        f"DRAFT — submission quality report for {report['broker']} (internal draft; not sent)",
        "",
        f"Submissions on record: {report['submissions']} · average computed "
        f"completeness {report['avg_completeness']}",
        trend_line,
        f"Extraction discrepancies: {report['discrepancy_total']} across "
        f"{report['submissions']} submission(s) — "
        f"{report['discrepancy_rate']} per submission; "
        f"{report['discrepancy_share']} of submissions carried at least one.",
        missing_line,
        "",
        "This draft was composed from the submission records for internal "
        "review. It is shared with the broker only if a human reviews and "
        "sends it — the agent never sends it.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics + audit export
# ---------------------------------------------------------------------------


def _letter_advisory_aggregate(records: list[ApprovalRecord]) -> dict:
    """The letter-advisory aggregate over the store's runs.

    One computation, several readers: ``/metrics`` reports the
    whole family as the drafting-quality trend (see
    :func:`compute_metrics`), and the ops digest carries its two
    headline counts — runs that drafted advisories, runs whose
    advisories the steered redraft cleared — in the counts its
    snapshots keep, so the kept history trends drafting quality
    alongside the operations counts instead of the trend living
    only as today's point-in-time rate.
    """
    advisory_runs = 0
    advisory_total = 0
    advisory_families: Counter = Counter()
    advisory_repaired_runs = 0
    for record in records:
        advisories = record.result.letter_advisories or []
        if advisories:
            advisory_runs += 1
            advisory_total += len(advisories)
            for advisory in advisories:
                advisory_families[advisory.get("family") or "unknown"] += 1
        if record.result.advisory_repaired:
            advisory_repaired_runs += 1
    return {
        "runs_with_advisories": advisory_runs,
        "total": advisory_total,
        "per_run": (
            round(advisory_total / len(records), 3) if records else None
        ),
        "by_family": dict(sorted(advisory_families.items())),
        "repaired_runs": advisory_repaired_runs,
        "clear_rate": (
            round(advisory_repaired_runs / advisory_runs, 3)
            if advisory_runs
            else None
        ),
    }


def _status_note_repair_aggregate(records: list[ApprovalRecord]) -> dict:
    """The status-note repair aggregate over the store's records.

    The note-side twin of :func:`_letter_advisory_aggregate`:
    since round 15 a wording-only status-note failure earns one
    steered redraft, recorded on the stored note, and since
    round 18 a refusal is recorded on the record too — so the
    whole drafting loop is countable: notes drafted, notes the
    steering repaired (and the families it rewrote), drafts
    refused, and the refusal split — refused after a steered
    candidate still failed the gate, against refused outright
    on substance (bind language, a premium, a promise, PII:
    never steered around). A repaired note means interpolated
    record prose carried a forbidden phrasing; a rising repair
    or refusal count is an intake-data-quality signal exactly
    parallel to the letter-advisory trend, and it was invisible
    while repairs lived note by note and refusals evaporated
    into the raised error. ``by_family`` counts the families on
    repairs recorded since families were kept (round 18) plus
    the families on refusals; earlier repaired notes count in
    ``repaired_notes`` but carry no family vocabulary to count.
    The refusal split's denominator discipline mirrors the
    advisories': the steer success rate is repaired notes over
    every steer attempted (repairs + refusals after steering).
    """
    notes_drafted = 0
    repaired_notes = 0
    families: Counter = Counter()
    refusals = 0
    refusals_after_steering = 0
    refusal_checks: Counter = Counter()
    for record in records:
        note = record.status_note
        if note:
            notes_drafted += 1
            if note.get("repaired"):
                repaired_notes += 1
                for family in note.get("repair_families") or []:
                    families[family] += 1
        for refusal in record.status_note_refusals or []:
            refusals += 1
            if refusal.get("steered"):
                refusals_after_steering += 1
            for name in refusal.get("failed_checks") or []:
                refusal_checks[name] += 1
            for family in refusal.get("repair_families") or []:
                families[family] += 1
    steer_attempts = repaired_notes + refusals_after_steering
    return {
        "notes_drafted": notes_drafted,
        "repaired_notes": repaired_notes,
        "by_family": dict(sorted(families.items())),
        "refusals": refusals,
        "refusals_after_steering": refusals_after_steering,
        "refusals_unsteered": refusals - refusals_after_steering,
        "by_refused_check": dict(sorted(refusal_checks.items())),
        "steer_attempts": steer_attempts,
        "steer_success_rate": (
            round(repaired_notes / steer_attempts, 3)
            if steer_attempts
            else None
        ),
    }


def _broker_quality_repair_aggregate(events: list[dict]) -> dict:
    """The broker-quality aggregate over the drafting events.

    The third surface in the drafting-repair family: the letter
    has its advisories, the status note its repairs and
    refusals, and the broker quality report — drafted on read,
    from record prose — records one event per drafting (see
    ``service.broker_quality_report``): clean, repaired (with
    the families the steering rewrote), or refused (with the
    failed checks and whether a steered candidate was produced
    first). The shape mirrors the note aggregate's — drafted /
    repaired / by-family / refusals / the refusal-vs-steer
    split / steer success rate — in the report's vocabulary, so
    the surfaces read side by side and a repair rate on any of
    them means the same thing: interpolated record prose
    carried a forbidden phrasing, and the drafting discipline
    caught it before a broker could see it.
    """
    drafted = len(events)
    repaired_reports = 0
    families: Counter = Counter()
    refusal_events: list[dict] = []
    refusal_checks: Counter = Counter()
    for event in events:
        for family in event.get("repair_families") or []:
            families[family] += 1
        if event.get("outcome") == "repaired":
            repaired_reports += 1
        elif event.get("outcome") == "refused":
            refusal_events.append(event)
            for name in event.get("failed_checks") or []:
                refusal_checks[name] += 1
    refusals_after_steering = sum(
        1 for event in refusal_events if event.get("steered")
    )
    steer_attempts = repaired_reports + refusals_after_steering
    return {
        "reports_drafted": drafted,
        "repaired_reports": repaired_reports,
        "by_family": dict(sorted(families.items())),
        "refusals": len(refusal_events),
        "refusals_after_steering": refusals_after_steering,
        "refusals_unsteered": len(refusal_events) - refusals_after_steering,
        "by_refused_check": dict(sorted(refusal_checks.items())),
        "steer_attempts": steer_attempts,
        "steer_success_rate": (
            round(repaired_reports / steer_attempts, 3)
            if steer_attempts
            else None
        ),
    }


def compute_metrics(
    records: list[ApprovalRecord],
    broker_drafting_events: list[dict] | None = None,
) -> dict:
    """Operational aggregates over the whole store.

    Run and decision counts, outcome and tier mix, guardrail and
    reviewer signals, the **letter-advisory trend** (below), the
    **status-note repair family** (the note-side twin: steered
    repairs, refusals, and the refusal-vs-steer split — see
    :func:`_status_note_repair_aggregate` — with the broker
    quality report's repairs and refusals nested in the same
    family as its own named source, counted from the drafting
    events the caller passes), and —
    from each run's recorded telemetry — model calls, tokens,
    estimated cost, and latency. Token/cost figures only include
    runs whose backend reported them (the offline fallback reports
    tokens as None rather than inventing numbers, and says so via
    ``costed_runs``/``token_runs``).

    The advisory trend aggregates what each run recorded about
    its broker letter's borderline phrasings (see
    ``letter_repair.letter_advisories``): how many runs drafted a
    letter a judge would still read as a commitment, how many
    advisories that was, the family mix, and how often the one
    steered redraft cleared them (``advisory_repaired`` — the
    shipped letter carries none). Advisories never block, so
    without this aggregate they are invisible until a judge
    finds them; as a metric family they are the drafting-quality
    signal the team watches — a rising per-run rate is a model
    or prompt regression arriving *before* it becomes guardrail
    failures.
    """
    statuses: Counter = Counter()
    outcomes: Counter = Counter()
    tiers: Counter = Counter()
    model_calls = 0
    input_tokens = 0
    output_tokens = 0
    token_runs = 0
    costs: list[float] = []
    latencies: list[float] = []
    guardrail_failures = 0
    reviewer_blocks = 0
    for record in records:
        result = record.result
        statuses[result.status] += 1
        outcomes[result.triage.outcome] += 1
        tiers[result.triage.risk_tier] += 1
        if result.validation is not None and not result.validation.passed:
            guardrail_failures += 1
        if result.reviewer_blocked:
            reviewer_blocks += 1
        telemetry = result.telemetry
        if telemetry is None:
            continue
        model_calls += telemetry.model_calls
        latencies.append(telemetry.latency_seconds)
        if telemetry.input_tokens is not None:
            token_runs += 1
            input_tokens += telemetry.input_tokens
            output_tokens += telemetry.output_tokens or 0
        if telemetry.estimated_cost_usd is not None:
            costs.append(telemetry.estimated_cost_usd)
    dispatch_attempts = 0
    dispatch_sent = 0
    dispatch_failed = 0
    for record in records:
        dispatch_attempts += len(record.dispatch_attempts or [])
        if record.dispatch_status == "sent":
            dispatch_sent += 1
        elif record.dispatch_status == "failed":
            dispatch_failed += 1
    decisions = statuses.get("approved", 0) + statuses.get("rejected", 0)
    return {
        "runs": len(records),
        "awaiting_underwriter": statuses.get("awaiting_underwriter", 0),
        "approved": statuses.get("approved", 0),
        "rejected": statuses.get("rejected", 0),
        "dispatch": {
            "attempts": dispatch_attempts,
            "sent_records": dispatch_sent,
            "failed_records": dispatch_failed,
        },
        "approval_rate": (
            round(statuses.get("approved", 0) / decisions, 3) if decisions else None
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "risk_tiers": dict(sorted(tiers.items())),
        "guardrail_failures": guardrail_failures,
        "reviewer_blocks": reviewer_blocks,
        "letter_advisories": _letter_advisory_aggregate(records),
        "status_note_repairs": {
            **_status_note_repair_aggregate(records),
            # The third drafting surface, named as its own
            # source inside the same family: the broker quality
            # report's repairs and refusals, counted from the
            # drafting events the service records per report.
            "broker_quality": _broker_quality_repair_aggregate(
                broker_drafting_events or []
            ),
        },
        "model_calls": model_calls,
        "token_runs": token_runs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "costed_runs": len(costs),
        "estimated_cost_usd": round(sum(costs), 4) if costs else None,
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
            "total": round(sum(latencies), 3),
        },
    }


AUDIT_COLUMNS = [
    "submission_id",
    "insured_name",
    "broker_name",
    "line_of_business",
    "outcome",
    "risk_tier",
    "readiness",
    "completeness_score",
    "status",
    "decided_by",
    "decision_reason",
    "decided_at",
    "dispatch_status",
    "dispatch_attempts",
    "guardrails_passed",
    "reviewer_blocked",
    "backend",
    "model",
    "model_calls",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "latency_seconds",
]


def audit_rows(records: list[ApprovalRecord]) -> list[dict]:
    """The audit trail: one flat row per stored submission — what was
    analysed, what the agent concluded, what the human decided (and
    why), and what the run cost."""
    rows: list[dict] = []
    for record in records:
        result = record.result
        entry = history_entry(record)
        telemetry = result.telemetry
        rows.append(
            {
                "submission_id": result.submission_id,
                "insured_name": entry["insured_name"],
                "broker_name": entry["broker_name"],
                "line_of_business": entry["line"],
                "outcome": result.triage.outcome,
                "risk_tier": result.triage.risk_tier,
                "readiness": result.facts.readiness,
                "completeness_score": result.facts.completeness_score,
                "status": result.status,
                "decided_by": result.decided_by or "",
                "decision_reason": result.decision_reason or "",
                "decided_at": result.decided_at or "",
                "dispatch_status": record.dispatch_status or "",
                "dispatch_attempts": len(record.dispatch_attempts or []),
                "guardrails_passed": (
                    result.validation.passed if result.validation else None
                ),
                "reviewer_blocked": result.reviewer_blocked,
                "backend": telemetry.backend if telemetry else "",
                "model": telemetry.model if telemetry else "",
                "model_calls": telemetry.model_calls if telemetry else 0,
                "input_tokens": telemetry.input_tokens if telemetry else None,
                "output_tokens": telemetry.output_tokens if telemetry else None,
                "estimated_cost_usd": (
                    telemetry.estimated_cost_usd if telemetry else None
                ),
                "latency_seconds": (
                    round(telemetry.latency_seconds, 3) if telemetry else None
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Queue aging
# ---------------------------------------------------------------------------

# (label, min_age_days, max_age_days) — max None is open-ended.
QUEUE_AGE_BUCKETS = [
    ("0-1 days", 0, 1),
    ("2-3 days", 2, 3),
    ("4-7 days", 4, 7),
    ("8-14 days", 8, 14),
    ("15+ days", 15, None),
]


# SLA budgets by risk tier, in days an awaiting submission may sit
# before the queue flags it breached. The budget SHRINKS as the tier
# rises: an elevated-risk file needs a senior underwriter's eyes
# soonest, and the longer a hard case sits the worse its options
# get; a low-tier file that triaged clean can wait out the week.
# Every budget is tunable per deployment (QUEUE_SLA_DAYS_<TIER>)
# without touching code; a tier the map does not know falls back to
# the moderate budget rather than to no budget at all.
QUEUE_SLA_DEFAULTS = {"elevated": 2, "high": 3, "moderate": 5, "low": 7}


def sla_budgets() -> dict[str, int]:
    """The queue SLA budgets by risk tier, from the environment."""
    return {
        tier: max(0, env_int(f"QUEUE_SLA_DAYS_{tier.upper()}", default))
        for tier, default in QUEUE_SLA_DEFAULTS.items()
    }


def _sla_for(item: dict, budgets: dict[str, int]) -> dict:
    """One queue item's SLA state against its tier's budget.

    ``within`` / ``breached`` once the age is known; ``unknown``
    when the submitted date cannot be parsed — an unparseable date
    is already surfaced by the aging view, and inventing an SLA
    state for it would hide that.
    """
    budget = budgets.get(item["risk_tier"], budgets["moderate"])
    age = item.get("age_days")
    if age is None:
        return {"sla_budget_days": budget, "sla_state": "unknown", "sla_days_over": None}
    over = age - budget
    return {
        "sla_budget_days": budget,
        "sla_state": "breached" if over > 0 else "within",
        "sla_days_over": max(0, over),
    }


def latest_status_note_refusal(record: ApprovalRecord) -> dict | None:
    """The latest refused status-note draft on a record, or None.

    A refused draft is recorded on the record before the refusal
    raises (see ``service.draft_status_note``) so the drafting
    quality can be trended in ``/metrics`` — but the aggregate
    is not where the work is. The underwriter who opens the
    case itself — the queue row, the case read — meets the
    refusal here: when the last draft was refused, which checks
    it failed, whether a steered candidate was produced and
    still failed (``steered``) or the draft was refused outright
    on substance, the wording families involved, and how many
    refusals the record carries in all. Silence where a note
    should be reads as "nobody tried"; this says somebody did,
    and what the record's own prose did to the draft — which is
    what the person reading it can actually fix.
    """
    refusals = record.status_note_refusals or []
    if not refusals:
        return None
    latest = refusals[-1]
    return {
        "refused_at": latest.get("refused_at", ""),
        "failed_checks": list(latest.get("failed_checks") or []),
        "steered": bool(latest.get("steered")),
        "repair_families": list(latest.get("repair_families") or []),
        "refusal_count": len(refusals),
    }


def queue_item_for(
    record: ApprovalRecord,
    now: datetime | None = None,
    budgets: dict[str, int] | None = None,
) -> dict | None:
    """The queue-view item for one record, or None when the record
    is not awaiting an underwriter. The single-item builder behind
    :func:`queue_aging` — the status-note drafting reads the same
    projection, so the note and the queue can never disagree about
    a case's age or SLA state."""
    result = record.result
    if result.status != "awaiting_underwriter":
        return None
    now = now or datetime.now(timezone.utc)
    budgets = budgets if budgets is not None else sla_budgets()
    entry = history_entry(record)
    raw = (record.submission or {}).get("submitted_at", "")
    submitted = _parse_date(raw)
    item = {
        "submission_id": result.submission_id,
        "insured_name": entry["insured_name"],
        "broker_name": entry["broker_name"],
        "outcome": result.triage.outcome,
        "risk_tier": result.triage.risk_tier,
        "submitted_at": raw,
        "age_days": None,
    }
    if submitted is not None:
        item["age_days"] = max(0, (now.date() - submitted).days)
    item.update(_sla_for(item, budgets))
    item["blocking_lines"] = _blocking_line_entries(result, item["age_days"])
    # The breach-webhook notice, when one has fired for this record
    # (service.notify_sla_breaches) — the queue shows whether a
    # breached case has been announced, not just that it breached.
    item["sla_breach_notice"] = record.sla_breach_notice
    # The latest refused status-note draft, when one was refused
    # for this record (see :func:`latest_status_note_refusal`) —
    # the queue row is where the underwriter who can fix the
    # record's prose meets the refusal, not only /metrics.
    item["status_note_refusal"] = latest_status_note_refusal(record)
    # The escalated follow-through flag is attached by
    # :func:`queue_aging`, which can see every record: the note
    # rotting on this insured may live on an older, decided record.
    # None here — the single-item builder cannot know.
    item["escalated_note"] = None
    # Likewise the breach's new/continuing flag against the
    # latest digest snapshot: a single item cannot know what the
    # snapshot held. None here — :func:`queue_aging` attaches it
    # when snapshots are passed.
    item["breach_since_snapshot"] = None
    return item


def _blocking_line_entries(result, age_days: int | None) -> list[dict]:
    """A bundle queue item's blocking lines, with their wait.

    The gate record names the lines whose own facts are missing
    required elements (``AgentResult.blocking_lines``); the queue
    adds what the underwriter actually chases: each line's missing
    items, from that line's stored sub-analysis, and how long the
    bundle has waited on the line. A blocking line has blocked
    since the bundle was submitted — the line cannot complete
    before the broker answers, and the broker was asked at intake —
    so its age is the bundle's queue age, stated per line so a
    bundle stranded on one line reads differently from a wholly
    unready submission. Empty for single-line submissions and for
    bundles no line blocks.
    """
    if not result.multi_line or not result.blocking_lines:
        return []
    by_line = {
        normalize_line(a.line_of_business): a for a in result.line_analyses
    }
    entries = []
    for line in result.blocking_lines:
        analysis = by_line.get(normalize_line(line))
        entries.append(
            {
                "line_of_business": line,
                "submission_id": analysis.submission_id if analysis else "",
                "age_days": age_days,
                "missing_items": (
                    list(analysis.result.facts.missing_elements)
                    if analysis
                    else []
                ),
            }
        )
    return entries


def queue_aging(
    records: list[ApprovalRecord],
    now: datetime | None = None,
    *,
    sla_budgets_map: dict[str, int] | None = None,
    follow_escalation_days: int | None = None,
    digest_snapshots: list[dict] | None = None,
) -> dict:
    """The awaiting queue by age — the stranded-queue early warning.

    Age is whole days from the submission's ``submitted_at`` to
    today (the same date arithmetic ``facts.submission_age_days``
    uses). Every awaiting submission lands in exactly one bucket —
    or in ``unknown_age`` when its submitted date cannot be parsed,
    which is itself worth seeing. Decided records are not queue.

    Every item also carries its **SLA state**: the budget for its
    risk tier (see :data:`QUEUE_SLA_DEFAULTS`), whether it is within
    or past it, and by how many days. Breached items are listed
    again in ``breaches``, worst overrun first — that list is the
    queue's worklist, and it is what the drafted status note (see
    :func:`compose_status_note`) is for.

    And every item carries its insured's **escalated-note flag**:
    when any record for the same insured holds a status note whose
    follow-through loop has aged past the escalation window (see
    :func:`follow_through_digest`), the item names that note and
    its age — the rotting loop shows where the work happens, not
    only in the digest. All escalated notes are listed again in
    ``escalated_notes``, each with the awaiting submissions its
    insured currently has in this queue (possibly none — the case
    the note was drafted on may be decided while the loop rots).

    With ``digest_snapshots`` — the kept ops-digest snapshots —
    the breach list is also read **against the latest snapshot**:
    every breached item is flagged ``breach_since_snapshot`` as
    ``"new"`` (not among the snapshot's breached ids — a fire
    that started since the last kept picture) or ``"continuing"``
    (already breached when it was taken — an old fire), and the
    view carries ``breach_movement``: the new, continuing, and
    cleared id sets (cleared = breached at the snapshot, no
    longer in this queue's breaches — decided, most often).
    "New" is measured against the snapshot's own budgets and
    moment, the same caveat the digest's movement block carries.
    Without snapshots the block is None and the flags stay None —
    direction is absent, never invented.

    And an item whose own breach notice was **acknowledged** —
    the operator called the broker, closing the "nobody was
    told" gap — carries that acknowledgement once the case has
    still not been decided at **twice** the follow-through
    escalation window (``acknowledged_escalation``: who called,
    when, their note, the acknowledgement's age), and the cases
    are listed again in ``escalated_acknowledgements``. The
    first window resurfaces the case in the ops digest's
    still-waiting section, where operators read; the second
    rung puts it here, where the underwriter who finally picks
    it up works — with the first touch on the record, so the
    second touch never pretends to be the first. A decided case
    drops off by the acknowledged read's own construction (the
    exhaustion worklist only ever holds undecided cases).
    """
    now = now or datetime.now(timezone.utc)
    budgets = sla_budgets_map if sla_budgets_map is not None else sla_budgets()
    buckets: dict[str, list[dict]] = {label: [] for label, _, _ in QUEUE_AGE_BUCKETS}
    unknown: list[str] = []
    by_outcome: Counter = Counter()
    oldest: dict | None = None
    breaches: list[dict] = []
    all_items: list[dict] = []
    blocked_bundles = 0
    total = 0
    for record in records:
        result = record.result
        if result.status != "awaiting_underwriter":
            continue
        total += 1
        by_outcome[result.triage.outcome] += 1
        item = queue_item_for(record, now, budgets)
        assert item is not None  # status checked above
        all_items.append(item)
        if item["blocking_lines"]:
            blocked_bundles += 1
        if item["age_days"] is None:
            unknown.append(result.submission_id)
            continue
        age = item["age_days"]
        for label, low, high in QUEUE_AGE_BUCKETS:
            if age >= low and (high is None or age <= high):
                buckets[label].append(item)
                break
        if item["sla_state"] == "breached":
            breaches.append(item)
        if oldest is None or age > oldest["age_days"]:
            oldest = item
    breaches.sort(key=lambda i: (-(i["sla_days_over"] or 0), -i["age_days"], i["submission_id"]))
    # Breach movement against the latest digest snapshot: which
    # of today's fires are new since the last kept picture, which
    # were already burning, and which of the snapshot's fires are
    # no longer in this list at all.
    breach_movement: dict | None = None
    if digest_snapshots:
        snapshot = digest_snapshots[-1]
        snapshot_breached = set(
            (snapshot.get("sections") or {}).get("breached_queue") or []
        )
        current_breached = {item["submission_id"] for item in breaches}
        for item in breaches:
            item["breach_since_snapshot"] = (
                "continuing"
                if item["submission_id"] in snapshot_breached
                else "new"
            )
        breach_movement = {
            "since_snapshot": snapshot.get("snapshot_id", ""),
            "snapshot_taken_at": snapshot.get("taken_at", ""),
            "snapshot_taken_by": snapshot.get("taken_by", ""),
            "new_breaches": sorted(current_breached - snapshot_breached),
            "continuing_breaches": sorted(
                current_breached & snapshot_breached
            ),
            "cleared_breaches": sorted(snapshot_breached - current_breached),
        }
    # Escalated follow-through, matched by insured: the note lives
    # on its own record (often an older, decided submission), while
    # the flag belongs on the work the underwriter is looking at
    # now. One flag per item — the insured's oldest rotting loop.
    follow = follow_through_digest(
        records, now, escalation_days=follow_escalation_days
    )
    escalated_by_insured: dict[str, dict] = {}
    for entry in follow["escalated"]:
        name = entry["insured_name"]
        if not name:
            continue
        current = escalated_by_insured.get(name)
        if current is None or (entry["note_age_days"] or 0) > (
            current["note_age_days"] or 0
        ):
            escalated_by_insured[name] = entry
    awaiting_by_insured: dict[str, list[str]] = {}
    flagged = 0
    for item in all_items:
        note = escalated_by_insured.get(item["insured_name"])
        if note is not None:
            item["escalated_note"] = {
                "submission_id": note["submission_id"],
                "drafted_at": note["drafted_at"],
                "note_age_days": note["note_age_days"],
                "state": note["state"],
            }
            flagged += 1
        if item["insured_name"]:
            awaiting_by_insured.setdefault(item["insured_name"], []).append(
                item["submission_id"]
            )
    escalated_notes = [
        {
            "submission_id": entry["submission_id"],
            "insured_name": entry["insured_name"],
            "broker_name": entry["broker_name"],
            "drafted_at": entry["drafted_at"],
            "note_age_days": entry["note_age_days"],
            "state": entry["state"],
            "still_outstanding_items": entry["still_outstanding_items"],
            "awaiting_submission_ids": sorted(
                awaiting_by_insured.get(entry["insured_name"], [])
            ),
        }
        for entry in follow["escalated"]
    ]
    # Acknowledged breach notices on their second rung: the
    # acknowledgement closed the telling gap, and two
    # follow-through windows later the case is still undecided.
    # The flag rides the item itself (the notice lives on this
    # record, unlike the follow-through notes above, which match
    # by insured across records), with the acknowledgement
    # attached — who called the broker, when, and what they
    # recorded — and the cases are listed again below.
    ack_window = (
        follow_escalation_days
        if follow_escalation_days is not None
        else follow_through_escalation_days()
    )
    record_by_id = {r.result.submission_id: r for r in records}
    escalated_ack_by_id: dict[str, dict] = {}
    for entry in _exhausted_breach_notices(
        records,
        now,
        env_int("DISPATCH_ATTEMPT_BUDGET", 8),
        acknowledged=True,
    ):
        ack = entry.get("acknowledgement") or {}
        acked_at = _parse_timestamp(ack.get("acknowledged_at") or "")
        if acked_at is None:
            continue
        ack_age = max(0, (now.date() - acked_at.date()).days)
        if ack_age >= 2 * ack_window:
            record = record_by_id.get(entry["submission_id"])
            escalated_ack_by_id[entry["submission_id"]] = {
                **entry,
                "acknowledged_age_days": ack_age,
                "escalation_days": 2 * ack_window,
                # The escalation webhook's state for this case —
                # null until the escalation check has fired.
                "escalation_notice": (
                    record.escalation_notice if record is not None else None
                ),
            }
    ack_flagged = 0
    for item in all_items:
        escalated = escalated_ack_by_id.get(item["submission_id"])
        if escalated is None:
            continue
        ack = escalated.get("acknowledgement") or {}
        item["acknowledged_escalation"] = {
            "acknowledged_by": ack.get("acknowledged_by", ""),
            "acknowledged_at": ack.get("acknowledged_at", ""),
            "note": ack.get("note", ""),
            "acknowledged_age_days": escalated["acknowledged_age_days"],
        }
        ack_flagged += 1
    escalated_acknowledgements = sorted(
        escalated_ack_by_id.values(),
        key=lambda e: (-e["acknowledged_age_days"], e["submission_id"]),
    )
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "total_awaiting": total,
        "buckets": [
            {"bucket": label, "count": len(buckets[label]), "submissions": buckets[label]}
            for label, _, _ in QUEUE_AGE_BUCKETS
        ],
        "unknown_age": unknown,
        "by_outcome": dict(sorted(by_outcome.items())),
        "oldest": oldest,
        "sla_budgets": budgets,
        "breach_count": len(breaches),
        "breaches": breaches,
        "breach_movement": breach_movement,
        "blocked_bundles": blocked_bundles,
        "escalated_notes": escalated_notes,
        "items_with_escalated_notes": flagged,
        "escalated_acknowledgements": escalated_acknowledgements,
        "items_with_escalated_acknowledgements": ack_flagged,
    }


def compose_status_note(
    record: ApprovalRecord, item: dict, now: datetime | None = None
) -> str:
    """The drafted underwriter status note for one SLA-breached case.

    Composed in code from the record and the queue item (the same
    projection the /queue view renders — the note and the queue
    cannot disagree), in the broker letter's status-only
    discipline: where the case stands, how far past its SLA budget
    it is, and what is outstanding. No timelines, no outcomes, no
    figures the record does not carry. The note is an internal
    draft for the underwriter — it is never sent to anyone, and the
    service guardrail-checks it before it is stored (see
    ``guardrails.validate_status_note``).
    """
    result = record.result
    facts = result.facts
    now = now or datetime.now(timezone.utc)
    lines = [
        f"INTERNAL DRAFT — underwriter status note for submission "
        f"{result.submission_id} (agent-drafted {now.date().isoformat()}; not sent)",
        "",
        f"Insured: {item['insured_name'] or result.insured_name} · "
        f"broker: {item['broker_name'] or 'not stated'} · "
        f"line: {result.line_of_business}",
        f"Status: awaiting underwriter decision. Submitted "
        f"{item['submitted_at'] or 'date not recorded'}; "
        f"{item['age_days']} day(s) in the queue.",
        f"SLA: risk tier {item['risk_tier']} carries a "
        f"{item['sla_budget_days']}-day budget — this case is "
        f"{item['sla_days_over']} day(s) past it.",
        f"Triage: {result.triage.outcome} / {result.triage.risk_tier}, "
        f"readiness {facts.readiness} (completeness {facts.completeness_score}).",
    ]
    if item["blocking_lines"]:
        for entry in item["blocking_lines"]:
            missing = ", ".join(entry["missing_items"]) or "items on file"
            lines.append(
                f"Blocking line: {entry['line_of_business']} — missing "
                f"{missing}; the bundle has waited {entry['age_days']} "
                "day(s) on this line."
            )
    elif facts.missing_elements:
        lines.append(
            "Outstanding: missing " + ", ".join(facts.missing_elements) + "."
        )
    elif facts.gap_sources:
        lines.append("Outstanding: readiness gaps — " + "; ".join(facts.gap_sources) + ".")
    else:
        lines.append(
            "Outstanding: nothing missing — the file is complete and "
            "awaiting a decision."
        )
    if result.triage.referral_triggers:
        lines.append(
            f"Referral triggers on file: {len(result.triage.referral_triggers)}."
        )
    if result.review is not None and result.review.verdict != "pass":
        lines.append(f"Independent review verdict: {result.review.verdict}.")
    if result.needs_information:
        lines.append(
            "An information request has been composed for the broker "
            "(composed only — not sent)."
        )
    lines.append(
        "Next step: underwriter review and decision. This note is an "
        "internal draft; the underwriter decides what happens with "
        "this case."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Status-note follow-through — what the note said was outstanding,
# checked against what the next submission actually brought
# ---------------------------------------------------------------------------


def status_note_follow_through(
    note: dict,
    *,
    missing_elements: list[str],
    blocking_lines: list[dict],
    line_of_business: str | None = None,
) -> dict:
    """Check a stored status note's outstanding items against a new
    submission's computed facts.

    The note snapshots what was outstanding when it was drafted
    (``note["outstanding"]``: whole-submission missing elements and
    per-line blocking items). A resubmission from the same insured
    answers some of it, or none of it — this computes which, in
    code, from the new run's facts: an outstanding item is *still
    outstanding* when the same element is missing in the new facts
    (at note level, or on the blocking line it belonged to), and
    *answered* when it is missing nowhere relevant any more.

    ``line_of_business`` scopes the check to one line's run (a
    bundle's per-line sub-analysis): only the note-level items and
    that line's blocking items are judged, against that line's
    missing elements. Without it, the check reads the new result
    whole (the service's record-marking path).

    Notes drafted before the outstanding snapshot existed carry
    nothing to check: the result says so (``recorded`` False,
    state ``not_recorded``) instead of pretending the note was
    answered.
    """
    note = note or {}
    base = {
        "prior_submission_id": note.get("submission_id", ""),
        "note_drafted_at": note.get("drafted_at", ""),
    }
    outstanding = note.get("outstanding")
    if outstanding is None:
        return {
            **base,
            "recorded": False,
            "state": "not_recorded",
            "answered": [],
            "still_outstanding": [],
        }
    entries = _outstanding_entries(note)
    if line_of_business:
        target = normalize_line(line_of_business)
        entries = [
            e
            for e in entries
            if not e["line"] or normalize_line(e["line"]) == target
        ]
    current_missing = set(missing_elements or [])
    current_by_line = {
        normalize_line(b.get("line_of_business", "")): set(
            b.get("missing_items") or []
        )
        for b in blocking_lines or []
    }

    def _still(entry: dict) -> bool:
        if entry["item"] in current_missing:
            return True
        if entry["line"]:
            return entry["item"] in current_by_line.get(
                normalize_line(entry["line"]), set()
            )
        if line_of_business:
            return entry["item"] in current_by_line.get(
                normalize_line(line_of_business), set()
            )
        return False

    answered = [e for e in entries if not _still(e)]
    still = [e for e in entries if _still(e)]
    if not entries:
        state = "nothing_outstanding"
    elif not still:
        state = "answered"
    elif not answered:
        state = "still_outstanding"
    else:
        state = "partially_answered"
    return {
        **base,
        "recorded": True,
        "state": state,
        "answered": answered,
        "still_outstanding": still,
    }


def _outstanding_entries(note: dict) -> list[dict]:
    """A note's outstanding snapshot as flat ``{line, item}``
    entries — the note-level missing elements plus each blocking
    line's missing items. The follow-through check and the inbox
    digest read the same flattening, so they cannot disagree
    about what a note was waiting on."""
    outstanding = (note or {}).get("outstanding") or {}
    entries: list[dict] = [
        {"line": "", "item": item}
        for item in outstanding.get("missing_elements") or []
    ]
    for entry in outstanding.get("blocking_lines") or []:
        entries += [
            {"line": entry.get("line_of_business", ""), "item": item}
            for item in entry.get("missing_items") or []
        ]
    return entries


def _follow_items_text(entries: list[dict]) -> str:
    return "; ".join(
        f"{e['item']} ({e['line']})" if e["line"] else e["item"] for e in entries
    )


def follow_through_packet_lines(note: dict, follow: dict) -> list[str]:
    """The follow-through as a section of the new submission's packet.

    The note was drafted for the underwriter on an earlier case; the
    packet of the case that answers it is where the loop closes —
    what was outstanding then, what arrived with this submission,
    what is still missing — so the earlier note never reads as the
    current state of the case. Status only, like the note itself.
    """
    header = (
        f"Follow-through on the underwriter status note for "
        f"{follow['prior_submission_id'] or 'an earlier submission'}"
        + (
            f" (drafted {follow['note_drafted_at'][:10]}):"
            if follow["note_drafted_at"]
            else ":"
        )
    )
    if not follow["recorded"]:
        return [
            header,
            "- that note predates outstanding-item tracking, so there "
            "is nothing recorded to check this submission against; "
            "read this case on its own facts.",
        ]
    if follow["state"] == "nothing_outstanding":
        return [
            header,
            "- the note recorded nothing outstanding: the earlier case "
            "was complete and waiting on a decision, and this "
            "submission is read on its own facts.",
        ]
    outstanding = follow["answered"] + follow["still_outstanding"]
    lines = [
        header,
        f"- outstanding when the note was drafted: {_follow_items_text(outstanding)}",
        "- arrived with this submission: "
        + (_follow_items_text(follow["answered"]) if follow["answered"] else "none of it"),
        "- still outstanding: "
        + (
            _follow_items_text(follow["still_outstanding"])
            if follow["still_outstanding"]
            else "none"
        ),
    ]
    lines.append(
        {
            "answered": "- the note is answered: everything it waited on "
            "has arrived with this submission.",
            "partially_answered": "- the note is partially answered: it "
            "stays open on the items still outstanding above.",
            "still_outstanding": "- none of the note's outstanding items "
            "arrived; the note still describes what this case is waiting on.",
        }[follow["state"]]
    )
    return lines


# ---------------------------------------------------------------------------
# Follow-through digest — every status note, and where its loop stands
# ---------------------------------------------------------------------------

#: Digest ordering: unresolved notes first (a loop nobody has
#: closed), then partial answers, then answered ones.
_FOLLOW_STATE_RANK = {
    "still_outstanding": 0,
    "awaiting_resubmission": 0,
    "not_recorded": 0,
    "partially_answered": 1,
    "answered": 2,
}


#: Follow-through states in which a note's loop is still open —
#: the states an aged note escalates from. (``answered`` is a
#: closed loop; ``not_recorded`` is a bookkeeping gap, not a loop
#: anyone is waiting on.)
_FOLLOW_OPEN_STATES = ("still_outstanding", "partially_answered", "awaiting_resubmission")

#: Days a status note's loop may stay open before the digest
#: escalates it (FOLLOW_THROUGH_ESCALATION_DAYS overrides).
DEFAULT_FOLLOW_THROUGH_ESCALATION_DAYS = 14


def follow_through_escalation_days() -> int:
    return max(
        0, env_int("FOLLOW_THROUGH_ESCALATION_DAYS", DEFAULT_FOLLOW_THROUGH_ESCALATION_DAYS)
    )


def follow_through_digest(
    records: list[ApprovalRecord],
    now: datetime | None = None,
    *,
    escalation_days: int | None = None,
) -> dict:
    """Every drafted status note, digested by follow-through state.

    The follow-through marking lives on the old note's record
    (written when a resubmission from the same insured was
    checked against the note's snapshot); until now it was only
    visible by opening that record. The digest is the inbox-level
    scan: which notes a resubmission **answered**, which it
    answered **partially**, and which are **still outstanding** —
    including the notes no resubmission has reached yet
    (``awaiting_resubmission``: everything the note snapshot is
    still outstanding, by definition). Oldest outstanding first,
    so the loop that has been open longest is the one the
    underwriter sees first. Pure projection — nothing is written,
    and a note's text is never touched.

    An open loop also **ages**: an entry whose note has been
    waiting at least ``FOLLOW_THROUGH_ESCALATION_DAYS`` (default
    14) is flagged ``escalated`` and listed again in the digest's
    ``escalated`` subset. Escalation is the note's own flag, not
    an SLA breach — the underlying case may be long decided
    while its note rots — and it is what the ops digest (see
    :func:`ops_digest`) works from, so an unanswered note is
    worked, not just visible.

    An escalated note can also be **worked**: an underwriter who
    acted on it records the touch on the note itself (see
    ``service.mark_escalated_note_worked`` — the note's
    ``escalation_touches``). The latest touch **quiets** the
    escalation for one escalation window from the touch: the
    entry stops counting as escalated — in this digest, in the
    ops digest's counts, and on the queue, which reads this same
    computation — and rides in the digest's ``worked`` subset
    with the touch attached, so the worked ones stay auditable
    from the same read. Quieting is a window, not a pardon: if
    the loop is still open when a full window has passed since
    the touch, the note escalates again — a worked state must
    not become a quieter place to rot — and a further touch can
    be recorded then. The note's text and its follow-through
    marking are never touched by any of this.

    The touches also price the response: each touched note
    carries a **response latency** — the days from the moment
    the note first escalated (drafted + one window) to its
    *first* touch — and the digest aggregates them as
    ``response_latency`` (notes measured, median / mean / max
    days). The worked counts say how many notes were touched;
    this says how fast. A note answered only after it
    re-escalated carries the whole wait: the first touch is the
    response the escalation got, whenever it finally came.
    """
    now = now or datetime.now(timezone.utc)
    if escalation_days is None:
        escalation_days = follow_through_escalation_days()
    entries: list[dict] = []
    for record in records:
        note = record.status_note
        if not note:
            continue
        result = record.result
        party = history_entry(record)
        follow = note.get("follow_through") or {}
        drafted_raw = note.get("drafted_at", "")
        drafted = _parse_timestamp(drafted_raw)
        if follow:
            raw_state = follow.get("state", "")
            if not follow.get("recorded", True) or raw_state == "not_recorded":
                state = "not_recorded"
            elif raw_state in ("answered", "nothing_outstanding"):
                state = "answered"
            elif raw_state == "partially_answered":
                state = "partially_answered"
            else:
                state = "still_outstanding"
            answered = list(follow.get("answered") or [])
            still = list(follow.get("still_outstanding") or [])
            by_submission_id = follow.get("by_submission_id", "")
            checked_at = follow.get("checked_at", "")
        else:
            state = "awaiting_resubmission"
            answered = []
            still = _outstanding_entries(note)
            by_submission_id = ""
            checked_at = ""
        note_age_days = (
            max(0, (now.date() - drafted.date()).days)
            if drafted is not None
            else None
        )
        touches = note.get("escalation_touches") or []
        latest_touch = touches[-1] if touches else None
        first_touch = touches[0] if touches else None
        first_touch_at = _parse_timestamp(
            (first_touch or {}).get("worked_at") or ""
        )
        # Response latency (see the docstring): the first
        # touch against the first escalation moment. Touches
        # append chronologically, so the first is the response;
        # later touches re-quiet, they do not re-answer.
        response_days = None
        if first_touch_at is not None and drafted is not None:
            escalated_on = drafted.date() + timedelta(
                days=escalation_days
            )
            response_days = max(
                0, (first_touch_at.date() - escalated_on).days
            )
        worked_at = _parse_timestamp(
            (latest_touch or {}).get("worked_at") or ""
        )
        worked_age_days = (
            max(0, (now.date() - worked_at.date()).days)
            if worked_at is not None
            else None
        )
        # The latest touch quiets the escalation for one window
        # from the touch; past that window a still-open loop
        # escalates again, touch or no touch (see the docstring).
        quieted_by_work = bool(
            state in _FOLLOW_OPEN_STATES
            and note_age_days is not None
            and note_age_days >= escalation_days
            and worked_age_days is not None
            and worked_age_days < escalation_days
        )
        entries.append(
            {
                "submission_id": result.submission_id,
                "insured_name": party["insured_name"],
                "broker_name": party["broker_name"],
                "drafted_at": drafted_raw,
                "note_age_days": note_age_days,
                "escalated": bool(
                    state in _FOLLOW_OPEN_STATES
                    and note_age_days is not None
                    and note_age_days >= escalation_days
                    and not quieted_by_work
                ),
                "worked": latest_touch,
                "worked_touches": len(touches),
                "worked_age_days": worked_age_days,
                "first_touch_at": (first_touch or {}).get("worked_at")
                or "",
                "response_days": response_days,
                "quieted_by_work": quieted_by_work,
                "state": state,
                "outstanding_at_drafting": len(_outstanding_entries(note)),
                "answered_items": answered,
                "still_outstanding_items": still,
                "by_submission_id": by_submission_id,
                "checked_at": checked_at,
            }
        )
    entries.sort(
        key=lambda e: (
            _FOLLOW_STATE_RANK.get(e["state"], 0),
            e["drafted_at"],
            e["submission_id"],
        )
    )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    escalated = [e for e in entries if e["escalated"]]
    worked = [
        e for e in entries if e["worked"] is not None and not e["escalated"]
    ]
    quieted = [e for e in entries if e["quieted_by_work"]]
    response_values = sorted(
        e["response_days"]
        for e in entries
        if e["response_days"] is not None
    )
    if response_values:
        mid = len(response_values) // 2
        median_days = (
            float(response_values[mid])
            if len(response_values) % 2
            else (response_values[mid - 1] + response_values[mid]) / 2
        )
        response_latency = {
            "notes_measured": len(response_values),
            "median_days": round(median_days, 1),
            "mean_days": round(
                sum(response_values) / len(response_values), 1
            ),
            "max_days": response_values[-1],
        }
    else:
        response_latency = {
            "notes_measured": 0,
            "median_days": None,
            "mean_days": None,
            "max_days": None,
        }
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "total_notes": len(entries),
        "counts": counts,
        "entries": entries,
        "escalation_days": escalation_days,
        "escalated_count": len(escalated),
        "escalated": escalated,
        "worked_count": len(worked),
        "worked": worked,
        "quieted_count": len(quieted),
        "quieted": quieted,
        "response_latency": response_latency,
    }


# ---------------------------------------------------------------------------
# Insured loss-development timeline
# ---------------------------------------------------------------------------


def _claim_key(date: str, description: str) -> str:
    """A claim's timeline identity, keyed the way the diffs report
    claims: date + normalised description. (Loss runs that carry
    carrier claim ids keep identity across wording edits inside a
    single diff; across a whole history the reported date +
    description is the stable handle the diffs themselves expose.)
    """
    return f"{(date or '').strip()}|{_norm_desc(description)}"


def insured_loss_timeline(
    records: list[ApprovalRecord], insured_name: str
) -> dict | None:
    """One insured's loss development across all their submissions.

    Every renewal diff already answers "what changed since last
    time" for its own pair; at the third renewal the underwriter's
    question is the whole arc. This chains the stored per-line
    diffs in store order into one history: an entry per submission
    (per line, for bundles) with its computed loss summary and its
    diff against the previous submission of that line, and — the
    point of the view — one event stream per claim: when it
    appeared on a run, how its incurred and status developed, and
    whether it is still listed. Returns None for an unknown
    insured. Read-only: the stored diffs are reported, never
    recomputed.
    """
    hits = [
        r for r in records if insured_name and _party_name(r, "insured") == insured_name
    ]
    if not hits:
        return None
    entries: list[dict] = []
    claims: dict[str, dict] = {}
    claim_order: list[str] = []
    totals = {
        "renewals": 0,
        "appeared": 0,
        "developed": 0,
        "no_longer_listed": 0,
        "cumulative_net_incurred_change": 0.0,
    }

    def _claim(key: str, date: str, description: str, submission_id: str) -> dict:
        claim = claims.get(key)
        if claim is None:
            claim = {
                "key": key,
                "date": date,
                "description": description,
                "first_seen_submission_id": submission_id,
                "state": "on_latest_run",
                "latest_incurred": 0.0,
                "latest_status": "",
                "events": [],
            }
            claims[key] = claim
            claim_order.append(key)
        return claim

    for record in hits:
        result = record.result
        submitted_at = (record.submission or {}).get("submitted_at", "")
        if result.multi_line:
            units = [
                (a.line_of_business, a.submission_id, a.result)
                for a in result.line_analyses
            ]
        else:
            units = [(result.line_of_business, result.submission_id, result)]
        for line, unit_id, unit in units:
            diff = unit.submission_diff
            entry = {
                "submission_id": unit_id,
                "record_submission_id": result.submission_id,
                "line_of_business": line,
                "submitted_at": submitted_at,
                "outcome": unit.triage.outcome,
                "risk_tier": unit.triage.risk_tier,
                "status": result.status,
                "loss_summary": unit.facts.loss_summary.model_dump(mode="json"),
                "prior_submission_id": diff.prior_submission_id if diff else None,
                "net_incurred_change": None,
                "events": {"appeared": 0, "developed": 0, "no_longer_listed": 0},
            }
            development = diff.loss_development if diff else None
            if diff is not None:
                totals["renewals"] += 1
            if development is not None:
                entry["net_incurred_change"] = development.net_incurred_change
                totals["cumulative_net_incurred_change"] = round(
                    totals["cumulative_net_incurred_change"]
                    + development.net_incurred_change,
                    2,
                )
                for ref in development.appeared:
                    claim = _claim(
                        _claim_key(ref.date, ref.description),
                        ref.date,
                        ref.description,
                        unit_id,
                    )
                    claim["events"].append(
                        {
                            "type": "appeared",
                            "submission_id": unit_id,
                            "incurred": ref.amount_incurred,
                            "status": ref.status,
                        }
                    )
                    claim["state"] = "on_latest_run"
                    claim["latest_incurred"] = ref.amount_incurred
                    claim["latest_status"] = ref.status
                    entry["events"]["appeared"] += 1
                    totals["appeared"] += 1
                for change in development.developed:
                    claim = _claim(
                        _claim_key(change.date, change.description),
                        change.date,
                        change.description,
                        unit_id,
                    )
                    claim["events"].append(
                        {
                            "type": "developed",
                            "submission_id": unit_id,
                            "prior_incurred": change.prior_incurred,
                            "incurred": change.current_incurred,
                            "delta_incurred": change.delta_incurred,
                            "prior_status": change.prior_status,
                            "status": change.current_status,
                        }
                    )
                    claim["state"] = "on_latest_run"
                    claim["latest_incurred"] = change.current_incurred
                    claim["latest_status"] = change.current_status
                    entry["events"]["developed"] += 1
                    totals["developed"] += 1
                for ref in development.disappeared:
                    claim = _claim(
                        _claim_key(ref.date, ref.description),
                        ref.date,
                        ref.description,
                        unit_id,
                    )
                    claim["events"].append(
                        {
                            "type": "no_longer_listed",
                            "submission_id": unit_id,
                            "incurred": ref.amount_incurred,
                            "status": ref.status,
                        }
                    )
                    claim["state"] = "no_longer_listed"
                    claim["latest_incurred"] = ref.amount_incurred
                    claim["latest_status"] = ref.status
                    entry["events"]["no_longer_listed"] += 1
                    totals["no_longer_listed"] += 1
            entries.append(entry)
    totals["claims_tracked"] = len(claim_order)
    return {
        "insured": insured_name,
        "submissions": len(hits),
        "entries": entries,
        "claims": [claims[key] for key in claim_order],
        "totals": totals,
    }


def loss_timeline_packet_lines(
    timeline: dict | None, current_diff=None
) -> list[str]:
    """The loss-development arc as a section of a renewal's packet.

    The timeline view (console drill-down) chains an insured's
    stored diffs; this composes the same arc into the decision
    document itself, so the underwriter reads the development
    history where they decide instead of behind a drill-down they
    may never open. The stored timeline covers the submissions
    already on file; the current submission's own diff is replayed
    over it (appeared / developed / no-longer-listed applied to the
    claim states, its net movement added to the cumulative), so the
    arc the packet tells is the arc *including this submission*.

    Compact by design — counts, cumulative movement, and the claims
    still open: the per-claim detail already lives in the changes
    section and the timeline view. Empty when there is no arc to
    tell (no stored history, or no claim ever moved).
    """
    if not timeline:
        return []
    claims: dict[str, dict] = {}
    order: list[str] = []
    for claim in timeline.get("claims") or []:
        key = claim["key"]
        claims[key] = {
            "date": claim["date"],
            "description": claim["description"],
            "state": claim["state"],
            "latest_incurred": claim["latest_incurred"],
            "latest_status": claim["latest_status"],
        }
        order.append(key)
    totals = timeline.get("totals") or {}
    renewals = totals.get("renewals", 0)
    cumulative = float(totals.get("cumulative_net_incurred_change", 0.0))
    development = current_diff.loss_development if current_diff else None
    if current_diff is not None:
        renewals += 1
    if development is not None:
        cumulative = round(cumulative + development.net_incurred_change, 2)

        def _apply(date: str, description: str, **updates) -> None:
            key = _claim_key(date, description)
            claim = claims.get(key)
            if claim is None:
                claim = {
                    "date": date,
                    "description": description,
                    "state": "on_latest_run",
                    "latest_incurred": 0.0,
                    "latest_status": "",
                }
                claims[key] = claim
                order.append(key)
            claim.update(updates)

        for ref in development.appeared:
            _apply(
                ref.date,
                ref.description,
                state="on_latest_run",
                latest_incurred=ref.amount_incurred,
                latest_status=ref.status,
            )
        for change in development.developed:
            _apply(
                change.date,
                change.description,
                state="on_latest_run",
                latest_incurred=change.current_incurred,
                latest_status=change.current_status,
            )
        for ref in development.disappeared:
            _apply(
                ref.date,
                ref.description,
                state="no_longer_listed",
                latest_incurred=ref.amount_incurred,
                latest_status=ref.status,
            )
    if renewals < 1 or not order:
        return []
    chained = [claims[key] for key in order]
    open_claims = [
        c
        for c in chained
        if c["state"] == "on_latest_run"
        and (c["latest_status"] or "").strip().lower() != "closed"
    ]
    gone = [c for c in chained if c["state"] == "no_longer_listed"]
    lines = [
        "Loss history arc (this insured's stored submissions, chained):",
        f"- {timeline.get('submissions', 0)} stored submission(s), "
        f"{renewals} renewal(s) chained including this one; "
        f"{len(order)} claim(s) tracked across the loss runs",
        f"- cumulative net incurred movement across the chained "
        f"renewals: {cumulative:+,.0f}",
    ]
    if open_claims:
        shown = "; ".join(
            f"{c['description'] or 'no description'} "
            f"({c['date'] or 'date not stated'}) — "
            f"${c['latest_incurred']:,.0f} incurred, "
            f"{c['latest_status'] or 'status not stated'}"
            for c in open_claims[:4]
        )
        extra = (
            f"; and {len(open_claims) - 4} more" if len(open_claims) > 4 else ""
        )
        lines.append(f"- still open on the latest run: {shown}{extra}")
    else:
        lines.append("- no claims still open on the latest run")
    if gone:
        lines.append(
            f"- no longer listed on the latest run: {len(gone)} claim(s)"
        )
    return lines


def audit_csv(rows: list[dict]) -> str:
    """The audit rows as CSV text (header + one line per row)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in AUDIT_COLUMNS})
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Dispatch operations — the webhook ledgers, read as one surface
# ---------------------------------------------------------------------------


def _next_backoff_seconds(attempts_made: int) -> float:
    """The backoff the retry schedule assigns the *next* attempt.

    Mirrors the schedule ``service._dispatch_once`` records: the
    first retry waits 30s, doubling to a 600s cap.
    """
    if attempts_made < 1:
        return 0
    return min(30 * 2 ** (attempts_made - 1), 600)


def _parse_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _age_bucket(age_days: int) -> str:
    for label, low, high in QUEUE_AGE_BUCKETS:
        if age_days >= low and (high is None or age_days <= high):
            return label
    return QUEUE_AGE_BUCKETS[-1][0]


def _exhausted_breach_notices(
    records: list[ApprovalRecord],
    now: datetime,
    attempt_budget: int,
    *,
    acknowledged: bool = False,
) -> list[dict]:
    """Breach notices that exhausted their own delivery ledger.

    The SLA-breach notice has its own ledger (see
    ``ApprovalRecord.sla_breach_attempts``), and it is read
    nowhere in the approval failures above — it is a different
    event with a different cost: an approval failure means a
    decided packet did not arrive; an exhausted breach notice
    means a case sat past its SLA budget, still undecided, and
    **nobody was ever told**. A notice is exhausted when its
    ledger has spent the attempt budget
    (``DISPATCH_ATTEMPT_BUDGET``, the same budget every delivery
    in the system is measured against) and its latest attempt
    still failed. A notice whose case has since been decided is
    left out — the call it needed already happened in the room;
    a notice later delivered is out by its last attempt's word.

    ``acknowledged`` selects which side of the worked line to
    read: the default is the worklist — exhausted notices no
    operator has acknowledged yet. With ``acknowledged=True``
    the same computation returns the worked ones, each carrying
    its acknowledgement (who picked the case up, when, their
    note). An acknowledgement marks a notice worked; it does not
    un-exhaust it, and the two reads together are the whole
    exhaustion picture.

    Each entry ages from the notice's first observation (the
    moment the breach was real, not the moment delivery gave
    up), bucketed on the queue's day buckets, oldest first.
    """
    entries: list[dict] = []
    for record in records:
        notice = record.sla_breach_notice
        attempts = record.sla_breach_attempts or []
        if not notice or not attempts:
            continue
        if record.result.status != "awaiting_underwriter":
            continue
        last = attempts[-1]
        if last.get("status") != "failed" or len(attempts) < attempt_budget:
            continue
        if (record.breach_notice_acknowledgement is not None) != acknowledged:
            continue
        entry = history_entry(record)
        first_at = _parse_timestamp(attempts[0].get("attempted_at") or "")
        observed = _parse_timestamp(notice.get("first_observed_at") or "")
        anchor = observed or first_at
        age_days = (
            max(0, (now.date() - anchor.date()).days)
            if anchor is not None
            else None
        )
        sla = notice.get("sla") or {}
        entries.append(
            {
                "submission_id": record.result.submission_id,
                "insured_name": entry["insured_name"],
                "broker_name": entry["broker_name"],
                "risk_tier": notice.get("risk_tier")
                or record.result.triage.risk_tier,
                "sla_budget_days": sla.get("budget_days"),
                "first_observed_at": notice.get("first_observed_at") or "",
                "attempts": len(attempts),
                "attempts_remaining": 0,
                "first_attempt_at": attempts[0].get("attempted_at") or "",
                "last_attempt_at": last.get("attempted_at") or "",
                "last_http_status": last.get("http_status"),
                "last_error": last.get("error"),
                "notice_age_days": age_days,
                "notice_age_bucket": (
                    _age_bucket(age_days) if age_days is not None else None
                ),
                "acknowledgement": record.breach_notice_acknowledgement,
                "_sort": anchor or datetime.max.replace(tzinfo=timezone.utc),
            }
        )
    entries.sort(key=lambda e: (e["_sort"], e["submission_id"]))
    for entry in entries:
        entry.pop("_sort", None)
    return entries


def _exhausted_escalation_notices(
    records: list[ApprovalRecord],
    now: datetime,
    attempt_budget: int,
) -> list[dict]:
    """Acknowledged-escalation notices that exhausted their own
    delivery ledger.

    The second rung has its own ledger (see
    ``ApprovalRecord.escalation_attempts``), read exactly the
    way the breach notices' ledger is read (see
    :func:`_exhausted_breach_notices`): the ledger has spent the
    shared attempt budget, the latest attempt still failed, and
    the case is still awaiting a decision. The event is a
    different one, and the entries keep the difference visible:
    a breach exhaustion means **nobody was told**; an escalation
    exhaustion means the case was told about, acknowledged, left
    waiting past the second window — and the escalation that
    said so is the thing that never landed. The acknowledgement
    the escalation carried rides on the entry, so the worklist
    shows the whole chain: told, called, still waiting, and the
    escalation undelivered.

    Each entry ages from the escalation's first observation,
    bucketed on the queue's day buckets, oldest first — the
    breach entries' shape, so the two lists read alike.
    """
    entries: list[dict] = []
    for record in records:
        notice = record.escalation_notice
        attempts = record.escalation_attempts or []
        if not notice or not attempts:
            continue
        if record.result.status != "awaiting_underwriter":
            continue
        last = attempts[-1]
        if last.get("status") != "failed" or len(attempts) < attempt_budget:
            continue
        entry = history_entry(record)
        first_at = _parse_timestamp(attempts[0].get("attempted_at") or "")
        observed = _parse_timestamp(notice.get("first_observed_at") or "")
        anchor = observed or first_at
        age_days = (
            max(0, (now.date() - anchor.date()).days)
            if anchor is not None
            else None
        )
        entries.append(
            {
                "submission_id": record.result.submission_id,
                "insured_name": entry["insured_name"],
                "broker_name": entry["broker_name"],
                "notice_kind": "acknowledged_escalation",
                "risk_tier": notice.get("risk_tier")
                or record.result.triage.risk_tier,
                "acknowledgement": notice.get("acknowledgement"),
                "acknowledged_age_days": notice.get(
                    "acknowledged_age_days"
                ),
                "first_observed_at": notice.get("first_observed_at") or "",
                "attempts": len(attempts),
                "attempts_remaining": 0,
                "first_attempt_at": attempts[0].get("attempted_at") or "",
                "last_attempt_at": last.get("attempted_at") or "",
                "last_http_status": last.get("http_status"),
                "last_error": last.get("error"),
                "notice_age_days": age_days,
                "notice_age_bucket": (
                    _age_bucket(age_days) if age_days is not None else None
                ),
                "_sort": anchor or datetime.max.replace(tzinfo=timezone.utc),
            }
        )
    entries.sort(key=lambda e: (e["_sort"], e["submission_id"]))
    for entry in entries:
        entry.pop("_sort", None)
    return entries


def dispatch_overview(
    records: list[ApprovalRecord],
    now: datetime | None = None,
    *,
    stuck_age_hours: float | None = None,
    attempt_budget: int | None = None,
) -> dict:
    """The dispatch operations view over every record's ledger.

    Each approval-webhook ledger answers for its own submission;
    this is the surface over all of them: how many deliveries were
    attempted, which records' latest attempt failed (oldest failure
    first — that is the order an operator works them), and which of
    those failures are *due* a retry under the recorded backoff
    schedule (last attempt + the next attempt's backoff <= now).
    Retries stay operator-triggered; "due" is the schedule's word,
    computed here, not a timer the service runs.

    Failures also **age**: a failure's age runs from its *first*
    failed attempt, bucketed on the same day buckets as the queue
    aging view, and a failure is flagged **stuck** when it is older
    than ``DISPATCH_STUCK_AGE_HOURS`` (default 24) and still has
    attempts left under ``DISPATCH_ATTEMPT_BUDGET`` (default 8) —
    old enough to be an incident, young enough in attempts that
    another retry is still the right move. A failure past its
    budget is exhausted, not stuck: it needs a decision (fix the
    receiver, reroute, write it off), not another attempt, and the
    view says which is which.

    The breach notices ride the same view, in their own list:
    :func:`_exhausted_breach_notices` — notices whose own ledger
    spent the budget still undelivered, on cases still awaiting a
    decision. They are not approval failures and never merge into
    that list; they are the cases out of automated nudges, and
    the next step for one is a human call, not a retry button.
    Notices an operator has acknowledged (see
    :func:`_exhausted_breach_notices`) ride in a second list,
    ``acknowledged_breach_notices``: the worklist counts only
    the unacknowledged — a case somebody has picked up is no
    longer silently rotting — but the worked ones stay visible
    with their acknowledgement, because "we called" is a claim
    the record should be able to show.

    The second rung's notices join this same worklist when
    their own ledger exhausts (see
    :func:`_exhausted_escalation_notices`): an exhausted
    ``acknowledged_escalation`` is the same operational shape —
    a signed event that spent its attempts on a case still
    awaiting a decision — so it is worked from the same surface
    in its own list, not from a new one.
    """
    now = now or datetime.now(timezone.utc)
    if stuck_age_hours is None:
        stuck_age_hours = env_float("DISPATCH_STUCK_AGE_HOURS", 24.0)
    if attempt_budget is None:
        attempt_budget = env_int("DISPATCH_ATTEMPT_BUDGET", 8)
    attempts_total = 0
    sent_records = 0
    failures: list[dict] = []
    oldest_failure_at: str | None = None
    oldest_key: datetime | None = None
    for record in records:
        attempts = record.dispatch_attempts or []
        attempts_total += len(attempts)
        if not attempts:
            continue
        if record.dispatch_status == "sent":
            sent_records += 1
            continue
        if record.dispatch_status != "failed":
            continue
        entry = history_entry(record)
        last = attempts[-1]
        last_at = _parse_timestamp(last.get("attempted_at") or "")
        first_at = _parse_timestamp(attempts[0].get("attempted_at") or "")
        backoff = _next_backoff_seconds(len(attempts))
        due_at = last_at + timedelta(seconds=backoff) if last_at else None
        if first_at is not None:
            age_seconds = max(0.0, (now - first_at).total_seconds())
            age_days = int(age_seconds // 86400)
            age_bucket: str | None = _age_bucket(age_days)
        else:
            age_seconds = None
            age_days = None
            age_bucket = None
        attempts_remaining = max(0, attempt_budget - len(attempts))
        stuck = bool(
            age_seconds is not None
            and age_seconds >= stuck_age_hours * 3600
            and attempts_remaining > 0
        )
        failures.append(
            {
                "submission_id": record.result.submission_id,
                "insured_name": entry["insured_name"],
                "broker_name": entry["broker_name"],
                "attempts": len(attempts),
                "first_attempt_at": attempts[0].get("attempted_at") or "",
                "last_attempt_at": last.get("attempted_at") or "",
                "last_http_status": last.get("http_status"),
                "last_error": last.get("error"),
                "next_attempt": len(attempts) + 1,
                "next_backoff_seconds": backoff,
                "retry_due": due_at is None or now >= due_at,
                "failure_age_days": age_days,
                "failure_age_bucket": age_bucket,
                "attempts_remaining": attempts_remaining,
                "stuck": stuck,
                "_sort": last_at or datetime.max.replace(tzinfo=timezone.utc),
            }
        )
        if last_at is not None and (oldest_key is None or last_at < oldest_key):
            oldest_key = last_at
            oldest_failure_at = last.get("attempted_at")
    failures.sort(key=lambda f: (f["_sort"], f["submission_id"]))
    for failure in failures:
        failure.pop("_sort", None)
    bucket_counts: dict[str, list[str]] = {
        label: [] for label, _, _ in QUEUE_AGE_BUCKETS
    }
    unknown_age: list[str] = []
    for failure in failures:
        if failure["failure_age_bucket"] is None:
            unknown_age.append(failure["submission_id"])
        else:
            bucket_counts[failure["failure_age_bucket"]].append(
                failure["submission_id"]
            )
    stuck_failures = [f for f in failures if f["stuck"]]
    exhausted_notices = _exhausted_breach_notices(
        records, now, attempt_budget
    )
    acknowledged_notices = _exhausted_breach_notices(
        records, now, attempt_budget, acknowledged=True
    )
    exhausted_escalations = _exhausted_escalation_notices(
        records, now, attempt_budget
    )
    notice_buckets: dict[str, list[str]] = {
        label: [] for label, _, _ in QUEUE_AGE_BUCKETS
    }
    unknown_notice_age: list[str] = []
    for notice_entry in exhausted_notices:
        if notice_entry["notice_age_bucket"] is None:
            unknown_notice_age.append(notice_entry["submission_id"])
        else:
            notice_buckets[notice_entry["notice_age_bucket"]].append(
                notice_entry["submission_id"]
            )
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "attempts_total": attempts_total,
        "records_with_attempts": sent_records + len(failures),
        "sent_records": sent_records,
        "failed_records": len(failures),
        "due_retries": [f for f in failures if f["retry_due"]],
        "failures": failures,
        "oldest_failure_at": oldest_failure_at,
        "failure_age_buckets": [
            {
                "bucket": label,
                "count": len(bucket_counts[label]),
                "submission_ids": bucket_counts[label],
            }
            for label, _, _ in QUEUE_AGE_BUCKETS
        ],
        "unknown_failure_age": unknown_age,
        "stuck_count": len(stuck_failures),
        "stuck_failures": stuck_failures,
        "stuck_age_hours": stuck_age_hours,
        "attempt_budget": attempt_budget,
        "exhausted_breach_notices": exhausted_notices,
        "exhausted_breach_count": len(exhausted_notices),
        "acknowledged_breach_notices": acknowledged_notices,
        "acknowledged_breach_count": len(acknowledged_notices),
        "exhausted_escalation_notices": exhausted_escalations,
        "exhausted_escalation_count": len(exhausted_escalations),
        "exhausted_breach_age_buckets": [
            {
                "bucket": label,
                "count": len(notice_buckets[label]),
                "submission_ids": notice_buckets[label],
            }
            for label, _, _ in QUEUE_AGE_BUCKETS
        ],
        "unknown_breach_notice_age": unknown_notice_age,
    }


# ---------------------------------------------------------------------------
# Ops digest — the rotting-quietly surfaces, in one worklist
# ---------------------------------------------------------------------------

#: Days before a live proposal's evidence expiry at which the
#: ops digest starts listing it as nearing expiry
#: (OPS_DIGEST_PROPOSAL_EXPIRY_WARNING_DAYS overrides).
DEFAULT_OPS_PROPOSAL_EXPIRY_WARNING_DAYS = 3


def ops_proposal_expiry_warning_days() -> int:
    return max(
        0,
        env_int(
            "OPS_DIGEST_PROPOSAL_EXPIRY_WARNING_DAYS",
            DEFAULT_OPS_PROPOSAL_EXPIRY_WARNING_DAYS,
        ),
    )


def ops_unannounced_gap_days() -> int:
    """Days past the second rung an escalated case may sit
    unannounced before the digest counts the gap itself as
    stale (``OPS_DIGEST_UNANNOUNCED_GAP_DAYS`` overrides).

    The default is one further follow-through window: an
    escalation check that has not run since this morning is a
    late sweep; one that has missed a whole window since the
    case crossed the rung is a sweep that has stopped, and the
    digest says so in those terms.
    """
    return max(
        0,
        env_int(
            "OPS_DIGEST_UNANNOUNCED_GAP_DAYS",
            follow_through_escalation_days(),
        ),
    )


def _acknowledged_still_waiting(
    acknowledged_entries: list[dict], now: datetime, wait_days: int
) -> list[dict]:
    """Acknowledged breach notices whose case still has not moved.

    An acknowledgement closes the "nobody was told" gap, not the
    case: the operator made the call, and the file went back to
    waiting for a decision. A notice acknowledged at least
    ``wait_days`` ago whose case is *still* undecided resurfaces
    here — the worked state must not become a quieter place to
    rot. The entries are the acknowledged read's own (the case
    is awaiting a decision by that computation's construction),
    annotated with the acknowledgement's age; the window is the
    follow-through escalation window, a sibling knob rather than
    a new family: a fortnight of silence after the call is the
    same operational fact as a fortnight of silence after a note.
    Oldest acknowledgement first.

    A still-waiting entry also carries its **second rung**:
    ``escalated`` once the acknowledgement is at least
    ``2 * wait_days`` old. The first window resurfaces the case
    here, where operators read; the second puts it on the
    queue's escalated surfaces (see :func:`queue_aging`), where
    the underwriter who finally picks it up works — with the
    acknowledgement attached, so the second touch never pretends
    to be the first.
    """
    waiting: list[dict] = []
    for entry in acknowledged_entries:
        ack = entry.get("acknowledgement") or {}
        acked_at = _parse_timestamp(ack.get("acknowledged_at") or "")
        if acked_at is None:
            continue
        ack_age = max(0, (now.date() - acked_at.date()).days)
        if ack_age < wait_days:
            continue
        waiting.append(
            {
                **entry,
                "acknowledged_age_days": ack_age,
                "escalated": ack_age >= 2 * wait_days,
            }
        )
    waiting.sort(
        key=lambda e: (-e["acknowledged_age_days"], e["submission_id"])
    )
    return waiting


def _escalation_notice_state(
    record: ApprovalRecord, exhausted_ids: set[str]
) -> dict:
    """One still-waiting case's escalation-telling state.

    The second rung has a telling of its own (see
    ``service.notify_acknowledged_escalations``) and the digest
    is where an operator reads whether it happened: ``delivered``
    (the escalation event landed), ``announced`` (fired, latest
    attempt failed, attempts remain — delivery is in trouble but
    a retry is still the move), ``exhausted`` (its ledger spent
    the attempt budget undelivered — the case was told about and
    called on, and the escalation saying so never landed), or
    ``unannounced`` (no notice on the record at all: the
    escalation check has not run for this case, or ran with no
    webhook configured — either way nothing has left the
    building, and the digest is the surface that notices a
    sweep nobody ran). A case not yet on the second rung reads
    ``not_escalated``: no telling is owed yet.
    """
    notice = record.escalation_notice
    attempts = record.escalation_attempts or []
    if notice is None:
        state = "unannounced"
    elif record.result.submission_id in exhausted_ids:
        state = "exhausted"
    elif attempts and attempts[-1].get("status") == "sent":
        state = "delivered"
    else:
        state = "announced"
    last = attempts[-1] if attempts else {}
    return {
        "state": state,
        "attempts": len(attempts),
        "first_observed_at": (notice or {}).get("first_observed_at") or "",
        "last_attempt_at": last.get("attempted_at") or "",
        "last_http_status": last.get("http_status"),
        "last_error": last.get("error"),
    }


def ops_digest(
    records: list[ApprovalRecord],
    proposals: list[dict] | None = None,
    now: datetime | None = None,
    *,
    sla_budgets_map: dict[str, int] | None = None,
    follow_escalation_days: int | None = None,
    proposal_warning_days: int | None = None,
    acknowledged_wait_days: int | None = None,
    unannounced_gap_days: int | None = None,
    latest_snapshot: dict | None = None,
    broker_drafting_events: list[dict] | None = None,
) -> dict:
    """One operator digest over the surfaces that rot quietly.

    Worklists that used to live on separate screens, composed
    into a single read: **stuck dispatches** (approval
    webhooks failing past the stuck age with retries still left —
    :func:`dispatch_overview`), **exhausted breach notices**
    (SLA-breach notices whose own ledger spent its attempts
    undelivered, on cases still undecided — the same overview,
    a different list: nobody was told, and no retry is coming;
    notices an operator has acknowledged count nowhere here —
    the case has been picked up — and ride along in
    ``acknowledged_breach_notices`` with the acknowledgement,
    so the worked ones stay auditable from the same read), and
    **acknowledged but still waiting** (an acknowledged notice
    whose case is still undecided a follow-through-escalation
    window later — the call happened; the file still has not
    moved — counted in ``total_attention`` on purpose: the
    breached-queue count asks who picks the case up, this one
    asks why the decision still has not landed; entries still
    waiting at twice the window are flagged ``escalated`` and
    counted in ``acknowledged_escalated`` — a subset, never
    counted into ``total_attention`` a second time: the same
    cases, louder, and by then also on the queue's escalated
    surfaces with the acknowledgement attached). Each
    still-waiting entry also carries its **escalation-telling
    state** (see :func:`_escalation_notice_state`): whether the
    second rung's own webhook was ever fired for the case, and
    what became of it. The telling gets its own counts —
    ``exhausted_escalation_notices`` (the escalation's ledger
    spent, undelivered: a different event from an exhausted
    breach notice, where nobody was told at all — here the case
    was told about and called on; the escalation saying so is
    what never landed) and ``unannounced_escalations``
    (escalated cases with no notice on the record: the
    escalation check has not run, and this digest is the
    surface that notices a sweep nobody ran). The unannounced
    are also **aged**: each carries ``unannounced_days`` — the
    days since the case crossed the second rung (its
    acknowledgement reaching twice the window) with no telling
    — the headline names the oldest, and the cases unannounced
    past the gap threshold (``unannounced_gap_days``: one
    further window by default) are counted on their own as
    ``stale_unannounced_escalations``, because a sweep that has
    not run yet today and a sweep that stopped a month ago are
    the same count and different facts. Every telling count is
    a subset of cases already counted in
    ``acknowledged_still_waiting``, so none enters
    ``total_attention`` — the same cases, read for the state
    of their telling, not counted twice),
    **breached queue cases** (awaiting
    submissions past their tier's SLA budget — :func:`queue_aging`,
    measured against the budgets in force), **outstanding
    follow-through** (status notes whose loop is still open, the
    aged ones flagged escalated — :func:`follow_through_digest`),
    and **calibration proposals nearing expiry** (live drafts
    whose evidence window closes within the warning days, plus
    the expired drafts awaiting a refresh — the caller passes the
    proposals annotated by ``proposals.annotate_proposal``). The
    counts also carry the **letter-advisory aggregates** (runs
    that drafted advisories, runs whose advisories the steered
    redraft cleared — the full family rides along as
    ``letter_advisories``, the same computation ``/metrics``
    reports), not as worklist items but so the snapshots that
    keep these counts trend **drafting quality** across the kept
    history beside the operations movement. The note-side
    drafting quality rides the same counts — status notes the
    steering repaired and status-note drafts refused, the full
    family along as ``status_note_repairs`` — so the kept
    history trends both broker-facing drafts, not only the
    letter. The third drafting surface rides the same counts
    too — the **broker quality report's** drafted / repaired /
    refused counts, computed from the drafting events the
    service passes in (``broker_drafting_events``), the full
    aggregate along as ``broker_quality_repairs`` — so a kept
    snapshot trends all three surfaces the drafting discipline
    guards, each under its own source's numbers. The counts likewise
    carry the **worked follow-through response** (notes touched,
    and escalations a fresh touch is currently quieting —
    :func:`follow_through_digest`'s own counts): the problem
    counts alone cannot say whether the response keeps pace
    with the escalations, and the kept snapshots are where that
    pace is read. The counts also carry how many notes have a
    measured **response latency** (``follow_through_responses``:
    escalation to first touch), and the digest body carries the
    latency aggregate itself
    (``follow_through_response_latency`` — median / mean / max
    days), so the morning read prices the response's speed as
    well as its volume. Effort, like drafting quality, never
    enters ``total_attention``.

    The digest is computed, never sent: it carries the composed
    ``headline`` an operator reads first and the section entries
    behind it. Pure projection — reading it writes nothing,
    retries nothing, and refreshes nothing.

    With ``latest_snapshot`` — the newest stored digest snapshot
    (see :func:`build_digest_snapshot`) — the digest is also
    **trend-aware**: a ``movement`` block reads the live counts
    and section membership against the snapshot's (per-count
    change and direction, per-section entered/cleared ids), and
    the headline gains one sentence speaking that movement.
    Without a snapshot the block is ``None`` — movement is
    absent, never invented.
    """
    now = now or datetime.now(timezone.utc)
    if proposal_warning_days is None:
        proposal_warning_days = ops_proposal_expiry_warning_days()
    if acknowledged_wait_days is None:
        acknowledged_wait_days = (
            follow_escalation_days
            if follow_escalation_days is not None
            else follow_through_escalation_days()
        )
    if unannounced_gap_days is None:
        unannounced_gap_days = ops_unannounced_gap_days()

    dispatch = dispatch_overview(records, now)
    still_waiting = _acknowledged_still_waiting(
        dispatch["acknowledged_breach_notices"], now, acknowledged_wait_days
    )
    # The still-waiting entries read acknowledgement entries,
    # which do not carry the record; the escalation telling
    # lives on the record, so it is joined here by id. Only an
    # escalated entry is owed a telling — the rest read
    # not_escalated, so the field is uniform down the list.
    record_by_id = {r.result.submission_id: r for r in records}
    exhausted_escalation_ids = {
        entry["submission_id"]
        for entry in dispatch["exhausted_escalation_notices"]
    }
    unannounced_escalations = 0
    stale_unannounced_escalations = 0
    oldest_unannounced_days = 0
    for entry in still_waiting:
        record = record_by_id.get(entry["submission_id"])
        if not entry["escalated"] or record is None:
            entry["escalation_notice_state"] = {"state": "not_escalated"}
            continue
        state = _escalation_notice_state(record, exhausted_escalation_ids)
        entry["escalation_notice_state"] = state
        if state["state"] == "unannounced":
            unannounced_escalations += 1
            # The gap, aged: the case crossed the second rung
            # when its acknowledgement reached twice the window,
            # so the unannounced age is what has accumulated
            # since — the entry knows both numbers, and a count
            # without the age cannot tell a sweep that has not
            # run yet today from one that stopped a month ago.
            entry["unannounced_days"] = max(
                0, entry["acknowledged_age_days"] - 2 * acknowledged_wait_days
            )
            oldest_unannounced_days = max(
                oldest_unannounced_days, entry["unannounced_days"]
            )
            if entry["unannounced_days"] >= unannounced_gap_days:
                stale_unannounced_escalations += 1
    queue = queue_aging(records, now, sla_budgets_map=sla_budgets_map)
    follow = follow_through_digest(
        records, now, escalation_days=follow_escalation_days
    )

    outstanding = [
        e for e in follow["entries"] if e["state"] in _FOLLOW_OPEN_STATES
    ]
    expiring_soon: list[dict] = []
    expired: list[dict] = []
    for proposal in proposals or []:
        if proposal.get("status") != "draft":
            continue
        expires = _parse_timestamp(proposal.get("expires_at"))
        entry = {
            "proposal_id": proposal.get("proposal_id", ""),
            "kind": proposal.get("kind", ""),
            "dimension": proposal.get("dimension", ""),
            "current_value": proposal.get("current_value"),
            "proposed_value": proposal.get("proposed_value"),
            "expires_at": proposal.get("expires_at"),
            "days_until_expiry": (
                max(0, (expires.date() - now.date()).days)
                if expires is not None
                else None
            ),
        }
        is_expired = proposal.get("expired")
        if is_expired is None:
            # Un-annotated input: resolve expiry from the stamp.
            is_expired = expires is not None and now > expires
        if is_expired:
            expired.append(entry)
        elif (
            entry["days_until_expiry"] is not None
            and entry["days_until_expiry"] <= proposal_warning_days
        ):
            expiring_soon.append(entry)
    expiring_soon.sort(
        key=lambda e: (e["days_until_expiry"], e["proposal_id"])
    )
    expired.sort(key=lambda e: e["proposal_id"])

    advisory = _letter_advisory_aggregate(records)
    note_repairs = _status_note_repair_aggregate(records)
    broker = _broker_quality_repair_aggregate(broker_drafting_events or [])
    counts = {
        "stuck_dispatches": dispatch["stuck_count"],
        "exhausted_breach_notices": dispatch["exhausted_breach_count"],
        "acknowledged_still_waiting": len(still_waiting),
        "acknowledged_escalated": sum(
            1 for entry in still_waiting if entry["escalated"]
        ),
        # The second rung's telling, counted on the subset
        # discipline (see the docstring): exhausted escalation
        # notices are the still-waiting cases whose escalation
        # event spent its attempts undelivered, and unannounced
        # escalations are the escalated cases no escalation
        # check has ever fired for. Neither enters
        # total_attention — the cases are already counted in
        # acknowledged_still_waiting; these counts read the
        # state of their telling.
        "exhausted_escalation_notices": dispatch[
            "exhausted_escalation_count"
        ],
        "unannounced_escalations": unannounced_escalations,
        # The stale subset of the unannounced: cases unannounced
        # for at least the gap threshold past the rung — the
        # gap itself, counted. A subset of a subset of
        # still-waiting, so it never enters total_attention.
        "stale_unannounced_escalations": stale_unannounced_escalations,
        "breached_queue": queue["breach_count"],
        "follow_through_outstanding": len(outstanding),
        "follow_through_escalated": follow["escalated_count"],
        # The worked response rides the same counts, on the
        # advisory discipline: notes an underwriter has touched
        # (worked) and escalations a fresh touch is currently
        # quieting (quieted) are effort signals the snapshots
        # keep so the trend can say whether touches keep pace
        # with escalations — never part of total_attention,
        # which counts work outstanding, not work done.
        "follow_through_worked": follow["worked_count"],
        "follow_through_quieted": follow["quieted_count"],
        # The pace of that response, counted: how many notes
        # carry a measured response latency (escalation to
        # first touch — see follow_through_digest). The count
        # rides the snapshots like the other effort counts; the
        # median itself is a statistic, not a count, so it
        # rides the digest body as follow_through_response_latency
        # — the same split the letter advisories use (counts in
        # the snapshots, rates in the aggregate).
        "follow_through_responses": follow["response_latency"][
            "notes_measured"
        ],
        # Drafting quality rides the same counts the snapshots
        # keep (advisory_runs_with / advisory_repaired_runs), so
        # the kept history trends it beside the operations
        # counts; the full aggregate rides along below as
        # ``letter_advisories``. Neither enters total_attention —
        # advisories are a quality signal, not a worklist.
        "advisory_runs_with": advisory["runs_with_advisories"],
        "advisory_repaired_runs": advisory["repaired_runs"],
        # The note-side drafting quality rides the same counts
        # on the same discipline: status notes the steering
        # repaired, and status-note drafts refused outright —
        # the full family rides along below as
        # ``status_note_repairs``. Neither enters
        # total_attention — a repaired or refused draft is a
        # data-quality signal, not a worklist.
        "status_note_repaired": note_repairs["repaired_notes"],
        "status_note_refusals": note_repairs["refusals"],
        # The third drafting surface rides the same counts on
        # the same discipline: broker quality reports drafted,
        # steered repairs among them, and drafts refused — the
        # full family rides along below as
        # ``broker_quality_repairs``, counted from the drafting
        # events (the report is drafted on read, so the events
        # are the count's only honest source). Neither enters
        # total_attention — drafting quality is a signal, not a
        # worklist, on this surface too.
        "broker_quality_drafted": broker["reports_drafted"],
        "broker_quality_repaired": broker["repaired_reports"],
        "broker_quality_refusals": broker["refusals"],
        "proposals_expiring_soon": len(expiring_soon),
        "proposals_expired": len(expired),
    }
    counts["total_attention"] = (
        counts["stuck_dispatches"]
        + counts["exhausted_breach_notices"]
        + counts["acknowledged_still_waiting"]
        + counts["breached_queue"]
        + counts["follow_through_escalated"]
        + counts["proposals_expired"]
    )
    parts = []
    if counts["stuck_dispatches"]:
        parts.append(
            f"{counts['stuck_dispatches']} stuck dispatch(es)"
        )
    if counts["exhausted_breach_notices"]:
        parts.append(
            f"{counts['exhausted_breach_notices']} exhausted breach "
            "notice(s) — a case nobody was told about"
        )
    if counts["acknowledged_still_waiting"]:
        part = (
            f"{counts['acknowledged_still_waiting']} acknowledged "
            "breach notice(s) still waiting on a decision — the "
            "call happened; the file has not moved"
        )
        if counts["acknowledged_escalated"]:
            part += (
                f"; {counts['acknowledged_escalated']} of them past "
                "the second window, escalated to the queue"
            )
        parts.append(part)
    if counts["exhausted_escalation_notices"]:
        parts.append(
            f"{counts['exhausted_escalation_notices']} exhausted "
            "escalation notice(s) — the case was told about and "
            "called on; the escalation saying so never landed"
        )
    if counts["unannounced_escalations"]:
        part = (
            f"{counts['unannounced_escalations']} escalated "
            "case(s) never announced — the escalation check "
            "has not run"
        )
        if oldest_unannounced_days:
            part += (
                f"; the oldest has sat unannounced "
                f"{oldest_unannounced_days} day(s)"
            )
        parts.append(part)
    if counts["stale_unannounced_escalations"]:
        parts.append(
            f"{counts['stale_unannounced_escalations']} of them "
            f"unannounced past the {unannounced_gap_days}-day "
            "gap threshold — the sweep gap itself needs attention"
        )
    if counts["breached_queue"]:
        parts.append(f"{counts['breached_queue']} breached queue case(s)")
    if counts["follow_through_escalated"]:
        parts.append(
            f"{counts['follow_through_escalated']} escalated "
            "follow-through note(s)"
        )
    elif counts["follow_through_outstanding"]:
        parts.append(
            f"{counts['follow_through_outstanding']} follow-through "
            "note(s) still outstanding"
        )
    if counts["proposals_expired"]:
        parts.append(
            f"{counts['proposals_expired']} expired proposal(s) "
            "awaiting refresh"
        )
    if counts["proposals_expiring_soon"]:
        parts.append(
            f"{counts['proposals_expiring_soon']} proposal(s) nearing "
            "expiry"
        )
    headline = (
        "Ops digest: " + "; ".join(parts) + "."
        if parts
        else "Ops digest: nothing needs attention — no stuck "
        "dispatches, no exhausted breach notices, no exhausted "
        "escalation notices, no acknowledged "
        "notices still waiting, no breached cases, no escalated "
        "notes, no proposals nearing expiry."
    )
    sections = {
        "stuck_dispatches": dispatch["stuck_failures"],
        "exhausted_breach_notices": dispatch["exhausted_breach_notices"],
        "exhausted_escalation_notices": dispatch[
            "exhausted_escalation_notices"
        ],
        "acknowledged_still_waiting": still_waiting,
        "breached_queue": queue["breaches"],
        "follow_through_outstanding": outstanding,
        "follow_through_escalated": follow["escalated"],
        "follow_through_worked": follow["worked"],
        "proposals_expiring_soon": expiring_soon,
        "proposals_expired": expired,
    }
    # Trend-aware: when a snapshot exists, the digest reads
    # itself against it — the movement block carries the deltas
    # as data and the headline speaks them as one sentence, so
    # the first screen an operator opens already says what got
    # worse. No snapshot, no movement: the block is None rather
    # than a fabricated baseline.
    movement = None
    if latest_snapshot is not None:
        movement = _digest_movement(counts, sections, latest_snapshot)
        headline = f"{headline} {_movement_sentence(movement)}"
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "headline": headline,
        "counts": counts,
        **sections,
        "acknowledged_breach_notices": dispatch[
            "acknowledged_breach_notices"
        ],
        "letter_advisories": advisory,
        "status_note_repairs": note_repairs,
        "broker_quality_repairs": broker,
        "follow_through_response_latency": follow["response_latency"],
        "acknowledged_wait_days": acknowledged_wait_days,
        "unannounced_gap_days": unannounced_gap_days,
        "proposal_warning_days": proposal_warning_days,
        "movement": movement,
    }


# ---------------------------------------------------------------------------
# Ops digest snapshots — the digest, kept, so movement is visible
# ---------------------------------------------------------------------------

#: Store document key for the ops-digest snapshots (see
#: :func:`build_digest_snapshot`): ``{"snapshots": [...]}``, one
#: appended document per operator-taken snapshot.
DIGEST_SNAPSHOTS_KEY = "ops_digest_snapshots"

#: The digest's headline counts, in headline order — the numbers
#: a trend compares across snapshots.
DIGEST_COUNT_KEYS = (
    "stuck_dispatches",
    "exhausted_breach_notices",
    "acknowledged_still_waiting",
    "acknowledged_escalated",
    "exhausted_escalation_notices",
    "unannounced_escalations",
    "stale_unannounced_escalations",
    "breached_queue",
    "follow_through_outstanding",
    "follow_through_escalated",
    "follow_through_worked",
    "follow_through_quieted",
    "follow_through_responses",
    "advisory_runs_with",
    "advisory_repaired_runs",
    "status_note_repaired",
    "status_note_refusals",
    "broker_quality_drafted",
    "broker_quality_repaired",
    "broker_quality_refusals",
    "proposals_expiring_soon",
    "proposals_expired",
    "total_attention",
)

#: Digest sections whose entries carry a stable id, mapped to the
#: field that carries it — a snapshot keeps each section's ids so
#: a trend can say what entered and what cleared, not just that a
#: count moved.
_DIGEST_SECTION_IDS = {
    "stuck_dispatches": "submission_id",
    "exhausted_breach_notices": "submission_id",
    "exhausted_escalation_notices": "submission_id",
    "acknowledged_still_waiting": "submission_id",
    "breached_queue": "submission_id",
    "follow_through_outstanding": "submission_id",
    "follow_through_escalated": "submission_id",
    "proposals_expiring_soon": "proposal_id",
    "proposals_expired": "proposal_id",
}


#: Headline-count labels for the digest's own prose — the
#: movement sentence the trend-aware digest speaks.
_COUNT_LABELS = {
    "stuck_dispatches": "stuck dispatches",
    "exhausted_breach_notices": "exhausted breach notices",
    "acknowledged_still_waiting": "acknowledged notices still waiting",
    "acknowledged_escalated": "acknowledged notices escalated",
    "exhausted_escalation_notices": "exhausted escalation notices",
    "unannounced_escalations": "unannounced escalations",
    "stale_unannounced_escalations": "stale unannounced escalations",
    "breached_queue": "breached queue cases",
    "follow_through_outstanding": "outstanding follow-through notes",
    "follow_through_escalated": "escalated follow-through notes",
    "follow_through_worked": "worked follow-through notes",
    "follow_through_quieted": "work-quieted follow-through notes",
    "follow_through_responses": "follow-through responses",
    "advisory_runs_with": "runs drafting letter advisories",
    "advisory_repaired_runs": "advisory repairs",
    "status_note_repaired": "repaired status notes",
    "status_note_refusals": "refused status-note drafts",
    "broker_quality_drafted": "broker quality reports drafted",
    "broker_quality_repaired": "repaired broker quality drafts",
    "broker_quality_refusals": "refused broker quality drafts",
    "proposals_expiring_soon": "proposals nearing expiry",
    "proposals_expired": "expired proposals",
    "total_attention": "total attention",
}


def _digest_movement(
    counts: dict, sections: dict, snapshot: dict
) -> dict:
    """A live digest read against the latest stored snapshot.

    Per headline count: the snapshot's value, the current value,
    the change, and its direction. Per section: which ids
    **entered** and which **cleared** since the snapshot — the
    same entered/cleared discipline :func:`digest_trend` applies
    between two snapshots, here between a snapshot and now. A
    snapshot taken before a count or section existed reads as
    zero/empty for it — absence in the snapshot is a fact about
    the snapshot, not something to invent around.
    """
    snap_counts = snapshot.get("counts") or {}
    count_rows: dict[str, dict] = {}
    for key in DIGEST_COUNT_KEYS:
        before = snap_counts.get(key, 0)
        current = counts.get(key, 0)
        change = current - before
        count_rows[key] = {
            "snapshot": before,
            "current": current,
            "change": change,
            "direction": (
                "up" if change > 0 else "down" if change < 0 else "flat"
            ),
        }
    snap_sections = snapshot.get("sections") or {}
    section_rows: dict[str, dict] = {}
    for section, id_field in _DIGEST_SECTION_IDS.items():
        before_ids = set(snap_sections.get(section) or [])
        current_ids = {
            entry[id_field]
            for entry in sections.get(section) or []
            if entry.get(id_field)
        }
        section_rows[section] = {
            "entered": sorted(current_ids - before_ids),
            "cleared": sorted(before_ids - current_ids),
        }
    return {
        "since_snapshot": snapshot.get("snapshot_id", ""),
        "snapshot_taken_at": snapshot.get("taken_at", ""),
        "snapshot_taken_by": snapshot.get("taken_by", ""),
        "counts": count_rows,
        "sections": section_rows,
    }


def _movement_sentence(movement: dict) -> str:
    """The movement block as prose in the digest's own voice:
    each headline count that moved, named and directed, and the
    membership behind the movement — which ids entered and which
    cleared, woven into the same line — or the plain statement
    that nothing did. Counts alone read as arithmetic on every
    surface the headline reaches; an operator works ids, so the
    headline names them. A count can sit flat while its
    membership churns (one case cleared, another entered), and
    the line says so rather than reporting no movement."""
    moved = [
        f"{_COUNT_LABELS[key]} {row['direction']} "
        f"{abs(row['change'])}"
        for key, row in movement["counts"].items()
        if row["change"]
    ]
    membership = []
    sections = movement.get("sections") or {}
    for section in _DIGEST_SECTION_IDS:
        row = sections.get(section) or {}
        label = _COUNT_LABELS[section]
        if row.get("entered"):
            membership.append(
                f"{', '.join(row['entered'])} entered {label}"
            )
        if row.get("cleared"):
            membership.append(
                f"{', '.join(row['cleared'])} cleared {label}"
            )
    if not moved and not membership:
        return (
            f"Since {movement['since_snapshot']}: no movement in "
            "the headline counts."
        )
    sentence = (
        f"Since {movement['since_snapshot']}: "
        + (", ".join(moved) if moved else "no movement in the headline counts")
    )
    if membership:
        sentence += " — " + "; ".join(membership)
    return sentence + "."


def build_digest_snapshot(
    digest: dict, snapshot_id: str, actor: str
) -> dict:
    """One stored snapshot of an ops digest, taken on demand.

    The digest is computed at read time and evaporates; a snapshot
    is the digest as an operator chose to keep it — its headline,
    its counts, and the ids behind each section — stamped with
    who took it and when (the digest's own ``generated_at``, so
    the snapshot's date is the digest's date, not a second clock).
    Snapshots are never taken on a timer: a snapshot nobody asked
    for is noise, and the trend below reads only what an operator
    decided was worth keeping.
    """
    counts = {
        key: (digest.get("counts") or {}).get(key, 0)
        for key in DIGEST_COUNT_KEYS
    }
    sections = {
        section: sorted(
            {
                entry[id_field]
                for entry in digest.get(section) or []
                if entry.get(id_field)
            }
        )
        for section, id_field in _DIGEST_SECTION_IDS.items()
    }
    return {
        "snapshot_id": snapshot_id,
        "taken_at": digest.get("generated_at", ""),
        "taken_by": actor,
        "headline": digest.get("headline", ""),
        "counts": counts,
        "sections": sections,
    }


def digest_trend(snapshots: list[dict]) -> dict:
    """Movement across the stored digest snapshots.

    The headline counts as a series (oldest snapshot first, the
    order they were taken), each count's movement from the first
    snapshot to the latest and from the previous snapshot to the
    latest, and — between the latest two snapshots — which ids
    **entered** and which **cleared** each section. A count that
    moves without its membership changing reads differently from
    churn, and the trend keeps the two apart. Fewer than two
    snapshots is not a trend: the deltas say so instead of
    inventing movement.
    """
    series = [
        {
            "snapshot_id": snap.get("snapshot_id", ""),
            "taken_at": snap.get("taken_at", ""),
            "taken_by": snap.get("taken_by", ""),
            "counts": {
                key: (snap.get("counts") or {}).get(key, 0)
                for key in DIGEST_COUNT_KEYS
            },
        }
        for snap in snapshots
    ]
    deltas: dict[str, dict] = {}
    for key in DIGEST_COUNT_KEYS:
        if not series:
            deltas[key] = {"first": None, "latest": None, "change": None, "since_previous": None}
            continue
        first = series[0]["counts"][key]
        latest = series[-1]["counts"][key]
        previous = series[-2]["counts"][key] if len(series) > 1 else None
        deltas[key] = {
            "first": first,
            "latest": latest,
            "change": latest - first if len(series) > 1 else None,
            "since_previous": (
                latest - previous if previous is not None else None
            ),
        }
    movement: dict | None = None
    if len(snapshots) > 1:
        before = snapshots[-2].get("sections") or {}
        after = snapshots[-1].get("sections") or {}
        movement = {
            "from_snapshot": snapshots[-2].get("snapshot_id", ""),
            "to_snapshot": snapshots[-1].get("snapshot_id", ""),
            "sections": {
                section: {
                    "entered": sorted(
                        set(after.get(section) or [])
                        - set(before.get(section) or [])
                    ),
                    "cleared": sorted(
                        set(before.get(section) or [])
                        - set(after.get(section) or [])
                    ),
                }
                for section in _DIGEST_SECTION_IDS
            },
        }
    return {
        "snapshot_count": len(series),
        "series": series,
        "deltas": deltas,
        "movement": movement,
    }


# ---------------------------------------------------------------------------
# Retention — one discipline for the kept documents: the two
# snapshot documents, the broker drafting events, and (in
# proposals.py) the proposals document's decided history
# ---------------------------------------------------------------------------


def prune_snapshot_list(
    snapshots: list[dict], keep: int
) -> tuple[list[dict], list[dict]]:
    """Split a kept-snapshot list for a keep-last-N prune:
    (survivors, pruned), order preserved in both.

    The one retention discipline for every kept-report document
    — the digest snapshots above and the calibration snapshots
    in proposals.py. Snapshots are kept on demand and accumulate
    one per ask, unboundedly: a year of weekly snapshots becomes
    an archive nobody trends. Pruning is the only way a kept
    snapshot ever leaves its document, and it is operator-called,
    never a timer. ``keep`` below 1 is refused: a prune that
    emptied the document would erase the baseline every trend
    reads against, silently.
    """
    if keep < 1:
        raise ValueError(
            "keep must be at least 1 — a prune keeps the newest "
            "snapshots; emptying the document would erase the "
            "baseline the trends read against"
        )
    if len(snapshots) <= keep:
        return list(snapshots), []
    return list(snapshots[-keep:]), list(snapshots[:-keep])


def snapshot_prune_preview(snapshots: list[dict], keep: int) -> dict:
    """A prune, dry-run: what a keep-last-N prune *would* do.

    The restore/refresh preview idiom applied to retention: the
    prune is exact about what left only after it left, so an
    operator deciding whether the oldest kept reports may go had
    to commit blind. The preview runs the same split the prune
    runs (:func:`prune_snapshot_list` — ``keep`` below 1 is
    refused here too) and reports the ids that would leave and
    the range that would survive, writing nothing: the preview
    and the prune that follows it on the same document cannot
    disagree, because both read the one computation.
    """
    kept, pruned = prune_snapshot_list(snapshots, keep)
    return {
        "snapshot_count": len(snapshots),
        "keep": keep,
        "would_change": bool(pruned),
        "would_prune_count": len(pruned),
        "would_prune_ids": [s.get("snapshot_id", "") for s in pruned],
        "surviving_count": len(kept),
        "surviving_first": kept[0].get("snapshot_id", "") if kept else "",
        "surviving_last": kept[-1].get("snapshot_id", "") if kept else "",
    }


def snapshot_prune_entry(
    *,
    actor: str,
    keep: int,
    pruned: list[dict],
    kept: list[dict],
    pruned_at: str,
) -> dict:
    """The prune's own record, kept on the document it pruned:
    who pruned, when, what they asked to keep, which snapshots
    left, and the surviving range — so a gap in a trend is
    explained by the document itself, not by archaeology."""
    return {
        "pruned_at": pruned_at,
        "pruned_by": actor,
        "requested_keep": keep,
        "pruned_count": len(pruned),
        "pruned_ids": [s.get("snapshot_id", "") for s in pruned],
        "surviving_first": kept[0].get("snapshot_id", "") if kept else "",
        "surviving_last": kept[-1].get("snapshot_id", "") if kept else "",
    }


def prune_drafting_event_list(
    events: list[dict], keep: int
) -> tuple[list[dict], list[dict]]:
    """Split a drafting-event list for a keep-last-N prune:
    (survivors, pruned), order preserved in both.

    The snapshot retention discipline applied to the
    broker-quality drafting events: one event lands per served
    report, unboundedly, and the aggregate that reads them
    needs history, not infinity. Pruning is the only way an
    event ever leaves its document, and it is operator-called,
    never a timer. ``keep`` below 1 is refused: the events are
    the aggregate's only source, and emptying the document
    would zero the broker drafting counts everywhere they are
    read — snapshots included — with no record of why.
    """
    if keep < 1:
        raise ValueError(
            "keep must be at least 1 — a prune keeps the newest "
            "drafting events; emptying the document would erase "
            "the history the broker drafting counts are computed "
            "from"
        )
    if len(events) <= keep:
        return list(events), []
    return list(events[-keep:]), list(events[:-keep])


def drafting_event_prune_preview(events: list[dict], keep: int) -> dict:
    """A drafting-event prune, dry-run: what it *would* do.

    The snapshot preview's idiom over events, which carry no
    ids: the plan is stated as the count that would leave and
    the drafted-at range it spans, beside the range that would
    survive — writing nothing and recording nothing, so the
    preview and the prune that follows it on the same document
    cannot disagree: both read the one split
    (:func:`prune_drafting_event_list`, whose refusal of
    ``keep`` below 1 applies here too).
    """
    kept, pruned = prune_drafting_event_list(events, keep)
    return {
        "event_count": len(events),
        "keep": keep,
        "would_change": bool(pruned),
        "would_prune_count": len(pruned),
        "would_prune_oldest": (
            pruned[0].get("drafted_at", "") if pruned else ""
        ),
        "would_prune_newest": (
            pruned[-1].get("drafted_at", "") if pruned else ""
        ),
        "surviving_count": len(kept),
        "surviving_first": kept[0].get("drafted_at", "") if kept else "",
        "surviving_last": kept[-1].get("drafted_at", "") if kept else "",
    }


def drafting_event_prune_entry(
    *,
    actor: str,
    keep: int,
    pruned: list[dict],
    kept: list[dict],
    pruned_at: str,
) -> dict:
    """The event prune's own record, kept on the document it
    pruned: who pruned, when, what they asked to keep, how many
    events left and the drafted-at range they spanned, and the
    range that survived — so a step down in the broker drafting
    counts is explained by the document itself: the counts are
    computed from the events, and a prune is the one act that
    legitimately moves them backwards."""
    return {
        "pruned_at": pruned_at,
        "pruned_by": actor,
        "requested_keep": keep,
        "pruned_count": len(pruned),
        "pruned_oldest": pruned[0].get("drafted_at", "") if pruned else "",
        "pruned_newest": pruned[-1].get("drafted_at", "") if pruned else "",
        "surviving_first": kept[0].get("drafted_at", "") if kept else "",
        "surviving_last": kept[-1].get("drafted_at", "") if kept else "",
    }


def next_snapshot_id(snapshots: list[dict], prefix: str) -> str:
    """The next id for a kept snapshot: one past the highest
    number still in the document, not one past its length. After
    a prune, a length-based id would reissue a number a pruned
    snapshot already wore — and a citation like "snap-0007" must
    name exactly one kept report, forever."""
    highest = 0
    for snap in snapshots:
        sid = str(snap.get("snapshot_id") or "")
        if sid.startswith(prefix):
            try:
                highest = max(highest, int(sid[len(prefix):]))
            except ValueError:
                continue
    return f"{prefix}{highest + 1:04d}"
