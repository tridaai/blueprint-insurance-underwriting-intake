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
    like every draft in this system, never sent by the agent.
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
    report["draft"] = compose_broker_quality_draft(report)
    return report


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


def compute_metrics(records: list[ApprovalRecord]) -> dict:
    """Operational aggregates over the whole store.

    Run and decision counts, outcome and tier mix, guardrail and
    reviewer signals, and — from each run's recorded telemetry —
    model calls, tokens, estimated cost, and latency. Token/cost
    figures only include runs whose backend reported them (the
    offline fallback reports tokens as None rather than inventing
    numbers, and says so via ``costed_runs``/``token_runs``).
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
    """
    now = now or datetime.now(timezone.utc)
    budgets = sla_budgets_map if sla_budgets_map is not None else sla_budgets()
    buckets: dict[str, list[dict]] = {label: [] for label, _, _ in QUEUE_AGE_BUCKETS}
    unknown: list[str] = []
    by_outcome: Counter = Counter()
    oldest: dict | None = None
    breaches: list[dict] = []
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
        "blocked_bundles": blocked_bundles,
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
    }
