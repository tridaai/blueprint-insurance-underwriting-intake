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

from .facts import _parse_date
from .store import ApprovalRecord, history_entry


def _party_name(record: ApprovalRecord, kind: str) -> str:
    entry = history_entry(record)
    return entry["insured_name"] if kind == "insured" else entry["broker_name"]


def inbox_groups(records: list[ApprovalRecord], kind: str) -> list[dict]:
    """Group stored submissions by insured (``kind="insured"``) or
    broker (``kind="broker"``).

    Records arrive in store order (oldest first); "latest" is the
    most recently analysed submission for the party. Groups are
    returned most-recent-activity first.
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
    return {
        "kind": kind,
        "name": name,
        "submissions": len(hits),
        "outcome_mix": dict(sorted(outcome_mix.items())),
        "decisions": decisions,
        "approvals": approvals,
        "rejections": rejections,
        "approval_rate": round(approvals / decisions, 3) if decisions else None,
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


def queue_aging(
    records: list[ApprovalRecord], now: datetime | None = None
) -> dict:
    """The awaiting queue by age — the stranded-queue early warning.

    Age is whole days from the submission's ``submitted_at`` to
    today (the same date arithmetic ``facts.submission_age_days``
    uses). Every awaiting submission lands in exactly one bucket —
    or in ``unknown_age`` when its submitted date cannot be parsed,
    which is itself worth seeing. Decided records are not queue.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()
    buckets: dict[str, list[dict]] = {label: [] for label, _, _ in QUEUE_AGE_BUCKETS}
    unknown: list[str] = []
    by_outcome: Counter = Counter()
    oldest: dict | None = None
    total = 0
    for record in records:
        result = record.result
        if result.status != "awaiting_underwriter":
            continue
        total += 1
        by_outcome[result.triage.outcome] += 1
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
        if submitted is None:
            unknown.append(result.submission_id)
            continue
        age = max(0, (today - submitted).days)
        item["age_days"] = age
        for label, low, high in QUEUE_AGE_BUCKETS:
            if age >= low and (high is None or age <= high):
                buckets[label].append(item)
                break
        if oldest is None or age > oldest["age_days"]:
            oldest = item
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


def dispatch_overview(
    records: list[ApprovalRecord], now: datetime | None = None
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
    """
    now = now or datetime.now(timezone.utc)
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
        backoff = _next_backoff_seconds(len(attempts))
        due_at = last_at + timedelta(seconds=backoff) if last_at else None
        failures.append(
            {
                "submission_id": record.result.submission_id,
                "insured_name": entry["insured_name"],
                "broker_name": entry["broker_name"],
                "attempts": len(attempts),
                "last_attempt_at": last.get("attempted_at") or "",
                "last_http_status": last.get("http_status"),
                "last_error": last.get("error"),
                "next_attempt": len(attempts) + 1,
                "next_backoff_seconds": backoff,
                "retry_due": due_at is None or now >= due_at,
                "_sort": last_at or datetime.max.replace(tzinfo=timezone.utc),
            }
        )
        if last_at is not None and (oldest_key is None or last_at < oldest_key):
            oldest_key = last_at
            oldest_failure_at = last.get("attempted_at")
    failures.sort(key=lambda f: (f["_sort"], f["submission_id"]))
    for failure in failures:
        failure.pop("_sort", None)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "attempts_total": attempts_total,
        "records_with_attempts": sent_records + len(failures),
        "sent_records": sent_records,
        "failed_records": len(failures),
        "due_retries": [f for f in failures if f["retry_due"]],
        "failures": failures,
        "oldest_failure_at": oldest_failure_at,
    }
