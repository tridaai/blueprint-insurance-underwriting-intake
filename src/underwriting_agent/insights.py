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
    decisions = statuses.get("approved", 0) + statuses.get("rejected", 0)
    return {
        "runs": len(records),
        "awaiting_underwriter": statuses.get("awaiting_underwriter", 0),
        "approved": statuses.get("approved", 0),
        "rejected": statuses.get("rejected", 0),
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


def audit_csv(rows: list[dict]) -> str:
    """The audit rows as CSV text (header + one line per row)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in AUDIT_COLUMNS})
    return buffer.getvalue()
