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
