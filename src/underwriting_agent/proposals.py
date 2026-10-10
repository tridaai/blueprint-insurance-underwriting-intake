"""Calibration proposals + the threshold settings record.

Round 7's calibration view replays the stored renewals at
candidate diff-trigger thresholds and reports what each setting
would have referred — but it stops at arithmetic: the human still
has to notice which row matters. This module closes that loop,
carefully:

- :func:`draft_proposals` scans the same replay for the settings
  whose moved referrals carry one-sided decision evidence — a
  lower floor whose *gained* referrals are all cases underwriters
  rejected (the referral would have been right, and it adds none
  on a case they approved), or a higher floor whose *lost*
  referrals are all cases they approved anyway (the referral
  changed nothing the record shows). A candidate whose moved set
  contains an undecided or contradicting case drafts nothing:
  proposals are made of evidence, not vibes.
- A proposal is a stored document a human accepts or rejects. It
  changes nothing by existing.
- **Accepting** is the one write with consequences: it writes the
  proposed value into the *settings record* (another stored
  document), and the pipeline resolves its diff-trigger thresholds
  from that record first — environment configuration is the base
  the record overrides, per dimension. The record keeps the full
  history (proposal, dimension, from → to, who, when), so every
  threshold in force can be traced to the human who set it and the
  evidence they saw. Appetite stays a human decision; the code
  just stops making the human do the arithmetic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .calibration import DEFAULT_CANDIDATES, current_thresholds, renewal_pairs
from .diffing import diff_referral_triggers
from .ports import Store
from .store import ApprovalRecord

PROPOSALS_KEY = "calibration_proposals"
SETTINGS_KEY = "diff_trigger_thresholds"

_DIMENSION_LABELS = {
    "limit_jump_pct": "limit-jump percentage",
    "new_claim_min_incurred": "new-claim incurred floor",
    "development_min_increase": "adverse-development incurred floor",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _value_text(dimension: str, value: float) -> str:
    if dimension == "limit_jump_pct":
        return f"{value:g}%"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# Documents: proposals + settings
# ---------------------------------------------------------------------------


def load_proposals(store: Store) -> list[dict]:
    """Every stored calibration proposal, oldest first."""
    document = store.get_document(PROPOSALS_KEY) or {}
    return list(document.get("proposals") or [])


def _save_proposals(store: Store, proposals: list[dict]) -> None:
    store.save_document(PROPOSALS_KEY, {"proposals": proposals})


def load_settings(store: Store) -> dict | None:
    """The accepted-thresholds settings record, or None when no
    proposal has ever been accepted (thresholds then come from the
    environment alone)."""
    return store.get_document(SETTINGS_KEY)


def effective_thresholds(store: Store) -> dict:
    """The diff-trigger thresholds the pipeline runs with.

    The environment configuration is the base; the settings record
    (written only by accepting a proposal) overrides it per
    dimension. This is the single resolution the classifier, the
    calibration replay, and the proposal scan all read — they can
    never disagree about which thresholds are in force.
    """
    base = current_thresholds()
    settings = load_settings(store)
    if settings and settings.get("thresholds"):
        for dimension, value in settings["thresholds"].items():
            if dimension in base:
                base[dimension] = float(value)
    return base


def threshold_settings_view(store: Store) -> dict:
    """The thresholds in force, and where they came from."""
    settings = load_settings(store)
    return {
        "thresholds": effective_thresholds(store),
        "source": "accepted_proposal" if settings else "environment",
        "environment_thresholds": current_thresholds(),
        "settings": settings,
    }


# ---------------------------------------------------------------------------
# Drafting proposals from the replay
# ---------------------------------------------------------------------------


def _referred(pair: dict, thresholds: dict) -> bool:
    """Would this stored renewal fire a diff trigger at these
    thresholds? (The replay — the same computation calibration
    reports, per pair.)"""
    return bool(
        diff_referral_triggers(
            pair["current"], pair["current_facts"], pair["prior"], **thresholds
        )
    )


def _decision_split(pairs: list[dict]) -> dict:
    split = {"approved": 0, "rejected": 0, "undecided": 0}
    for pair in pairs:
        if pair["status"] in ("approved", "rejected"):
            split[pair["status"]] += 1
        else:
            split["undecided"] += 1
    return split


def draft_proposals(store: Store, records: list[ApprovalRecord]) -> dict:
    """Scan the stored renewals for proposal-worthy threshold moves.

    One candidate move per dimension at most: among the swept
    values whose moved referrals are *entirely* decided and
    one-sided (see the module docstring), the one moving the most
    decided cases wins; ties go to the smallest change.

    A human's earlier answers bind the scan: a dimension with a
    live (still-draft) proposal is left alone — one open question
    per knob — and a *rejected direction* on a dimension is never
    re-proposed in another size (the human said no to that
    direction of travel; new evidence is weighed by a human
    re-opening the question, not by the scanner re-asking it).
    Returns the newly drafted proposals plus the full stored list.
    """
    pairs = renewal_pairs(records)
    current = effective_thresholds(store)
    existing = load_proposals(store)
    open_dimensions = {p["dimension"] for p in existing if p["status"] == "draft"}
    rejected_directions = {
        (p["dimension"], p["direction"])
        for p in existing
        if p["status"] == "rejected"
    }
    drafted: list[dict] = []
    decided_pairs = [p for p in pairs if p["status"] in ("approved", "rejected")]

    for dimension in (
        "limit_jump_pct",
        "new_claim_min_incurred",
        "development_min_increase",
    ):
        if dimension in open_dimensions:
            continue
        values = sorted(
            {float(v) for v in DEFAULT_CANDIDATES.get(dimension, [])}
            | {current[dimension]}
        )
        best: dict | None = None
        for value in values:
            if value == current[dimension]:
                continue
            direction = "lower" if value < current[dimension] else "raise"
            if (dimension, direction) in rejected_directions:
                continue
            trial = {**current, dimension: value}
            gained = [
                p for p in pairs if _referred(p, trial) and not _referred(p, current)
            ]
            lost = [
                p for p in pairs if _referred(p, current) and not _referred(p, trial)
            ]
            moved = gained if direction == "lower" else lost
            if not moved:
                continue
            split = _decision_split(moved)
            if split["undecided"]:
                continue  # no decision evidence on part of the move — no proposal
            if direction == "lower" and split["approved"]:
                continue  # would add referrals on cases underwriters approved
            if direction == "raise" and split["rejected"]:
                continue  # would drop referrals on cases underwriters rejected
            candidate = {
                "value": value,
                "direction": direction,
                "gained": gained,
                "lost": lost,
                "moved": moved,
                "split": split,
            }
            if (
                best is None
                or len(moved) > len(best["moved"])
                or (
                    len(moved) == len(best["moved"])
                    and abs(value - current[dimension])
                    < abs(best["value"] - current[dimension])
                )
            ):
                best = candidate
        if best is None:
            continue
        label = _DIMENSION_LABELS[dimension]
        moved_ids = [p["submission_id"] for p in best["moved"]]
        if best["direction"] == "lower":
            rationale = (
                f"Lowering the {label} from "
                f"{_value_text(dimension, current[dimension])} to "
                f"{_value_text(dimension, best['value'])} would have referred "
                f"{len(best['moved'])} more stored renewal(s) "
                f"({', '.join(moved_ids)}); underwriters rejected every one "
                "of them, and the move adds no referral on a case they "
                "approved."
            )
        else:
            rationale = (
                f"Raising the {label} from "
                f"{_value_text(dimension, current[dimension])} to "
                f"{_value_text(dimension, best['value'])} would have referred "
                f"{len(best['moved'])} fewer stored renewal(s) "
                f"({', '.join(moved_ids)}); underwriters approved every one "
                "of them anyway, and the move drops no referral on a case "
                "they rejected."
            )
        proposal = {
            "proposal_id": f"PROP-{len(existing) + len(drafted) + 1:04d}",
            "kind": "diff_trigger_threshold",
            "dimension": dimension,
            "current_value": current[dimension],
            "proposed_value": best["value"],
            "direction": best["direction"],
            "status": "draft",
            "drafted_at": _now(),
            "rationale": rationale,
            "evidence": {
                "renewals": len(pairs),
                "decided_renewals": len(decided_pairs),
                "gained_referral_ids": [
                    p["submission_id"] for p in best["gained"]
                ],
                "lost_referral_ids": [p["submission_id"] for p in best["lost"]],
                "gained_decisions": _decision_split(best["gained"]),
                "lost_decisions": _decision_split(best["lost"]),
            },
            "decided_by": None,
            "decided_at": None,
            "decision_note": "",
        }
        drafted.append(proposal)
    if drafted:
        _save_proposals(store, existing + drafted)
    return {"drafted": drafted, "proposals": existing + drafted}


# ---------------------------------------------------------------------------
# Deciding a proposal — the one consequential write
# ---------------------------------------------------------------------------


def decide_proposal(
    store: Store,
    proposal_id: str,
    actor: str,
    *,
    accept: bool,
    note: str = "",
) -> dict:
    """Record a human's decision on one proposal.

    Rejecting changes only the proposal. Accepting ALSO writes the
    proposed value into the settings record the classifier reads
    (see :func:`effective_thresholds`), with the change appended to
    the record's history — proposal, dimension, from → to, actor,
    timestamp — so the threshold in force is always traceable to
    the decision that set it. A proposal can be decided once.
    """
    proposals = load_proposals(store)
    proposal = next(
        (p for p in proposals if p["proposal_id"] == proposal_id), None
    )
    if proposal is None:
        raise KeyError(f"Unknown proposal_id: {proposal_id}")
    if proposal["status"] != "draft":
        raise ValueError(
            f"Proposal {proposal_id} is already {proposal['status']} "
            f"(decided by {proposal.get('decided_by')} at "
            f"{proposal.get('decided_at')}) — a proposal is decided once"
        )
    now = _now()
    proposal["status"] = "accepted" if accept else "rejected"
    proposal["decided_by"] = actor
    proposal["decided_at"] = now
    proposal["decision_note"] = note
    if accept:
        before = effective_thresholds(store)
        after = {**before, proposal["dimension"]: proposal["proposed_value"]}
        settings = load_settings(store) or {"thresholds": before, "history": []}
        settings = {
            "thresholds": after,
            "updated_by": actor,
            "updated_at": now,
            "proposal_id": proposal_id,
            "history": [
                *(settings.get("history") or []),
                {
                    "proposal_id": proposal_id,
                    "dimension": proposal["dimension"],
                    "from": before[proposal["dimension"]],
                    "to": proposal["proposed_value"],
                    "actor": actor,
                    "at": now,
                },
            ],
        }
        store.save_document(SETTINGS_KEY, settings)
        proposal["settings_written"] = after
    _save_proposals(store, proposals)
    return proposal
