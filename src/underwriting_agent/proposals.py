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

Round 9 gives the queue's SLA budgets the same machinery. The
evidence is the wait: :func:`decided_wait_cases` replays every
*decided* record's wait (submitted → decided, in days, the same
date arithmetic the queue ages by) against its risk tier's
budget. A tighter budget drafts a proposal only when the cases it
would newly flag as breached were **all rejected** — they sat
past the tighter line and were declined, so the alarm would have
been right; a looser budget only when the cases it would
un-flag were **all approved anyway** — the alarm fired and the
wait decided nothing. Cases still awaiting a decision are not
evidence either way, and a mixed moved set drafts nothing. An
accepted SLA proposal writes its budget into a second settings
document, and the queue resolves its budgets through
:func:`effective_sla_budgets` — settings record over environment,
the same single-resolution discipline as the thresholds.

A draft proposal's evidence is a snapshot: renewals decided
since it was drafted can quietly invalidate it. So a draft
**expires** — after ``PROPOSAL_EXPIRY_DAYS`` (default 14) it can
no longer be decided. Expiry is not the end of the question,
though: :func:`refresh_proposal` re-runs the evidence scan for
that one setting against the store as it stands now. When the
evidence still carries a one-sided move, the proposal is renewed
in place (same id, fresh snapshot, fresh window); when it no
longer does, the proposal is **withdrawn** and says why. Every
refresh is appended to the proposal's ``refreshes`` audit list —
who refreshed it, when, what the stale snapshot said, and how
the re-scan came out — so a proposal's evidence is never
silently stale, and never silently replaced either.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .calibration import DEFAULT_CANDIDATES, current_thresholds, renewal_pairs
from .config import env_int
from .diffing import diff_referral_triggers
from .facts import _parse_date
from .insights import QUEUE_SLA_DEFAULTS, sla_budgets
from .ports import Store
from .store import ApprovalRecord

PROPOSALS_KEY = "calibration_proposals"
SETTINGS_KEY = "diff_trigger_thresholds"
SLA_SETTINGS_KEY = "queue_sla_budgets"

_DIMENSION_LABELS = {
    "limit_jump_pct": "limit-jump percentage",
    "new_claim_min_incurred": "new-claim incurred floor",
    "development_min_increase": "adverse-development incurred floor",
}

# Candidate SLA budgets (days) swept per tier; the budget in force
# is always added, so the current setting is a row in its own sweep.
SLA_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 10, 14]

_THRESHOLD_KIND = "diff_trigger_threshold"
_SLA_KIND = "sla_budget"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_dt(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _fmt(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Proposal expiry — a draft's evidence is a snapshot, snapshots age
# ---------------------------------------------------------------------------

#: Days a drafted proposal stays decidable before its evidence is
#: considered stale (PROPOSAL_EXPIRY_DAYS overrides per deployment).
DEFAULT_PROPOSAL_EXPIRY_DAYS = 14


def proposal_expiry_days() -> int:
    return max(1, env_int("PROPOSAL_EXPIRY_DAYS", DEFAULT_PROPOSAL_EXPIRY_DAYS))


def proposal_expires_at(proposal: dict) -> str | None:
    """When a proposal's evidence goes stale: its stamped
    ``expires_at``, or — for a draft stamped before expiry
    existed — its ``drafted_at`` plus the configured window."""
    if proposal.get("expires_at"):
        return proposal["expires_at"]
    drafted = _parse_ts(proposal.get("drafted_at"))
    if drafted is None:
        return None
    return _fmt(drafted + timedelta(days=proposal_expiry_days()))


def is_expired(proposal: dict, now: datetime | None = None) -> bool:
    """A live draft past its evidence window. Decided and
    withdrawn proposals are history — they do not expire."""
    if proposal.get("status") != "draft":
        return False
    expires = _parse_ts(proposal_expires_at(proposal))
    return expires is not None and _now_dt(now) > expires


def annotate_proposal(proposal: dict, now: datetime | None = None) -> dict:
    """A copy of one proposal with its expiry resolved for
    display: the effective ``expires_at`` and whether it has
    lapsed. Stored documents are never mutated by annotation."""
    return {
        **proposal,
        "expires_at": proposal_expires_at(proposal),
        "expired": is_expired(proposal, now),
    }


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


def load_sla_settings(store: Store) -> dict | None:
    """The accepted-budgets settings record, or None when no SLA
    proposal has ever been accepted (budgets then come from the
    environment alone)."""
    return store.get_document(SLA_SETTINGS_KEY)


def effective_sla_budgets(store: Store) -> dict[str, int]:
    """The queue SLA budgets the service runs with.

    The environment budgets (``insights.sla_budgets``) are the
    base; the SLA settings record (written only by accepting an
    SLA proposal) overrides them per tier. This is the single
    resolution the queue projection, the breach check, and the
    status-note drafting all read — they can never disagree about
    which budget a case is measured against.
    """
    base = sla_budgets()
    settings = load_sla_settings(store)
    if settings and settings.get("budgets"):
        for tier, value in settings["budgets"].items():
            if tier in base:
                base[tier] = max(0, int(value))
    return base


def sla_budget_settings_view(store: Store) -> dict:
    """The SLA budgets in force, and where they came from."""
    settings = load_sla_settings(store)
    return {
        "budgets": effective_sla_budgets(store),
        "source": "accepted_proposal" if settings else "environment",
        "environment_budgets": sla_budgets(),
        "settings": settings,
    }


def _apply_settings_change(
    store: Store,
    *,
    key: str,
    label: str,
    current_values: dict,
    dimension: str,
    value,
    proposal_id: str,
    actor: str,
    at: str,
) -> dict:
    """Write one accepted value into a settings document.

    The document keeps only the values in force plus its audit
    history — one entry per change: proposal, dimension,
    from → to, who, when. Returns the values now in force.
    """
    document = store.get_document(key) or {}
    after = {**current_values, dimension: value}
    history = [
        *(document.get("history") or []),
        {
            "proposal_id": proposal_id,
            "dimension": dimension,
            "from": current_values[dimension],
            "to": value,
            "actor": actor,
            "at": at,
        },
    ]
    store.save_document(
        key,
        {
            label: after,
            "updated_by": actor,
            "updated_at": at,
            "proposal_id": proposal_id,
            "history": history,
        },
    )
    return after


def _revert_settings_change(
    store: Store,
    *,
    key: str,
    label: str,
    current_values: dict,
    actor: str,
    now: datetime | None,
) -> dict:
    """Undo a settings record's most recent change.

    The prior value is restored as a **new** history entry — the
    record is append-only, so a revert never rewrites the history
    it undoes. The entry names the proposal whose change it
    reverses (``revert: True``); reverting a revert re-applies
    the change it undid, which is the same rule one level down
    and needs no special case. Returns the updated document.
    """
    document = store.get_document(key)
    history = list((document or {}).get("history") or [])
    if not document or not history:
        raise ValueError(
            f"Nothing to revert — the {label} settings record has no "
            "accepted change in its history"
        )
    last = history[-1]
    at = _fmt(_now_dt(now))
    if last.get("changes"):
        # A multi-dimension entry (a point-in-history restore, or
        # a revert of one): undo every dimension it moved, as one
        # revert entry carrying the same changes shape.
        restored = dict(current_values)
        changes = []
        for change in last["changes"]:
            dimension = change["dimension"]
            restored[dimension] = change["from"]
            changes.append(
                {
                    "dimension": dimension,
                    "from": current_values[dimension],
                    "to": change["from"],
                }
            )
        entry = {
            "proposal_id": last.get("proposal_id"),
            "changes": changes,
            "actor": actor,
            "at": at,
            "revert": True,
        }
        if last.get("restore"):
            entry["reverts_restore"] = True
            entry["restores_index"] = last.get("restores_index")
    else:
        dimension = last["dimension"]
        restored = {**current_values, dimension: last["from"]}
        entry = {
            "proposal_id": last["proposal_id"],
            "dimension": dimension,
            "from": current_values[dimension],
            "to": last["from"],
            "actor": actor,
            "at": at,
            "revert": True,
        }
    updated = {
        label: restored,
        "updated_by": actor,
        "updated_at": at,
        "proposal_id": last["proposal_id"],
        "history": [*history, entry],
    }
    store.save_document(key, updated)
    return updated


def revert_threshold_settings(
    store: Store, actor: str, now: datetime | None = None
) -> dict:
    """Revert the threshold settings record's latest accepted
    change — the recorded way back when a threshold change
    misbehaves in production, instead of editing the database."""
    return _revert_settings_change(
        store,
        key=SETTINGS_KEY,
        label="thresholds",
        current_values=effective_thresholds(store),
        actor=actor,
        now=now,
    )


def revert_sla_budget_settings(
    store: Store, actor: str, now: datetime | None = None
) -> dict:
    """Revert the SLA-budget settings record's latest accepted
    change, the same append-only way."""
    return _revert_settings_change(
        store,
        key=SLA_SETTINGS_KEY,
        label="budgets",
        current_values=effective_sla_budgets(store),
        actor=actor,
        now=now,
    )


def _values_as_of(base: dict, history: list[dict], index: int, cast) -> dict:
    """The values in force immediately after history entry
    ``index`` was applied: the environment base replayed through
    every entry up to and including it. Entries are either flat
    (one dimension: an accept or a revert) or carry a ``changes``
    list (a restore, or a revert of one) — the replay reads both.
    """
    values = dict(base)
    for entry in history[: index + 1]:
        if entry.get("changes"):
            for change in entry["changes"]:
                if change["dimension"] in values:
                    values[change["dimension"]] = cast(change["to"])
        elif entry.get("dimension") in values:
            values[entry["dimension"]] = cast(entry["to"])
    return values


def _restore_settings_to_entry(
    store: Store,
    *,
    key: str,
    label: str,
    base_values: dict,
    current_values: dict,
    history_index: int,
    actor: str,
    now: datetime | None,
    cast,
) -> dict:
    """Restore a settings record to a chosen history entry's state.

    Revert walks one step back; a bad *sequence* of accepts needs
    the whole record returned to an earlier state. The values as
    of the chosen entry are replayed from the environment base
    through the history (see :func:`_values_as_of`), and the
    restore is appended as **one** new history entry naming the
    entry it restores to — its index, and the proposal that entry
    carried — with every dimension it moves listed in ``changes``
    (from the value in force to the restored value). History is
    never rewritten: the mis-tuned afternoon stays on the record,
    and the restore itself can be reverted like any other entry.
    Returns the updated document.
    """
    document = store.get_document(key)
    history = list((document or {}).get("history") or [])
    if not document or not history:
        raise ValueError(
            f"Nothing to restore — the {label} settings record has no "
            "accepted change in its history"
        )
    if not 0 <= history_index < len(history):
        raise ValueError(
            f"History index {history_index} is out of range — the "
            f"{label} settings record has {len(history)} history "
            f"entries (0 to {len(history) - 1})"
        )
    target = _values_as_of(base_values, history, history_index, cast)
    changes = [
        {"dimension": dimension, "from": current_values[dimension], "to": target[dimension]}
        for dimension in target
        if current_values.get(dimension) != target[dimension]
    ]
    if not changes:
        raise ValueError(
            f"The {label} settings already match history entry "
            f"{history_index} — restoring it would change nothing"
        )
    restored_entry = history[history_index]
    at = _fmt(_now_dt(now))
    entry = {
        "proposal_id": restored_entry.get("proposal_id"),
        "restore": True,
        "restores_index": history_index,
        "restores_proposal_id": restored_entry.get("proposal_id"),
        "changes": changes,
        "actor": actor,
        "at": at,
    }
    updated = {
        label: target,
        "updated_by": actor,
        "updated_at": at,
        "proposal_id": restored_entry.get("proposal_id"),
        "history": [*history, entry],
    }
    store.save_document(key, updated)
    return updated


def restore_threshold_settings(
    store: Store,
    history_index: int,
    actor: str,
    now: datetime | None = None,
) -> dict:
    """Restore the threshold settings record to the state in force
    after its history entry ``history_index`` — one audited change
    naming the entry restored to, however many dimensions move."""
    return _restore_settings_to_entry(
        store,
        key=SETTINGS_KEY,
        label="thresholds",
        base_values=current_thresholds(),
        current_values=effective_thresholds(store),
        history_index=history_index,
        actor=actor,
        now=now,
        cast=float,
    )


def restore_sla_budget_settings(
    store: Store,
    history_index: int,
    actor: str,
    now: datetime | None = None,
) -> dict:
    """Restore the SLA-budget settings record to a chosen history
    entry's state, the same append-only way."""
    return _restore_settings_to_entry(
        store,
        key=SLA_SETTINGS_KEY,
        label="budgets",
        base_values=sla_budgets(),
        current_values=effective_sla_budgets(store),
        history_index=history_index,
        actor=actor,
        now=now,
        cast=lambda v: max(0, int(v)),
    )


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


def _proposal_kind(proposal: dict) -> str:
    """A stored proposal's kind (round-8 proposals all carry one;
    anything older is a threshold proposal by definition)."""
    return proposal.get("kind") or _THRESHOLD_KIND


def _rejected_directions(existing: list[dict], kind: str) -> set:
    """The (dimension, direction) pairs of one kind a human has
    already rejected — the scanner never re-asks a direction."""
    return {
        (p["dimension"], p["direction"])
        for p in existing
        if p["status"] == "rejected" and _proposal_kind(p) == kind
    }


def _best_threshold_candidate(
    pairs: list[dict],
    dimension: str,
    current: dict,
    rejected_directions: set,
) -> dict | None:
    """The best one-sided decided move for one threshold dimension.

    Sweeps the candidate values; a value qualifies when the
    referrals it moves are entirely decided and one-sided (lower:
    gained referrals all rejected; raise: lost referrals all
    approved). Most moved cases wins; ties go to the smallest
    change. None when no value qualifies.
    """
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
    return best


def _threshold_fields(
    dimension: str,
    current: dict,
    best: dict,
    pairs: list[dict],
    decided_pairs: list[dict],
) -> dict:
    """The proposal fields for one threshold candidate — shared by
    drafting and by the evidence refresh, so a refreshed proposal
    is built exactly the way a fresh one is."""
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
    return {
        "kind": _THRESHOLD_KIND,
        "dimension": dimension,
        "current_value": current[dimension],
        "proposed_value": best["value"],
        "direction": best["direction"],
        "rationale": rationale,
        "evidence": {
            "renewals": len(pairs),
            "decided_renewals": len(decided_pairs),
            "gained_referral_ids": [p["submission_id"] for p in best["gained"]],
            "lost_referral_ids": [p["submission_id"] for p in best["lost"]],
            "gained_decisions": _decision_split(best["gained"]),
            "lost_decisions": _decision_split(best["lost"]),
        },
    }


def draft_proposals(
    store: Store, records: list[ApprovalRecord], now: datetime | None = None
) -> dict:
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
    open_dimensions = {
        p["dimension"]
        for p in existing
        if p["status"] == "draft" and _proposal_kind(p) == _THRESHOLD_KIND
    }
    rejected_directions = _rejected_directions(existing, _THRESHOLD_KIND)
    drafted: list[dict] = []
    decided_pairs = [p for p in pairs if p["status"] in ("approved", "rejected")]

    for dimension in (
        "limit_jump_pct",
        "new_claim_min_incurred",
        "development_min_increase",
    ):
        if dimension in open_dimensions:
            continue
        best = _best_threshold_candidate(
            pairs, dimension, current, rejected_directions
        )
        if best is None:
            continue
        drafted_dt = _now_dt(now)
        proposal = {
            "proposal_id": f"PROP-{len(existing) + len(drafted) + 1:04d}",
            **_threshold_fields(dimension, current, best, pairs, decided_pairs),
            "status": "draft",
            "drafted_at": _fmt(drafted_dt),
            "expires_at": _fmt(
                drafted_dt + timedelta(days=proposal_expiry_days())
            ),
            "decided_by": None,
            "decided_at": None,
            "decision_note": "",
        }
        drafted.append(proposal)
    if drafted:
        _save_proposals(store, existing + drafted)
    return {"drafted": drafted, "proposals": existing + drafted}


# ---------------------------------------------------------------------------
# SLA-budget proposals — the wait is the evidence
# ---------------------------------------------------------------------------


def decided_wait_cases(records: list[ApprovalRecord]) -> list[dict]:
    """Every decided record as SLA evidence: how long it waited.

    The wait is whole days from the submission's ``submitted_at``
    to the decision's ``decided_at`` — the same date arithmetic
    the queue ages awaiting cases by. Records still awaiting a
    decision, and decided records whose dates cannot be read, are
    not evidence and are left out (the scan says how many decided
    cases it weighed, so the exclusion is visible, not silent).
    """
    cases: list[dict] = []
    for record in records:
        result = record.result
        if result.status not in ("approved", "rejected"):
            continue
        submitted = _parse_date((record.submission or {}).get("submitted_at", ""))
        decided = _parse_date(result.decided_at or "")
        if submitted is None or decided is None:
            continue
        cases.append(
            {
                "submission_id": result.submission_id,
                "insured_name": result.insured_name
                or (record.submission or {}).get("insured_name", ""),
                "risk_tier": result.triage.risk_tier,
                "status": result.status,
                "wait_days": max(0, (decided - submitted).days),
            }
        )
    return cases


def _best_sla_candidate(
    cases: list[dict],
    tier: str,
    current_budget: int,
    rejected_directions: set,
) -> dict | None:
    """The best one-sided decided budget move for one risk tier.

    A case *breached* its budget when it waited longer than the
    budget (the queue's own rule). Lowering the budget moves the
    cases it would newly flag — those qualify only when
    underwriters rejected **every** one of them: they sat past the
    tighter line and were declined. Raising it moves the cases it
    would un-flag — those qualify only when underwriters approved
    **every** one anyway: the breach alarm fired and the wait
    decided nothing. Most moved cases wins; ties go to the
    smallest change.
    """
    tier_cases = [c for c in cases if c["risk_tier"] == tier]
    best: dict | None = None
    for value in sorted(set(SLA_CANDIDATES) | {current_budget}):
        if value == current_budget:
            continue
        direction = "lower" if value < current_budget else "raise"
        if (tier, direction) in rejected_directions:
            continue
        if direction == "lower":
            moved = [
                c for c in tier_cases if value < c["wait_days"] <= current_budget
            ]
            if not moved or any(c["status"] != "rejected" for c in moved):
                continue
        else:
            moved = [
                c for c in tier_cases if current_budget < c["wait_days"] <= value
            ]
            if not moved or any(c["status"] != "approved" for c in moved):
                continue
        candidate = {
            "value": value,
            "direction": direction,
            "moved": moved,
            "split": _decision_split(moved),
        }
        if (
            best is None
            or len(moved) > len(best["moved"])
            or (
                len(moved) == len(best["moved"])
                and abs(value - current_budget) < abs(best["value"] - current_budget)
            )
        ):
            best = candidate
    return best


def _sla_fields(
    tier: str, current_budget: int, best: dict, tier_cases: list[dict]
) -> dict:
    """The proposal fields for one SLA-budget candidate — shared
    by drafting and by the evidence refresh."""
    moved_ids = [c["submission_id"] for c in best["moved"]]
    if best["direction"] == "lower":
        rationale = (
            f"Lowering the {tier} SLA budget from {current_budget} to "
            f"{best['value']} day(s) would have flagged "
            f"{len(best['moved'])} decided case(s) "
            f"({', '.join(moved_ids)}) as breached while they waited; "
            "underwriters rejected every one of them, and the tighter "
            "budget flags no case they approved. Cases still awaiting "
            "a decision are not counted as evidence."
        )
    else:
        rationale = (
            f"Raising the {tier} SLA budget from {current_budget} to "
            f"{best['value']} day(s) would have kept "
            f"{len(best['moved'])} decided case(s) "
            f"({', '.join(moved_ids)}) within budget through their "
            "wait; underwriters approved every one of them anyway, and "
            "the looser budget un-flags no case they rejected. Cases "
            "still awaiting a decision are not counted as evidence."
        )
    return {
        "kind": _SLA_KIND,
        "dimension": tier,
        "current_value": current_budget,
        "proposed_value": best["value"],
        "direction": best["direction"],
        "rationale": rationale,
        "evidence": {
            "tier": tier,
            "decided_cases_in_tier": len(tier_cases),
            "moved_ids": moved_ids,
            "moved_decisions": best["split"],
            "waits": {c["submission_id"]: c["wait_days"] for c in best["moved"]},
        },
    }


def draft_sla_proposals(
    store: Store, records: list[ApprovalRecord], now: datetime | None = None
) -> dict:
    """Scan the decided cases' waits for proposal-worthy SLA moves.

    The same drafting discipline as the threshold scan: one open
    question per knob (a tier with a live draft SLA proposal is
    left alone), a rejected direction is never re-proposed in
    another size, and only one-sided decided evidence drafts
    anything. SLA proposals share the proposals document with the
    threshold proposals — one list, one decision path, one audit.
    Returns the newly drafted proposals plus the full stored list.
    """
    cases = decided_wait_cases(records)
    budgets = effective_sla_budgets(store)
    existing = load_proposals(store)
    open_tiers = {
        p["dimension"]
        for p in existing
        if p["status"] == "draft" and _proposal_kind(p) == _SLA_KIND
    }
    rejected_directions = _rejected_directions(existing, _SLA_KIND)
    drafted: list[dict] = []
    for tier in QUEUE_SLA_DEFAULTS:
        if tier in open_tiers:
            continue
        tier_cases = [c for c in cases if c["risk_tier"] == tier]
        if not tier_cases:
            continue
        best = _best_sla_candidate(
            cases, tier, budgets[tier], rejected_directions
        )
        if best is None:
            continue
        drafted_dt = _now_dt(now)
        proposal = {
            "proposal_id": f"PROP-{len(existing) + len(drafted) + 1:04d}",
            **_sla_fields(tier, budgets[tier], best, tier_cases),
            "status": "draft",
            "drafted_at": _fmt(drafted_dt),
            "expires_at": _fmt(
                drafted_dt + timedelta(days=proposal_expiry_days())
            ),
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
    now: datetime | None = None,
) -> dict:
    """Record a human's decision on one proposal.

    Rejecting changes only the proposal. Accepting ALSO writes the
    proposed value into the settings record for the proposal's
    kind — the diff-trigger thresholds the classifier reads (see
    :func:`effective_thresholds`), or the queue SLA budgets (see
    :func:`effective_sla_budgets`) — with the change appended to
    the record's history — proposal, dimension, from → to, actor,
    timestamp — so the value in force is always traceable to the
    decision that set it. A proposal can be decided once.
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
    if is_expired(proposal, now):
        raise ValueError(
            f"Proposal {proposal_id} expired at "
            f"{proposal_expires_at(proposal)} — its evidence is a stale "
            "snapshot. Refresh it to re-run the evidence scan before "
            "deciding"
        )
    now = _fmt(_now_dt(now))
    proposal["status"] = "accepted" if accept else "rejected"
    proposal["decided_by"] = actor
    proposal["decided_at"] = now
    proposal["decision_note"] = note
    if accept:
        if _proposal_kind(proposal) == _SLA_KIND:
            after = _apply_settings_change(
                store,
                key=SLA_SETTINGS_KEY,
                label="budgets",
                current_values=effective_sla_budgets(store),
                dimension=proposal["dimension"],
                value=int(proposal["proposed_value"]),
                proposal_id=proposal_id,
                actor=actor,
                at=now,
            )
        else:
            after = _apply_settings_change(
                store,
                key=SETTINGS_KEY,
                label="thresholds",
                current_values=effective_thresholds(store),
                dimension=proposal["dimension"],
                value=proposal["proposed_value"],
                proposal_id=proposal_id,
                actor=actor,
                at=now,
            )
        proposal["settings_written"] = after
    _save_proposals(store, proposals)
    return proposal


# ---------------------------------------------------------------------------
# Refreshing an expired proposal — re-run its evidence scan
# ---------------------------------------------------------------------------


def refresh_proposal(
    store: Store,
    records: list[ApprovalRecord],
    proposal_id: str,
    actor: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Re-run one expired draft's evidence scan against the store
    as it stands now.

    The scan is the drafting scan, restricted to the proposal's
    own setting and re-based on the values in force today (which
    may themselves have moved since the draft). Two outcomes:

    - **Renewed** — the evidence still carries a one-sided decided
      move: the proposal keeps its id and takes the fresh snapshot
      (values, rationale, evidence, a new drafted-at and a new
      expiry window). The move it proposes may differ from the
      stale one — it is whatever today's evidence supports.
    - **Withdrawn** — no one-sided move remains: the proposal
      leaves the live list with the reason recorded, instead of
      waiting to be decided on evidence that no longer holds.

    Either way the refresh is appended to the proposal's
    ``refreshes`` audit list with the stale snapshot it replaced.
    Only an expired draft can be refreshed — a live draft's
    evidence is current by definition, and a decided or withdrawn
    proposal is history.
    """
    proposals = load_proposals(store)
    proposal = next(
        (p for p in proposals if p["proposal_id"] == proposal_id), None
    )
    if proposal is None:
        raise KeyError(f"Unknown proposal_id: {proposal_id}")
    if proposal["status"] != "draft":
        raise ValueError(
            f"Proposal {proposal_id} is {proposal['status']} — only a "
            "live draft can be refreshed"
        )
    now_dt = _now_dt(now)
    if not is_expired(proposal, now_dt):
        raise ValueError(
            f"Proposal {proposal_id} is still live until "
            f"{proposal_expires_at(proposal)} — refresh applies once "
            "a draft has expired"
        )
    kind = _proposal_kind(proposal)
    rejected = _rejected_directions(proposals, kind)
    fields: dict | None = None
    if kind == _SLA_KIND:
        budgets = effective_sla_budgets(store)
        tier = proposal["dimension"]
        cases = decided_wait_cases(records)
        tier_cases = [c for c in cases if c["risk_tier"] == tier]
        best = _best_sla_candidate(cases, tier, budgets[tier], rejected)
        if best is not None:
            fields = _sla_fields(tier, budgets[tier], best, tier_cases)
    else:
        current = effective_thresholds(store)
        pairs = renewal_pairs(records)
        decided_pairs = [
            p for p in pairs if p["status"] in ("approved", "rejected")
        ]
        best = _best_threshold_candidate(
            pairs, proposal["dimension"], current, rejected
        )
        if best is not None:
            fields = _threshold_fields(
                proposal["dimension"], current, best, pairs, decided_pairs
            )
    entry = {
        "at": _fmt(now_dt),
        "by": actor,
        "previous": {
            "current_value": proposal.get("current_value"),
            "proposed_value": proposal.get("proposed_value"),
            "direction": proposal.get("direction"),
            "drafted_at": proposal.get("drafted_at"),
            "expires_at": proposal_expires_at(proposal),
        },
    }
    if fields is None:
        proposal["status"] = "withdrawn"
        proposal["withdrawn_at"] = _fmt(now_dt)
        proposal["withdrawn_by"] = actor
        proposal["withdrawal_reason"] = (
            "the evidence refresh found no one-sided decided move "
            "for this setting — the snapshot it was drafted from no "
            "longer holds"
        )
        entry["outcome"] = "withdrawn"
    else:
        proposal.update(fields)
        proposal["drafted_at"] = _fmt(now_dt)
        proposal["expires_at"] = _fmt(
            now_dt + timedelta(days=proposal_expiry_days())
        )
        entry["outcome"] = "renewed"
    proposal.setdefault("refreshes", []).append(entry)
    _save_proposals(store, proposals)
    return proposal


def refresh_expired_proposals(
    store: Store,
    records: list[ApprovalRecord],
    actor: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Refresh every expired draft in one operator call.

    The sweep is a loop, not a shortcut: each expired draft goes
    through :func:`refresh_proposal` on its own — its own evidence
    scan re-run against the store as it stands, its own refresh
    entry appended to its own audit list, renewed or withdrawn on
    its own evidence. Nothing is bulk-decided: a renewed draft
    still waits for a human decision, and a draft whose evidence
    no longer holds is withdrawn one by one, with its reason.
    Live drafts are untouched (their evidence is current by
    definition) and decided or withdrawn proposals are history.

    Returns the sweep report: the renewed drafts and the
    withdrawn ones, side by side, each annotated with its new
    expiry state.
    """
    now_dt = _now_dt(now)
    expired_ids = [
        p["proposal_id"]
        for p in load_proposals(store)
        if is_expired(p, now_dt)
    ]
    renewed: list[dict] = []
    withdrawn: list[dict] = []
    for proposal_id in expired_ids:
        refreshed = refresh_proposal(store, records, proposal_id, actor, now=now_dt)
        target = renewed if refreshed["status"] == "draft" else withdrawn
        target.append(annotate_proposal(refreshed, now_dt))
    return {
        "refreshed_at": _fmt(now_dt),
        "by": actor,
        "renewed": renewed,
        "withdrawn": withdrawn,
        "counts": {
            "swept": len(expired_ids),
            "renewed": len(renewed),
            "withdrawn": len(withdrawn),
        },
    }
