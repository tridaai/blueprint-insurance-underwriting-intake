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

Round 13 closes the calibration loop's remaining gap on the
report side: the SLA calibration report flags ``proposal_ready``
rows, but drafting used to mean running the whole scan and
accepting its best row per tier.
:func:`draft_sla_proposal_from_row` drafts a proposal for one
operator-chosen (tier, value) row instead, carrying the row's
evidence verbatim — the report is no longer read-only. The
report itself can also be kept:
:func:`build_calibration_snapshot` stores it dated and under
the taker's name (like the digest snapshots), and
:func:`calibration_snapshot_diff` reads two kept reports'
proposal-ready rows against each other, so a budget debate can
cite the evidence as of a date.
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


def _restore_plan(
    store: Store,
    *,
    key: str,
    label: str,
    base_values: dict,
    current_values: dict,
    history_index: int,
    cast,
) -> dict:
    """The dry run of a restore: the values in force as of the
    chosen history entry, and exactly which dimensions a restore
    to it would move — computed without writing anything. The
    restore itself and its preview share this plan, so a preview
    cannot promise a different restore than the one that lands.
    Raises the same no-history / out-of-range errors the restore
    raises; a restore that would move nothing is NOT an error
    here — ``would_change`` says so, and the preview exists
    precisely to surface that before anyone commits to it.
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
    return {
        "label": label,
        "history_index": history_index,
        "restores_proposal_id": history[history_index].get("proposal_id"),
        "current_values": dict(current_values),
        "target_values": target,
        "changes": changes,
        "would_change": bool(changes),
        "history": history,
    }


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
    note: str = "",
) -> dict:
    """Restore a settings record to a chosen history entry's state.

    Revert walks one step back; a bad *sequence* of accepts needs
    the whole record returned to an earlier state. The values as
    of the chosen entry are replayed from the environment base
    through the history (see :func:`_values_as_of`), and the
    restore is appended as **one** new history entry naming the
    entry it restores to — its index, and the proposal that entry
    carried — with every dimension it moves listed in ``changes``
    (from the value in force to the restored value). The entry
    also carries the restorer's ``decision_note`` when they gave
    one: the record already answers what moved, who moved it,
    and when — the note is where *why* lives, so "we reverted
    after the mis-tuned afternoon" stops being archaeology.
    History is never rewritten: the mis-tuned afternoon stays on
    the record, and the restore itself can be reverted like any
    other entry. Returns the updated document.
    """
    plan = _restore_plan(
        store,
        key=key,
        label=label,
        base_values=base_values,
        current_values=current_values,
        history_index=history_index,
        cast=cast,
    )
    changes = plan["changes"]
    if not changes:
        raise ValueError(
            f"The {label} settings already match history entry "
            f"{history_index} — restoring it would change nothing"
        )
    target = plan["target_values"]
    restored_entry = plan["history"][history_index]
    at = _fmt(_now_dt(now))
    entry = {
        "proposal_id": restored_entry.get("proposal_id"),
        "restore": True,
        "restores_index": history_index,
        "restores_proposal_id": restored_entry.get("proposal_id"),
        "changes": changes,
        "actor": actor,
        "at": at,
        "decision_note": note,
    }
    updated = {
        label: target,
        "updated_by": actor,
        "updated_at": at,
        "proposal_id": restored_entry.get("proposal_id"),
        "history": [*plan["history"], entry],
    }
    store.save_document(key, updated)
    return updated


def _preview_restore(
    store: Store,
    *,
    key: str,
    label: str,
    base_values: dict,
    current_values: dict,
    history_index: int,
    cast,
    note: str = "",
) -> dict:
    """A restore preview: the shared plan (see
    :func:`_restore_plan`) with the internal replay state left
    out — what an operator reads before committing to a restore.
    Nothing is written, and nothing is appended: previewing a
    restore is not itself a change on the record. The preview is
    also where a restore's decision note is drafted: a ``note``
    passed here is echoed back as ``decision_note`` so the
    operator can word the "why" against the exact moves it will
    be stored with — the preview itself still stores nothing."""
    plan = _restore_plan(
        store,
        key=key,
        label=label,
        base_values=base_values,
        current_values=current_values,
        history_index=history_index,
        cast=cast,
    )
    preview = {
        field: plan[field]
        for field in (
            "label",
            "history_index",
            "restores_proposal_id",
            "current_values",
            "target_values",
            "changes",
            "would_change",
        )
    }
    preview["decision_note"] = note
    return preview


def preview_restore_threshold_settings(
    store: Store, history_index: int, note: str = ""
) -> dict:
    """Preview restoring the threshold settings record to a
    chosen history entry's state: which dimensions would move,
    from what to what — before anything is written. A drafted
    decision note is echoed back, stored nowhere."""
    return _preview_restore(
        store,
        key=SETTINGS_KEY,
        label="thresholds",
        base_values=current_thresholds(),
        current_values=effective_thresholds(store),
        history_index=history_index,
        cast=float,
        note=note,
    )


def preview_restore_sla_budget_settings(
    store: Store, history_index: int, note: str = ""
) -> dict:
    """Preview restoring the SLA-budget settings record to a
    chosen history entry's state, the same dry-run way."""
    return _preview_restore(
        store,
        key=SLA_SETTINGS_KEY,
        label="budgets",
        base_values=sla_budgets(),
        current_values=effective_sla_budgets(store),
        history_index=history_index,
        cast=lambda v: max(0, int(v)),
        note=note,
    )


def restore_threshold_settings(
    store: Store,
    history_index: int,
    actor: str,
    now: datetime | None = None,
    note: str = "",
) -> dict:
    """Restore the threshold settings record to the state in force
    after its history entry ``history_index`` — one audited change
    naming the entry restored to, however many dimensions move.
    The restorer's optional ``note`` is kept on the restore entry
    as its ``decision_note``."""
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
        note=note,
    )


def restore_sla_budget_settings(
    store: Store,
    history_index: int,
    actor: str,
    now: datetime | None = None,
    note: str = "",
) -> dict:
    """Restore the SLA-budget settings record to a chosen history
    entry's state, the same append-only way, with the same
    optional decision note on the restore entry."""
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
        note=note,
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
            "drafted_from": "scan",
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
            "drafted_from": "scan",
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


def draft_sla_proposal_from_row(
    store: Store,
    records: list[ApprovalRecord],
    tier: str,
    value: int,
    actor: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Draft one SLA-budget proposal for a chosen calibration row.

    The calibration report (:func:`sla_budget_calibration`) flags
    the rows whose moved set is one-sided ``proposal_ready``
    evidence — but until now the only way to act on one was the
    scan (:func:`draft_sla_proposals`), which drafts its own best
    row per tier and nothing else. An underwriter reading the
    report who wants a *different* ready row on the table — the
    smaller move, the one whose cases they remember — had no way
    to ask for it. This is the report's counterpart write: the
    chosen (tier, value) row is recomputed against the store as
    it stands and, when it is proposal-ready, drafted as a
    proposal whose rationale and evidence are the row's — the same
    moved set, the same decision splits, the same breach counts
    the report showed, kept on the proposal under
    ``evidence.calibration_row`` so the report's numbers and the
    proposal's numbers can never drift apart. The proposal is
    stamped ``drafted_from: "calibration_row"`` and
    ``drafted_by`` the operator, so a row-drafted proposal is
    distinguishable from a scan-drafted one on the record.

    The drafting discipline is the scan's, unchanged: the row
    must be proposal-ready (a moved set, one-sided — otherwise
    ValueError naming why not, and nothing is drafted); the value
    must be one of the report's swept candidates (``SLA_CANDIDATES``
    or the budget in force) — the row an operator drafts from is
    a row the report actually shows; a tier with a live draft SLA
    proposal is one open question, so a second row for that tier
    waits for the first to be decided; and a direction a human
    has already rejected on that tier is never re-asked in
    another size. Drafting writes only the proposal document — a
    human still decides, and accepting writes the budget exactly
    the way a scan-drafted proposal's acceptance does.
    """
    if tier not in QUEUE_SLA_DEFAULTS:
        raise ValueError(
            f"Unknown risk tier: {tier!r} — the SLA budgets know "
            f"{', '.join(QUEUE_SLA_DEFAULTS)}"
        )
    budgets = effective_sla_budgets(store)
    current_budget = budgets[tier]
    value = int(value)
    if value == current_budget:
        raise ValueError(
            f"{value} day(s) is already the {tier} SLA budget in "
            "force — the calibration row for the current budget "
            "moves nothing by definition"
        )
    if value not in set(SLA_CANDIDATES):
        raise ValueError(
            f"{value} day(s) is not one of the calibration report's "
            f"swept candidates for the {tier} tier "
            f"({', '.join(str(v) for v in SLA_CANDIDATES)}) — a "
            "proposal is drafted from a row the report shows"
        )
    existing = load_proposals(store)
    open_proposal = next(
        (
            p
            for p in existing
            if p["status"] == "draft"
            and _proposal_kind(p) == _SLA_KIND
            and p["dimension"] == tier
        ),
        None,
    )
    if open_proposal is not None:
        raise ValueError(
            f"Proposal {open_proposal['proposal_id']} already asks "
            f"the {tier} SLA question (a move to "
            f"{open_proposal['proposed_value']} day(s)) — one open "
            "question per tier; decide it before drafting another row"
        )
    direction = "lower" if value < current_budget else "raise"
    if (tier, direction) in _rejected_directions(existing, _SLA_KIND):
        raise ValueError(
            f"A {direction} move on the {tier} SLA budget was "
            "already rejected by a human — the scanner does not "
            "re-ask a rejected direction in another size, and "
            "neither does a row draft"
        )
    cases = decided_wait_cases(records)
    tier_cases = [c for c in cases if c["risk_tier"] == tier]
    row = _sla_sweep_row(tier_cases, value, current_budget)
    if not row["proposal_ready"]:
        if not row["moved_vs_current"]:
            raise ValueError(
                f"The {tier} row at {value} day(s) moves no decided "
                "case against the budget in force — there is no "
                "evidence to draft a proposal from"
            )
        raise ValueError(
            f"The {tier} row at {value} day(s) is not "
            "proposal-ready: its moved set is not one-sided "
            f"decided evidence ({row['moved_decisions']['approved']} "
            f"approved, {row['moved_decisions']['rejected']} "
            "rejected) — a proposal is drafted only from a move "
            "whose moved cases all point the same way"
        )
    by_id = {c["submission_id"]: c for c in tier_cases}
    best = {
        "value": value,
        "direction": direction,
        "moved": [by_id[e["submission_id"]] for e in row["moved_vs_current"]],
        "split": _decision_split(
            [by_id[e["submission_id"]] for e in row["moved_vs_current"]]
        ),
    }
    fields = _sla_fields(tier, current_budget, best, tier_cases)
    fields["evidence"]["calibration_row"] = {
        "budget_in_force": current_budget,
        "breached": row["breached"],
        "breached_decisions": row["breached_decisions"],
        "within_decisions": row["within_decisions"],
    }
    fields["rationale"] += (
        f" Drafted from the calibration report's {tier} row at "
        f"{value} day(s) by {actor} — the proposal carries that "
        "row's evidence."
    )
    drafted_dt = _now_dt(now)
    proposal = {
        "proposal_id": f"PROP-{len(existing) + 1:04d}",
        **fields,
        "status": "draft",
        "drafted_from": "calibration_row",
        "drafted_by": actor,
        "drafted_at": _fmt(drafted_dt),
        "expires_at": _fmt(
            drafted_dt + timedelta(days=proposal_expiry_days())
        ),
        "decided_by": None,
        "decided_at": None,
        "decision_note": "",
    }
    _save_proposals(store, existing + [proposal])
    return proposal


# ---------------------------------------------------------------------------
# SLA-budget calibration — the budgets' what-if, like the thresholds'
# ---------------------------------------------------------------------------


def sla_budget_sources(store: Store) -> dict[str, str]:
    """Where each tier's budget in force came from: the settings
    record (an accepted proposal wrote it) or the environment.

    Per tier: the settings document stores the whole budgets map
    once any change lands, so presence in the map proves nothing —
    the provenance is the history. A tier's budget traces to the
    record when some history entry moved it (an accept, a revert,
    or a restore's ``changes``); a tier no entry ever touched
    still reads the environment value. A record without history
    (hand-written) falls back to comparing values against the
    environment base.
    """
    settings = load_sla_settings(store)
    if not settings:
        return {tier: "environment" for tier in QUEUE_SLA_DEFAULTS}
    history = settings.get("history") or []
    if history:
        moved: set[str] = set()
        for entry in history:
            if entry.get("changes"):
                moved.update(c["dimension"] for c in entry["changes"])
            elif entry.get("dimension"):
                moved.add(entry["dimension"])
        return {
            tier: (
                "accepted_proposal" if tier in moved else "environment"
            )
            for tier in QUEUE_SLA_DEFAULTS
        }
    base = sla_budgets()
    current = settings.get("budgets") or {}
    return {
        tier: (
            "accepted_proposal"
            if tier in current and int(current[tier]) != base[tier]
            else "environment"
        )
        for tier in QUEUE_SLA_DEFAULTS
    }


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    """Nearest-rank percentile of an ascending list (None when
    empty): the smallest value at least ``pct`` percent of the
    waits sit at or below."""
    if not sorted_values:
        return None
    rank = max(1, (len(sorted_values) * pct + 99) // 100)
    return sorted_values[rank - 1]


def _sla_sweep_row(
    tier_cases: list[dict], value: int, current_budget: int
) -> dict:
    """One candidate budget read against a tier's decided waits.

    ``breached`` applies the queue's own rule (a case breached
    when it waited longer than the budget). ``moved_vs_current``
    is the proposal evidence: the decided cases whose breach
    state flips between the budget in force and this candidate —
    newly flagged when the candidate is tighter, un-flagged when
    it is looser — with their decisions, and whether that moved
    set is **one-sided** the way :func:`_best_sla_candidate`
    requires before a proposal may be drafted from it.
    """
    breached = [c for c in tier_cases if c["wait_days"] > value]
    within = [c for c in tier_cases if c["wait_days"] <= value]
    if value < current_budget:
        direction: str | None = "lower"
        moved = [
            c for c in tier_cases if value < c["wait_days"] <= current_budget
        ]
    elif value > current_budget:
        direction = "raise"
        moved = [
            c for c in tier_cases if current_budget < c["wait_days"] <= value
        ]
    else:
        direction = None
        moved = []
    moved_split = _decision_split(moved)
    if not moved:
        one_sided: bool | None = None
    elif direction == "lower":
        one_sided = moved_split["rejected"] == len(moved)
    else:
        one_sided = moved_split["approved"] == len(moved)
    return {
        "value": value,
        "is_current": value == current_budget,
        "direction": direction,
        "breached": len(breached),
        "breached_decisions": {
            "approved": sum(1 for c in breached if c["status"] == "approved"),
            "rejected": sum(1 for c in breached if c["status"] == "rejected"),
        },
        "within_decisions": {
            "approved": sum(1 for c in within if c["status"] == "approved"),
            "rejected": sum(1 for c in within if c["status"] == "rejected"),
        },
        "moved_vs_current": [
            {
                "submission_id": c["submission_id"],
                "wait_days": c["wait_days"],
                "decision": c["status"],
            }
            for c in sorted(moved, key=lambda c: c["submission_id"])
        ],
        "moved_decisions": {
            "approved": moved_split["approved"],
            "rejected": moved_split["rejected"],
        },
        "one_sided": one_sided,
        "proposal_ready": bool(moved) and one_sided is True,
    }


def sla_budget_calibration(
    records: list[ApprovalRecord],
    budgets: dict[str, int],
    *,
    budget_sources: dict[str, str] | None = None,
    candidates: list[int] | None = None,
) -> dict:
    """The SLA-budget calibration report over the whole store.

    The queue's budgets get the what-if :func:`trigger_calibration`
    gives the diff thresholds: per tier, the **decided** cases'
    waits as a distribution (the same replay
    :func:`decided_wait_cases` the SLA proposals are drafted from),
    and a sweep of candidate budgets — each row's breach count and
    decision splits, centred on the budget in force with its
    source named (``budget_sources``: the service resolves it from
    the settings record; direct callers get "environment"). Every
    row also carries the moved set against the budget in force
    and whether it is one-sided — the exact evidence
    :func:`draft_sla_proposals` consumes, so a human can read why
    a proposal exists (or why none does) without re-deriving it.

    Read-only by construction: waits are recomputed from the
    stored records and nothing is written. Cases still awaiting a
    decision are not evidence and never appear.
    """
    cases = decided_wait_cases(records)
    grid = candidates if candidates is not None else SLA_CANDIDATES
    sources = budget_sources or {}
    tiers: list[dict] = []
    for tier in QUEUE_SLA_DEFAULTS:
        budget = budgets[tier]
        tier_cases = [c for c in cases if c["risk_tier"] == tier]
        waits = sorted(c["wait_days"] for c in tier_cases)
        values = sorted({int(v) for v in grid} | {budget})
        rows = [_sla_sweep_row(tier_cases, v, budget) for v in values]
        at_current = next(r for r in rows if r["is_current"])
        tiers.append(
            {
                "tier": tier,
                "budget_in_force": budget,
                "budget_source": sources.get(tier, "environment"),
                "decided_cases": len(tier_cases),
                "waits": {
                    "min": waits[0] if waits else None,
                    "p50": _percentile(waits, 50),
                    "p75": _percentile(waits, 75),
                    "p90": _percentile(waits, 90),
                    "max": waits[-1] if waits else None,
                    "mean": (
                        round(sum(waits) / len(waits), 1) if waits else None
                    ),
                },
                "at_current": at_current,
                "sweeps": rows,
            }
        )
    return {
        "budgets": dict(budgets),
        "decided_cases_weighed": len(cases),
        "tiers": tiers,
        "note": (
            "Read-only what-if: decided waits are recomputed from "
            "the stored records at each candidate budget; no "
            "record, queue state, or budget is changed by reading "
            "this report. A row is proposal_ready when its moved "
            "set is one-sided decided evidence — the same bar the "
            "SLA proposal scan drafts from."
        ),
    }


# ---------------------------------------------------------------------------
# Calibration snapshots — the report, kept, so evidence has a date
# ---------------------------------------------------------------------------

#: Store document key for the SLA-calibration snapshots (see
#: :func:`build_calibration_snapshot`): ``{"snapshots": [...]}``,
#: one appended document per operator-kept report.
CALIBRATION_SNAPSHOTS_KEY = "sla_calibration_snapshots"

#: The reserved label for the live side of a calibration diff:
#: ``to_label=current`` diffs a kept report against the report
#: as it stands now, shaped in memory and never stored (see
#: :func:`live_calibration_snapshot`). It is refused as a
#: keepable label — a label is an address for a kept report,
#: and "current" names a moving target, not a kept one.
CURRENT_CALIBRATION_LABEL = "current"


def build_calibration_snapshot(
    report: dict,
    snapshot_id: str,
    actor: str,
    taken_at: str = "",
    label: str = "",
) -> dict:
    """One stored snapshot of the SLA calibration report.

    The report is computed live from the stored records and
    evaporates on read, like the ops digest did before its
    snapshots. A budget debate, though, cites evidence *as of a
    date*: "last month every high-tier row under 3 days was
    proposal-ready" is a claim about a report nobody kept. A
    snapshot keeps the report's decision-relevant shape — the
    budgets in force, how many decided cases were weighed, and
    per tier the budget, its source, the decided count, the wait
    distribution, and the candidate values whose rows were
    ``proposal_ready`` — dated and under the taker's name. The
    taker may also give it a ``label`` ("post-budget-change"):
    a kept report is cited in debates, and the citation needs a
    handle beyond ``cal-0007`` — the label rides the snapshot,
    the list, and the diff, so the argument reads in business
    terms. Retention makes this more pointed, not less: the
    survivors are the cited ones. The full sweep arithmetic is
    deliberately not kept: it is reproducible from the records,
    and what a later reader needs is which moves the evidence
    supported, not every row's splits. Snapshots are
    operator-kept, never on a timer — the same discipline as the
    digest snapshots.
    """
    tiers = []
    for tier_block in report.get("tiers") or []:
        tiers.append(
            {
                "tier": tier_block.get("tier", ""),
                "budget_in_force": tier_block.get("budget_in_force"),
                "budget_source": tier_block.get("budget_source", ""),
                "decided_cases": tier_block.get("decided_cases", 0),
                "waits": dict(tier_block.get("waits") or {}),
                "proposal_ready_rows": sorted(
                    row["value"]
                    for row in tier_block.get("sweeps") or []
                    if row.get("proposal_ready")
                ),
            }
        )
    return {
        "snapshot_id": snapshot_id,
        "taken_at": taken_at or _now(),
        "taken_by": actor,
        "label": label,
        "budgets": dict(report.get("budgets") or {}),
        "decided_cases_weighed": report.get("decided_cases_weighed", 0),
        "tiers": tiers,
    }


def live_calibration_snapshot(report: dict) -> dict:
    """The calibration report as it stands now, in snapshot
    shape — the newer side of a diff against the live report.

    The debates' commonest question is "what moved since
    'march'?" against the report as it stands, not against
    another kept report. This shapes the live report with the
    same builder the keeps use, under the reserved id and label
    ``current``, so :func:`calibration_snapshot_diff` reads it
    exactly as it reads a kept report and the diff cites the
    side as ``current``. Nothing is stored and nothing is
    stamped with a taker: the current side is not a kept
    report, and keeping it remains the operator's separate act.
    """
    return build_calibration_snapshot(
        report,
        CURRENT_CALIBRATION_LABEL,
        "",
        label=CURRENT_CALIBRATION_LABEL,
    )


def calibration_snapshot_diff(older: dict, newer: dict) -> dict:
    """Two kept calibration reports read against each other.

    The headline product is the **proposal-ready movement**: per
    tier, which candidate values' rows became proposal-ready
    between the two snapshots (``entered``) and which stopped
    being (``cleared``) — the evidence a budget debate actually
    turns on, stated as movement rather than two reports side by
    side. The budgets in force and the decided-case counts ride
    along per tier (from → to, and the change), so a ready row
    that appeared because the budget moved reads differently
    from one that appeared because the decided evidence grew.
    A tier missing from either snapshot reads as no ready rows
    and unknown values — absence in a snapshot is a fact about
    the snapshot, not something to invent around. Each side's
    label rides the diff (``from_label`` / ``to_label``) so the
    movement reads against the names the debate actually uses;
    a snapshot kept before labels existed reads as unlabelled.
    Each side's **provenance** rides it too (``from_kept_from``
    / ``to_kept_from``, and a ``provenance`` prose line — see
    :func:`_calibration_provenance_sentence`): a side kept from
    the live report at its keep moment (``kept_from:
    "current"`` — see ``service.keep_calibration_current``) is
    a different kind of evidence from a separately computed
    keep, and a debate reading "march → post-review" should be
    able to tell which it is holding; a plain keep reads as an
    empty provenance. The provenance rides *beside* the
    movement, never inside it: round 19's keep-from-diff pins
    the cited diff's movement sentence to the one the operator
    read, and an annotation about a side's origin must not
    rewrite the sentence that was read. The diff also carries
    the movement as one line of prose (``movement``, see
    :func:`_calibration_movement_sentence`): the tables are the
    evidence, but a debate opens with a sentence, and "rows
    became ready: 3, 5" is not one.

    The newer side may be the live report itself, shaped by
    :func:`live_calibration_snapshot`: the diff then cites it
    as ``current`` in ``to_snapshot`` / ``to_label``, and the
    movement sentence anchors on the kept side exactly as it
    does between two keeps — "what moved since 'march'" is
    answered against now, with nothing stored to answer it.
    """
    def _tiers_by_name(snapshot: dict) -> dict[str, dict]:
        return {
            block.get("tier", ""): block
            for block in snapshot.get("tiers") or []
        }

    before = _tiers_by_name(older)
    after = _tiers_by_name(newer)
    tiers: dict[str, dict] = {}
    for tier in QUEUE_SLA_DEFAULTS:
        old_block = before.get(tier) or {}
        new_block = after.get(tier) or {}
        old_ready = set(old_block.get("proposal_ready_rows") or [])
        new_ready = set(new_block.get("proposal_ready_rows") or [])
        old_budget = old_block.get("budget_in_force")
        new_budget = new_block.get("budget_in_force")
        old_cases = old_block.get("decided_cases")
        new_cases = new_block.get("decided_cases")
        tiers[tier] = {
            "budget_in_force": {
                "from": old_budget,
                "to": new_budget,
                "changed": (
                    old_budget != new_budget
                    and old_budget is not None
                    and new_budget is not None
                ),
            },
            "decided_cases": {
                "from": old_cases,
                "to": new_cases,
                "change": (
                    new_cases - old_cases
                    if old_cases is not None and new_cases is not None
                    else None
                ),
            },
            "proposal_ready_entered": sorted(new_ready - old_ready),
            "proposal_ready_cleared": sorted(old_ready - new_ready),
        }
    diff = {
        "from_snapshot": older.get("snapshot_id", ""),
        "to_snapshot": newer.get("snapshot_id", ""),
        "from_taken_at": older.get("taken_at", ""),
        "to_taken_at": newer.get("taken_at", ""),
        "from_label": older.get("label", ""),
        "to_label": newer.get("label", ""),
        "from_kept_from": older.get("kept_from") or "",
        "to_kept_from": newer.get("kept_from") or "",
        "budgets_changed": any(
            row["budget_in_force"]["changed"] for row in tiers.values()
        ),
        "tiers": tiers,
    }
    diff["movement"] = _calibration_movement_sentence(older, tiers)
    diff["provenance"] = _calibration_provenance_sentence(older, newer)
    return diff


def _rows_phrase(values: list) -> str:
    """Candidate rows named for prose: "3-day row", "2-day and
    4-day rows" — the values are day budgets, spoken as the
    rows the calibration report prints."""
    if len(values) == 1:
        return f"{values[0]}-day row"
    return " and ".join(f"{value}-day" for value in values) + " rows"


def _side_handle(snapshot: dict) -> str:
    """A kept report named the way the debate names it: its
    label in quotes, or its snapshot id when it has none."""
    label = snapshot.get("label") or ""
    return (
        f"'{label}'" if label else snapshot.get("snapshot_id", "")
    ) or "the earlier report"


def _calibration_provenance_sentence(older: dict, newer: dict) -> str:
    """The diff's sides, sourced: which of them were the live
    report once.

    A side kept from the live report is named with its keeper
    and its keep moment — "The newer side ('post-review') was
    kept from the live report by Ops Lead at … — the live
    reading at its keep moment, not a separately computed
    report." The live side itself, when a diff is taken against
    ``current``, is named as what it is: not a kept report at
    all. Two plain keeps have no provenance to state and the
    line is empty — provenance is spoken only when there is
    some. This rides beside ``movement``, never inside it (see
    :func:`calibration_snapshot_diff`): the movement sentence a
    citation carries is the sentence the operator read, and
    annotating a side's origin must not rewrite it."""
    notes: list[str] = []
    for side, snapshot in (("earlier", older), ("newer", newer)):
        if (snapshot.get("kept_from") or "") == "current":
            kept_by = snapshot.get("taken_by") or "an operator"
            kept_at = snapshot.get("taken_at") or ""
            notes.append(
                f"The {side} side ({_side_handle(snapshot)}) was "
                f"kept from the live report by {kept_by}"
                + (f" at {kept_at}" if kept_at else "")
                + " — the live reading at its keep moment, not a "
                "separately computed report."
            )
        elif snapshot.get("snapshot_id") == CURRENT_CALIBRATION_LABEL:
            notes.append(
                f"The {side} side is the live report itself "
                "('current') — not a kept report."
            )
    return " ".join(notes)


def _calibration_movement_sentence(older: dict, tiers: dict) -> str:
    """The calibration diff as one line of prose, in the digest
    headline's register: movement named by tier and row rather
    than counted — "Since 'pre-budget-change': the high tier's
    3-day row became proposal-ready as the budget held at 3
    days." The anchor is the older report by the handle the
    debate uses (its label, or its id when it has none). A
    ready row that appeared while its tier's budget moved says
    so — evidence growth and a budget change are different
    arguments — and a tier whose budget moved with no row
    movement still gets its clause. Nothing moved anywhere:
    the line says the rows held, rather than going quiet."""
    anchor = _side_handle(older)
    clauses: list[str] = []
    for tier, row in tiers.items():
        budget = row["budget_in_force"]
        if budget["changed"]:
            budget_clause = (
                f" as its budget moved from {budget['from']} to "
                f"{budget['to']} days"
            )
        elif budget["from"] is not None and budget["to"] is not None:
            budget_clause = f" as the budget held at {budget['to']} days"
        else:
            budget_clause = ""
        entered = row["proposal_ready_entered"]
        cleared = row["proposal_ready_cleared"]
        if entered:
            clauses.append(
                f"the {tier} tier's {_rows_phrase(entered)} became "
                f"proposal-ready{budget_clause}"
            )
        if cleared:
            clauses.append(
                f"the {tier} tier's {_rows_phrase(cleared)} stopped "
                "being proposal-ready"
                + (budget_clause if not entered else "")
            )
        if not entered and not cleared and budget["changed"]:
            clauses.append(
                f"the {tier} tier's budget moved from "
                f"{budget['from']} to {budget['to']} days"
            )
    if not clauses:
        return (
            f"Since {anchor}: no proposal-ready movement — every "
            "tier's ready rows held."
        )
    return f"Since {anchor}: " + "; ".join(clauses) + "."


def calibration_snapshot_labels(snapshots: list[dict]) -> list[str]:
    """The labels in use among the kept snapshots, in keep order.

    The list a refusal names: an operator told a label is taken
    (or unknown) is shown the handles that actually exist, so
    the correction is a choice, not a guessing game."""
    return [s.get("label", "") for s in snapshots if s.get("label")]


def calibration_snapshot_by_label(
    snapshots: list[dict], label: str
) -> dict | None:
    """The kept snapshot a label cites, or None.

    Labels are unique among snapshots kept since the discipline
    landed (keeping refuses a repeat — see the service), so a
    label addresses one report. Snapshots kept before it could
    repeat a label; there the latest kept wins, because the
    citation a debate makes is to the report the label most
    recently named. An empty label addresses nothing: unlabelled
    snapshots are cited by id, as they always were."""
    if not label:
        return None
    for snapshot in reversed(snapshots):
        if (snapshot.get("label") or "") == label:
            return snapshot
    return None


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

#: Why a refresh withdraws a draft, recorded on the proposal when
#: it happens and quoted by the refresh preview when it would.
_REFRESH_WITHDRAWAL_REASON = (
    "the evidence refresh found no one-sided decided move "
    "for this setting — the snapshot it was drafted from no "
    "longer holds"
)


def _refresh_scan(
    store: Store,
    records: list[ApprovalRecord],
    proposals: list[dict],
    proposal: dict,
) -> dict | None:
    """One draft's evidence scan, re-run against the store as it
    stands: the fresh proposal fields when today's evidence still
    carries a one-sided decided move for the draft's setting,
    else None. The scan is the drafting scan, restricted to the
    proposal's own setting and re-based on the values in force.
    Shared by the refresh and its preview, so a previewed outcome
    is the outcome the refresh itself produces. Reads only —
    nothing here saves."""
    kind = _proposal_kind(proposal)
    rejected = _rejected_directions(proposals, kind)
    if kind == _SLA_KIND:
        budgets = effective_sla_budgets(store)
        tier = proposal["dimension"]
        cases = decided_wait_cases(records)
        tier_cases = [c for c in cases if c["risk_tier"] == tier]
        best = _best_sla_candidate(cases, tier, budgets[tier], rejected)
        if best is not None:
            return _sla_fields(tier, budgets[tier], best, tier_cases)
        return None
    current = effective_thresholds(store)
    pairs = renewal_pairs(records)
    decided_pairs = [
        p for p in pairs if p["status"] in ("approved", "rejected")
    ]
    best = _best_threshold_candidate(
        pairs, proposal["dimension"], current, rejected
    )
    if best is not None:
        return _threshold_fields(
            proposal["dimension"], current, best, pairs, decided_pairs
        )
    return None


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
    fields = _refresh_scan(store, records, proposals, proposal)
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
        proposal["withdrawal_reason"] = _REFRESH_WITHDRAWAL_REASON
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


# ---------------------------------------------------------------------------
# Refresh preview — the sweep's shape, before anything is written
# ---------------------------------------------------------------------------


def _refresh_projection(
    proposal: dict, fields: dict | None, now_dt: datetime
) -> dict:
    """What a refresh of this expired draft would do, projected:
    the stale snapshot the draft holds now, and either the fresh
    snapshot it would take (``would_renew``, with the expiry
    window the renewal would open) or the reason it would leave
    the live list (``would_withdraw``). A projection is computed
    from the same scan the refresh runs — it is not stored, and
    projecting changes no proposal."""
    projection = {
        "proposal_id": proposal["proposal_id"],
        "kind": _proposal_kind(proposal),
        "dimension": proposal.get("dimension", ""),
        "current_value": proposal.get("current_value"),
        "proposed_value": proposal.get("proposed_value"),
        "direction": proposal.get("direction"),
        "expires_at": proposal_expires_at(proposal),
    }
    if fields is None:
        projection.update(
            {
                "outcome": "would_withdraw",
                "refreshed": None,
                "withdrawal_reason": _REFRESH_WITHDRAWAL_REASON,
            }
        )
    else:
        projection.update(
            {
                "outcome": "would_renew",
                "refreshed": {
                    "current_value": fields.get("current_value"),
                    "proposed_value": fields.get("proposed_value"),
                    "direction": fields.get("direction"),
                    "rationale": fields.get("rationale"),
                    "expires_at": _fmt(
                        now_dt + timedelta(days=proposal_expiry_days())
                    ),
                },
                "withdrawal_reason": None,
            }
        )
    return projection


def preview_refresh_proposal(
    store: Store,
    records: list[ApprovalRecord],
    proposal_id: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Preview one expired draft's refresh: run its evidence scan
    and report the outcome it *would* produce — renewed with this
    fresh snapshot, or withdrawn for this reason — without
    writing anything. The refresh's own validations apply (an
    unknown, decided, or still-live proposal fails here exactly
    as it fails there), because the preview shares the refresh's
    scan and its preconditions — it is the refresh with the save
    left out."""
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
    fields = _refresh_scan(store, records, proposals, proposal)
    return _refresh_projection(proposal, fields, now_dt)


def preview_refresh_expired_proposals(
    store: Store,
    records: list[ApprovalRecord],
    *,
    now: datetime | None = None,
) -> dict:
    """Preview the refresh-all sweep: every expired draft scanned
    individually — the same per-draft scans the sweep runs — and
    reported as would-renew / would-withdraw, side by side,
    without a single write. An operator sees the shape of a
    fortnight's drift before committing to it; the sweep that
    follows produces exactly these outcomes on the same store
    (a refresh never changes what another draft's scan reads —
    the scans weigh records and the settings in force, neither
    of which a refresh touches)."""
    now_dt = _now_dt(now)
    proposals = load_proposals(store)
    expired = [p for p in proposals if is_expired(p, now_dt)]
    would_renew: list[dict] = []
    would_withdraw: list[dict] = []
    for proposal in expired:
        fields = _refresh_scan(store, records, proposals, proposal)
        projection = _refresh_projection(proposal, fields, now_dt)
        target = would_renew if fields is not None else would_withdraw
        target.append(projection)
    return {
        "previewed_at": _fmt(now_dt),
        "would_renew": would_renew,
        "would_withdraw": would_withdraw,
        "counts": {
            "swept": len(expired),
            "would_renew": len(would_renew),
            "would_withdraw": len(would_withdraw),
        },
    }
