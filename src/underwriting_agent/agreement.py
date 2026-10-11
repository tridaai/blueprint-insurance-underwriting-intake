"""Triage-vs-decision agreement: did the desk's ending match intake?

The gate decision (approved / rejected) judges the drafted packet.
The underwriter's *final* decision — bound, declined, not taken up
— is what the desk ultimately did with the risk, and it is recorded
separately (see ``service.record_final_decision``). This module
reads the two together and answers the calibration question an
intake lead actually asks: **when triage said a risk was in
appetite, did it bind? When triage said out of appetite, did it
bind anyway?**

The expectation is stated, not implied:

* ``in_appetite`` — the decisive expectation is **bound**.
  Declined is a disagreement; bound is agreement.
* ``out_of_appetite`` — the decisive expectation is **declined**.
  Bound is a disagreement ("bound anyway"); declined is agreement.
* ``borderline`` carries no expectation either way: the desk's
  judgement *is* the decision there, so its counts are reported
  but it enters neither side of the agreement rate.
* ``incomplete`` submissions are excluded from the bands entirely
  (triage never reached a view) and counted on their own line.
* ``not_taken_up`` is never decisive: the broker or insured walked
  away, which neither confirms nor refutes the triage. It is
  counted in every band it lands in and kept out of the rate.

Everything here is counts. An agreement rate is computed over the
decisive pairs only, and the report carries an explicit
small-sample note whenever the decided base is under
:data:`SMALL_SAMPLE_FLOOR` — on a young book these are tallies to
read, not rates to lean on, and the report says so in words.
"""

from __future__ import annotations

#: Below this many decisive (band, decision) pairs, the report's
#: note states plainly that the counts are too small to read as
#: rates.
SMALL_SAMPLE_FLOOR = 10

#: The decisive expectation per triage outcome band: the final
#: decision that agrees with the band's view. Bands absent here
#: (borderline, incomplete) carry no expectation.
_EXPECTED_DECISION = {
    "in_appetite": "bound",
    "out_of_appetite": "declined",
}

#: The final underwriting outcomes a desk can record.
FINAL_DECISIONS = ("bound", "declined", "not_taken_up")

_DECISIONS = FINAL_DECISIONS
_BANDS = ("in_appetite", "borderline", "out_of_appetite")


def _empty_band() -> dict:
    return {
        "decided": 0,
        "bound": 0,
        "declined": 0,
        "not_taken_up": 0,
        "decisive": 0,
        "agreed": 0,
        "disagreed": 0,
        "expected_decision": None,
    }


def triage_agreement(records: list) -> dict:
    """Agreement counts over every record with a final decision.

    ``records`` are store records (``ApprovalRecord``); the current
    final decision is the last entry of each record's
    ``final_decisions`` history (superseded entries are history,
    not votes). Bands key on the triage outcome, tiers on the
    triage risk tier; every cell is a plain count.
    """
    bands: dict[str, dict] = {band: _empty_band() for band in _BANDS}
    for band, expected in _EXPECTED_DECISION.items():
        bands[band]["expected_decision"] = expected
    tiers: dict[str, dict] = {}
    incomplete_decided = 0
    decided_total = 0
    reason_codes: dict[str, int] = {}

    for record in records:
        history = getattr(record, "final_decisions", None) or []
        if not history:
            continue
        current = history[-1]
        decision = current.get("decision", "")
        if decision not in _DECISIONS:
            continue
        decided_total += 1
        code = (current.get("reason_code") or "").strip()
        if code:
            reason_codes[code] = reason_codes.get(code, 0) + 1

        tier = record.result.triage.risk_tier or "unknown"
        tier_row = tiers.setdefault(
            tier, {"decided": 0, "bound": 0, "declined": 0, "not_taken_up": 0}
        )
        tier_row["decided"] += 1
        tier_row[decision] += 1

        band = record.result.triage.outcome
        if band == "incomplete":
            incomplete_decided += 1
            continue
        row = bands.setdefault(band, _empty_band())
        row["decided"] += 1
        row[decision] += 1
        expected = _EXPECTED_DECISION.get(band)
        if expected is not None and decision != "not_taken_up":
            row["decisive"] += 1
            if decision == expected:
                row["agreed"] += 1
            else:
                row["disagreed"] += 1

    decisive = sum(row["decisive"] for row in bands.values())
    agreed = sum(row["agreed"] for row in bands.values())
    disagreed = sum(row["disagreed"] for row in bands.values())
    small_sample = decisive < SMALL_SAMPLE_FLOOR
    return {
        "decided_total": decided_total,
        "by_outcome_band": bands,
        "by_risk_tier": tiers,
        "incomplete_decided": incomplete_decided,
        "decisive": decisive,
        "agreed": agreed,
        "disagreed": disagreed,
        "agreement_rate": round(agreed / decisive, 4) if decisive else None,
        "reason_codes": reason_codes,
        "small_sample": small_sample,
        "small_sample_floor": SMALL_SAMPLE_FLOOR,
        "note": (
            f"Only {decisive} decisive decision(s) recorded — read the "
            "counts, not the rate; agreement on a base this small is a "
            "tally, not evidence."
            if small_sample
            else "Agreement is computed over decisive pairs only: "
            "in-appetite expects bound, out-of-appetite expects "
            "declined; borderline carries no expectation and "
            "not-taken-up is never decisive."
        ),
    }
