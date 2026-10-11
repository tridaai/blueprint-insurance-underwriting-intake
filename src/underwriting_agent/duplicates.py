"""Cross-broker duplicate detection: the same risk, submitted twice.

Specialty intake has a classic failure mode: one insured risk
arrives twice — through two brokers, under slightly different
names, with different submission ids — and two underwriters work
(and may quote) the same risk while the pipeline double-counts it.
Nothing in the pipeline caused that; the submissions are each
legitimate. What was missing is a pass that *compares* stored
submissions with each other. This module is that pass.

The pass is code, end to end — no model call. Candidate pairs are
scored from the extracted/stored submission fields:

* **insured-name similarity** — the dominant term, and a hard
  gate: name matching is normalised deterministically (case,
  punctuation, legal-entity suffixes — Inc / LLC / LP / Corp — and
  word order are all set aside before tokens are compared), so
  "Kestrel Ridge Vineyards LLC" and "Kestrel Ridge Vineyard, LLC"
  compare as the same insured while two genuinely different
  insureds in the same class and ZIP-less territory never flag on
  figures alone. Below ``NAME_GATE`` there is no finding, whatever
  the other fields say — figures alone never accuse two different
  insureds of being one risk.
* **class of business** and **line of business** agreement,
* **effective-date proximity** (full credit on the same date,
  fading to zero at ``DATE_WINDOW_DAYS``),
* **revenue** and **requested-limit proximity** (ratio of the
  smaller to the larger figure; a resubmission rarely restates
  either exactly).

The intake schema carries no street address or ZIP, so no
address signal is scored — the weights below redistribute over
the fields that exist, and the finding says so in its policy
block rather than implying a check that never ran.

Weights and the queue threshold live in :data:`DUPLICATE_POLICY`
(one place, tunable via the environment) and every finding
carries the policy it was scored under plus the field-by-field
evidence — each field's two values, its similarity, its weight,
and the points it contributed — so the human who reviews the
finding can see *why* the pair flagged, not just that it did.

Detection only flags. It never merges, blocks, or alters either
submission; a finding is a question for a human, and the human's
answer (see ``service.resolve_duplicate``) is what the queue
remembers.
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

from .config import env_float
from .triage import _limit_value

#: Store document key under which finding resolutions live (the
#: proposals/corpus document idiom): {resolutions: {pair_key:
#: {status, actor, reason, decided_at, history}}}. Detection
#: itself is computed on read over the records; only the human
#: answers are stored, and a stored answer is what keeps a
#: resolved pair from re-opening.
FINDINGS_DOCUMENT = "duplicate_findings"

# ---------------------------------------------------------------------------
# Policy: the weights, the gate, the threshold — one place.
# ---------------------------------------------------------------------------

#: Field weights for the pair score (they sum to 1.0). The name
#: dominates: this pass exists to catch the same insured arriving
#: twice, and every other field is corroboration.
DUPLICATE_POLICY: dict = {
    "name": 0.45,
    "class_of_business": 0.15,
    "line_of_business": 0.10,
    "effective_date": 0.10,
    "revenue": 0.10,
    "requested_limit": 0.10,
}

#: Default queue threshold (DUPLICATE_MATCH_THRESHOLD): a pair at
#: or above this score becomes a POSSIBLE DUPLICATE finding.
DEFAULT_MATCH_THRESHOLD = 0.70

#: The name gate (DUPLICATE_NAME_GATE): below this name similarity
#: a pair is never scored at all — different insured, full stop.
DEFAULT_NAME_GATE = 0.60

#: Effective dates this far apart still earn partial credit; the
#: credit fades linearly to zero at the window edge.
DATE_WINDOW_DAYS = 45

#: A token pair this similar (character ratio) counts as the same
#: token when token sets are compared — "vineyards" ~ "vineyard".
TOKEN_MATCH_RATIO = 0.80

#: Legal-entity and filler tokens set aside before name comparison:
#: they distinguish registrations, not risks.
_NAME_STOP_TOKENS = {
    "inc", "llc", "llp", "lp", "corp", "corporation", "company",
    "co", "ltd", "limited", "incorporated", "the", "and", "&",
}


def duplicate_policy() -> dict:
    """The scoring policy in force, with environment overrides."""
    return {
        "weights": dict(DUPLICATE_POLICY),
        "threshold": env_float(
            "DUPLICATE_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD
        ),
        "name_gate": env_float("DUPLICATE_NAME_GATE", DEFAULT_NAME_GATE),
        "date_window_days": DATE_WINDOW_DAYS,
        "signals_not_in_schema": [
            "street address / ZIP — the intake schema carries no "
            "address field, so no address signal is scored"
        ],
    }


# ---------------------------------------------------------------------------
# Name normalisation + similarity (deterministic, no model)
# ---------------------------------------------------------------------------


def normalize_insured_name(name: str) -> list[str]:
    """The comparable token list for an insured name.

    Lowercases, turns punctuation into spaces, drops legal-entity
    suffixes and filler tokens, and sorts — so word order, commas,
    and "LLC" vs "Inc" registrations never decide a comparison.
    """
    text = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    tokens = [t for t in text.split() if t not in _NAME_STOP_TOKENS]
    return sorted(tokens)


def _token_similarity(left: list[str], right: list[str]) -> float:
    """Fuzzy token-set similarity in [0, 1].

    Each token in the smaller set looks for its best character-ratio
    match in the larger set; a match at ``TOKEN_MATCH_RATIO`` or
    better counts in full, a near miss counts fractionally. The
    score is matched credit over the larger set's size, so an
    extra distinctive token ("creek" in "Alder Creek …") costs the
    pair even when the shared tokens agree.
    """
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    credit = 0.0
    for token in small:
        best = max(
            SequenceMatcher(None, token, other).ratio() for other in large
        )
        credit += best if best >= TOKEN_MATCH_RATIO else best * 0.5
    return credit / len(large)


def name_similarity(left_name: str, right_name: str) -> float:
    """Similarity of two insured names in [0, 1].

    The blend of the fuzzy token-set score and the character ratio
    of the normalised joined names; identical normalised names
    score exactly 1.0.
    """
    left = normalize_insured_name(left_name)
    right = normalize_insured_name(right_name)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    token_score = _token_similarity(left, right)
    char_score = SequenceMatcher(None, " ".join(left), " ".join(right)).ratio()
    return round(0.6 * token_score + 0.4 * char_score, 4)


# ---------------------------------------------------------------------------
# Field similarities
# ---------------------------------------------------------------------------


def _text_key(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _figure_proximity(left: float | None, right: float | None) -> float | None:
    """Smaller/larger ratio for two figures; None when either side
    is missing (a missing figure is no evidence either way)."""
    if not left or not right:
        return None
    return round(min(left, right) / max(left, right), 4)


def _date_proximity(left_raw: str, right_raw: str) -> float | None:
    """1.0 for the same effective date, fading to 0.0 at the
    policy window; None when either date is unparseable."""
    try:
        left = date.fromisoformat((left_raw or "")[:10])
        right = date.fromisoformat((right_raw or "")[:10])
    except ValueError:
        return None
    days = abs((left - right).days)
    if days >= DATE_WINDOW_DAYS:
        return 0.0
    return round(1.0 - days / DATE_WINDOW_DAYS, 4)


# ---------------------------------------------------------------------------
# The pair score + the pass over stored records
# ---------------------------------------------------------------------------


def score_pair(left: dict, right: dict, policy: dict | None = None) -> dict | None:
    """Score one candidate pair from two stored submission payloads.

    Returns None when the pair fails the name gate (different
    insureds — never a finding). Otherwise the score, the verdict
    against the threshold, and the field-by-field evidence: every
    scored field with both values, its similarity, its weight, and
    the points it contributed. A field missing on either side
    contributes nothing and says so — absence is not agreement.
    """
    policy = policy or duplicate_policy()
    weights = policy["weights"]
    name_sim = name_similarity(
        left.get("insured_name", ""), right.get("insured_name", "")
    )
    if name_sim < policy["name_gate"]:
        return None

    evidence: list[dict] = [
        {
            "field": "insured_name",
            "left": left.get("insured_name", ""),
            "right": right.get("insured_name", ""),
            "similarity": name_sim,
            "weight": weights["name"],
            "points": round(name_sim * weights["name"], 4),
        }
    ]

    def add(field: str, left_value, right_value, similarity: float | None) -> None:
        if similarity is None:
            evidence.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                    "similarity": None,
                    "weight": weights[field],
                    "points": 0.0,
                    "note": "missing on one side — no evidence either way",
                }
            )
            return
        evidence.append(
            {
                "field": field,
                "left": left_value,
                "right": right_value,
                "similarity": similarity,
                "weight": weights[field],
                "points": round(similarity * weights[field], 4),
            }
        )

    add(
        "class_of_business",
        left.get("class_of_business", ""),
        right.get("class_of_business", ""),
        1.0
        if _text_key(left.get("class_of_business", ""))
        == _text_key(right.get("class_of_business", ""))
        and _text_key(left.get("class_of_business", ""))
        else None,
    )
    add(
        "line_of_business",
        left.get("line_of_business", ""),
        right.get("line_of_business", ""),
        1.0
        if _text_key(left.get("line_of_business", ""))
        == _text_key(right.get("line_of_business", ""))
        and _text_key(left.get("line_of_business", ""))
        else None,
    )
    add(
        "effective_date",
        left.get("effective_date_requested", ""),
        right.get("effective_date_requested", ""),
        _date_proximity(
            left.get("effective_date_requested", ""),
            right.get("effective_date_requested", ""),
        ),
    )
    add(
        "revenue",
        left.get("revenue"),
        right.get("revenue"),
        _figure_proximity(left.get("revenue"), right.get("revenue")),
    )
    add(
        "requested_limit",
        left.get("requested_limit", ""),
        right.get("requested_limit", ""),
        _figure_proximity(
            _limit_value(left.get("requested_limit", "") or ""),
            _limit_value(right.get("requested_limit", "") or ""),
        ),
    )

    score = round(sum(entry["points"] for entry in evidence), 4)
    return {
        "score": score,
        "above_threshold": score >= policy["threshold"],
        "evidence": evidence,
        "context": {
            "broker_left": left.get("broker_name", ""),
            "broker_right": right.get("broker_name", ""),
            "different_brokers": _text_key(left.get("broker_name", ""))
            != _text_key(right.get("broker_name", "")),
            "prior_carrier_left": left.get("prior_carrier", ""),
            "prior_carrier_right": right.get("prior_carrier", ""),
        },
    }


def pair_key(left_id: str, right_id: str) -> str:
    """The stable identity of an unordered pair of submissions."""
    first, second = sorted([left_id, right_id])
    return f"{first}::{second}"


def find_candidate_pairs(records: list, policy: dict | None = None) -> list[dict]:
    """Every above-threshold pair across the stored records.

    ``records`` are store records (``ApprovalRecord``); the scored
    fields come from each record's stored submission payload.
    Pairs are unordered, a record is never paired with itself, and
    bundle sub-analyses are not separate records — a bundle is one
    stored submission and pairs once, as itself.
    """
    policy = policy or duplicate_policy()
    findings: list[dict] = []
    ordered = sorted(records, key=lambda r: r.result.submission_id)
    for index, left_record in enumerate(ordered):
        for right_record in ordered[index + 1:]:
            left_id = left_record.result.submission_id
            right_id = right_record.result.submission_id
            if left_id == right_id:
                continue
            scored = score_pair(
                left_record.submission or {},
                right_record.submission or {},
                policy,
            )
            if scored is None or not scored["above_threshold"]:
                continue
            findings.append(
                {
                    "pair_key": pair_key(left_id, right_id),
                    "submission_ids": [left_id, right_id],
                    "insured_names": [
                        (left_record.submission or {}).get("insured_name", ""),
                        (right_record.submission or {}).get("insured_name", ""),
                    ],
                    "score": scored["score"],
                    "evidence": scored["evidence"],
                    "context": scored["context"],
                }
            )
    findings.sort(key=lambda f: (-f["score"], f["pair_key"]))
    return findings
