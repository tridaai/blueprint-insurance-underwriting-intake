"""Loss development in the renewal diff: claims matched per claim.

The aggregate loss-history change says the totals moved. Underwriters
price how last year's losses *developed* — which claims are new on the
run, which dropped off it, and which ones quietly grew. Matching is by
claim id when the runs carry one, else date + description; adverse
development (incurred on known claims rising past the floor) is a
referral-trigger input, favourable development is not.
"""

from __future__ import annotations

from underwriting_agent.diffing import (
    compute_submission_diff,
    diff_packet_lines,
    diff_referral_triggers,
    match_loss_development,
)
from underwriting_agent.extractor import extract_documents
from underwriting_agent.facts import compute_facts
from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.samples import load_sample_submissions, sample_submission_models
from underwriting_agent.schemas import LossRecord
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
MODELS = {s.submission_id: s for s in sample_submission_models()}


def loss(date, amount, description, status="closed", claim_id=""):
    return LossRecord(
        date=date,
        amount_incurred=amount,
        description=description,
        status=status,
        claim_id=claim_id,
    )


def _facts_for(model):
    return compute_facts(model, extract_documents(model, MockModelBackend()))


def _prior_bundle(model, outcome="in_appetite", tier="low") -> dict:
    return {
        "submission_id": model.submission_id,
        "submitted_at": model.submitted_at,
        "outcome": outcome,
        "risk_tier": tier,
        "submission": model.model_dump(mode="json"),
        "facts": _facts_for(model).model_dump(mode="json"),
    }


def _renewal(prior_losses, current_losses, **overrides):
    """A SUB-1001-shaped renewal pair differing only in the loss run."""
    base = MODELS["SUB-1001"]
    prior = base.model_copy(
        update={"submission_id": "SUB-8001", "losses": prior_losses}
    )
    current = base.model_copy(
        update={"submission_id": "SUB-8002", "losses": current_losses, **overrides}
    )
    return prior, current


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_by_claim_id_survives_a_reworded_description():
    prior = [loss("2024-03-14", 100_000, "Water damage, sprinkler failure", "open", "CLM-77")]
    current = [loss("2024-03-14", 160_000, "Sprinkler line failure — water damage", "open", "CLM-77")]
    dev = match_loss_development(prior, current)
    assert dev is not None
    assert dev.appeared == [] and dev.disappeared == []
    assert len(dev.developed) == 1
    moved = dev.developed[0]
    assert moved.prior_incurred == 100_000 and moved.current_incurred == 160_000
    assert moved.delta_incurred == 60_000


def test_match_by_date_and_description_without_claim_ids():
    prior = [loss("2023-06-02", 48_000, "Wind damage to roof")]
    current = [loss("2023-06-02", 48_000, "Wind damage to roof", "closed")]
    dev = match_loss_development(prior, current)
    assert dev.appeared == [] and dev.disappeared == [] and dev.developed == []
    assert dev.net_incurred_change == 0


def test_match_by_description_when_the_date_was_corrected():
    prior = [loss("2025-01-19", 26_500, "Theft of copper piping")]
    current = [loss("2025-01-20", 30_000, "Theft of copper piping")]
    dev = match_loss_development(prior, current)
    assert dev.appeared == [] and dev.disappeared == []
    assert len(dev.developed) == 1 and dev.developed[0].delta_incurred == 3_500


def test_appeared_and_disappeared_claims_are_reported():
    prior = [
        loss("2023-06-02", 48_000, "Wind damage to roof"),
        loss("2022-11-05", 12_000, "Frozen pipe, annex"),
    ]
    current = [
        loss("2023-06-02", 48_000, "Wind damage to roof"),
        loss("2026-02-11", 75_000, "Equipment breakdown", "open"),
    ]
    dev = match_loss_development(prior, current)
    assert [c.description for c in dev.appeared] == ["Equipment breakdown"]
    assert [c.description for c in dev.disappeared] == ["Frozen pipe, annex"]
    assert dev.developed == []
    assert dev.net_incurred_change == 63_000  # 123,000 current − 60,000 prior


def test_status_only_change_counts_as_development():
    prior = [loss("2024-03-14", 100_000, "Water damage", "open")]
    current = [loss("2024-03-14", 100_000, "Water damage", "closed")]
    dev = match_loss_development(prior, current)
    assert len(dev.developed) == 1
    assert dev.developed[0].delta_incurred == 0
    assert dev.developed[0].prior_status == "open"
    assert dev.developed[0].current_status == "closed"


def test_no_claims_on_either_side_is_no_development():
    assert match_loss_development([], []) is None


def test_duplicate_claims_pair_in_order_not_all_against_all():
    prior = [
        loss("2024-01-10", 10_000, "Slip and fall", "closed", "CLM-1"),
        loss("2024-01-10", 20_000, "Slip and fall", "closed", "CLM-1"),
    ]
    current = [loss("2024-01-10", 15_000, "Slip and fall", "closed", "CLM-1")]
    dev = match_loss_development(prior, current)
    assert len(dev.developed) == 1 and len(dev.disappeared) == 1
    assert dev.developed[0].delta_incurred == 5_000  # first pairs with first


# ---------------------------------------------------------------------------
# On the diff: packet + evidence surfaces
# ---------------------------------------------------------------------------


def test_diff_carries_development_into_packet_and_evidence():
    prior, current = _renewal(
        [loss("2024-03-14", 100_000, "Water damage — sprinkler failure", "open")],
        [loss("2024-03-14", 190_000, "Water damage — sprinkler failure", "open")],
    )
    diff = compute_submission_diff(current, _facts_for(current), _prior_bundle(prior))
    assert diff is not None and diff.loss_development is not None
    assert len(diff.loss_development.developed) == 1

    packet = "\n".join(diff_packet_lines(diff))
    assert "Loss development since the prior loss run:" in packet
    assert "incurred $100,000 -> $190,000 (+90,000)" in packet
    assert "net incurred movement across the run: +90,000" in packet

    from underwriting_agent.diffing import diff_evidence_lines

    evidence = "\n".join(diff_evidence_lines(diff))
    assert "loss development — 1 claim(s) developed (incurred +90,000)" in evidence
    assert "claim developed — 2024-03-14" in evidence


def test_identical_loss_runs_show_no_development_section():
    claims = [loss("2024-03-14", 100_000, "Water damage", "open")]
    prior, current = _renewal(claims, list(claims))
    diff = compute_submission_diff(current, _facts_for(current), _prior_bundle(prior))
    assert diff is not None
    assert diff.loss_development is not None
    assert diff.loss_development.appeared == []
    assert diff.loss_development.developed == []
    assert "Loss development" not in "\n".join(diff_packet_lines(diff))


# ---------------------------------------------------------------------------
# Referral triggers: adverse development refers; favourable does not
# ---------------------------------------------------------------------------


def test_adverse_development_over_the_floor_triggers():
    prior, current = _renewal(
        [loss("2024-03-14", 100_000, "Water damage — sprinkler failure", "open")],
        [loss("2024-03-14", 175_000, "Water damage — sprinkler failure", "open")],
    )
    triggers = diff_referral_triggers(current, _facts_for(current), _prior_bundle(prior))
    assert len(triggers) == 1
    assert "adverse loss development" in triggers[0]
    assert "$75,000" in triggers[0]


def test_favourable_development_does_not_trigger():
    prior, current = _renewal(
        [loss("2024-03-14", 200_000, "Water damage — sprinkler failure", "open")],
        [loss("2024-03-14", 90_000, "Water damage — sprinkler failure", "closed")],
    )
    triggers = diff_referral_triggers(current, _facts_for(current), _prior_bundle(prior))
    assert triggers == []


def test_small_adverse_development_stays_under_the_floor():
    prior, current = _renewal(
        [loss("2024-03-14", 100_000, "Water damage — sprinkler failure", "open")],
        [loss("2024-03-14", 130_000, "Water damage — sprinkler failure", "open")],
    )
    triggers = diff_referral_triggers(current, _facts_for(current), _prior_bundle(prior))
    assert triggers == []


def test_appeared_claim_is_not_double_counted_as_development():
    # A claim that was NOT on the prior run is a new claim (its own
    # trigger family), never "development" — the two families read
    # disjoint sets by construction.
    prior, current = _renewal(
        [],
        [loss("2026-08-15", 120_000, "Business email compromise", "closed")],
    )
    triggers = diff_referral_triggers(current, _facts_for(current), _prior_bundle(prior))
    assert len(triggers) == 1
    assert "new claims" in triggers[0]


def test_development_refers_the_renewal_end_to_end():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    prior, current = _renewal(
        [loss("2024-03-14", 100_000, "Water damage — sprinkler failure", "open")],
        [loss("2024-03-14", 190_000, "Water damage — sprinkler failure", "open")],
    )
    service.analyze(prior.model_dump(mode="json"))
    result = service.analyze(current.model_dump(mode="json"))
    # One known claim, statically unremarkable (largest < $250k,
    # count 1): the referral exists because the claim DEVELOPED.
    assert result.triage.outcome == "borderline"
    assert any("adverse loss development" in t for t in result.triage.referral_triggers)
    assert result.submission_diff is not None
    assert result.submission_diff.loss_development is not None
    assert "Loss development since the prior loss run:" in (
        result.drafts.underwriter_packet
    )
