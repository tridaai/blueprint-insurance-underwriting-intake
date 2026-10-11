"""Provider-mode change impact: the opt-in second pass over the
rules-only change-impact report's moved lines. Only moved lines are
cross-checked (the call log proves it), the pass is bounded and
costed, its verdicts are reported separately from the rules report
it never rewrites, and without a real provider it reports itself
unavailable — the offline mock is never dressed up as a second
opinion.
"""

from __future__ import annotations

import pytest

from underwriting_agent.facts import compute_facts
from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.retriage import guideline_change_impact, provider_change_impact
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import SubmissionInput, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore
from underwriting_agent.triage import classify_submission

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


class ScriptedLLM:
    """A scripted provider: answers classify per submission id and
    logs every call, so a test can prove exactly which lines were
    cross-checked. Usage ticks a fixed, priced amount per call."""

    name = "scripted-llm"
    _model = "gpt-4o-mini"

    def __init__(self, verdicts: dict[str, dict]) -> None:
        self._verdicts = verdicts
        self.calls: list[str] = []
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def usage_totals(self):
        return dict(self._usage)

    def reset_usage(self):
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def classify_with_llm(self, context):
        self.calls.append(context["submission_id"])
        self._usage["calls"] += 1
        self._usage["input_tokens"] += 1000
        self._usage["output_tokens"] += 200
        verdict = self._verdicts[context["submission_id"]]
        return {**verdict, "backend": self.name}


def _tightened_appetite(sub, facts) -> Triage:
    """A simulated corpus change: cyber is now referred by default."""
    triage = classify_submission(sub, facts)
    if sub.line_of_business == "cyber" and triage.outcome == "in_appetite":
        return triage.model_copy(update={"outcome": "borderline"})
    return triage


def _tighten_everything(sub, facts) -> Triage:
    triage = classify_submission(sub, facts)
    if triage.outcome == "in_appetite":
        return triage.model_copy(update={"outcome": "borderline"})
    return triage


def _property_tightened(sub, facts) -> Triage:
    """A simulated corpus change that only bites the property line."""
    triage = classify_submission(sub, facts)
    if sub.line_of_business == "property" and triage.outcome == "in_appetite":
        return triage.model_copy(
            update={
                "outcome": "borderline",
                "referral_triggers": [
                    "largest claim $310,000 exceeds 250,000 "
                    "(referral trigger: severity)"
                ],
            }
        )
    return triage


def seed_store(*submission_ids: str) -> InMemoryStore:
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    for sid in submission_ids:
        service.analyze(SAMPLES[sid])
    return store


def fresh_triage(sid: str, classifier) -> Triage:
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(SAMPLES[sid])
    sub = SubmissionInput.model_validate(SAMPLES[sid])
    facts = compute_facts(sub, result.extractions)
    return classifier(sub, facts)


# ---------------------------------------------------------------------------
# Only the moved lines are cross-checked
# ---------------------------------------------------------------------------


def test_only_moved_lines_are_cross_checked():
    store = seed_store("SUB-1001", "SUB-1009", "SUB-1004")
    recorded = store.get("SUB-1001").result.triage
    fake = ScriptedLLM(
        {
            "SUB-1001": {
                "outcome": recorded.outcome,
                "risk_tier": recorded.risk_tier,
                "confidence": 0.9,
                "rationale": "scripted: the recorded outcome stands",
            }
        }
    )
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tightened_appetite
    )
    provider = report["provider"]
    assert provider["available"] is True
    assert provider["lines_moved"] == 1
    assert provider["lines_cross_checked"] == 1
    # The call log is the proof: SUB-1009 and SUB-1004 did not move,
    # so the model was never asked about them.
    assert fake.calls == ["SUB-1001"]

    (check,) = provider["checks"]
    assert check["submission_id"] == "SUB-1001"
    assert check["recorded_outcome"] == "in_appetite"
    assert check["rules_outcome"] == "borderline"
    assert check["llm_outcome"] == recorded.outcome
    # The model sided with the recorded outcome against the fresh
    # rules — reported as such, in the provider's own section. The
    # simulated change moves only the outcome, so on tier the
    # model's answer matches both sides — reported as "both",
    # exactly the distinction the field exists to make.
    assert check["agrees_with"] == "recorded"
    assert check["tier_agrees_with"] == "both"


def test_a_model_that_sides_with_the_rules_reports_agree():
    store = seed_store("SUB-1001")
    fresh = fresh_triage("SUB-1001", _tightened_appetite)
    fake = ScriptedLLM(
        {
            "SUB-1001": {
                "outcome": fresh.outcome,
                "risk_tier": fresh.risk_tier,
                "confidence": 0.88,
                "rationale": "scripted: the fresh rules are right",
            }
        }
    )
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tightened_appetite
    )
    (check,) = report["provider"]["checks"]
    assert check["agrees_with"] == "rules"
    assert check["tier_agrees_with"] == "both"  # the change moved no tier
    assert check["cross_check_resolution"] == "agree"


def _tighten_outcome_and_tier(sub, facts) -> Triage:
    """A simulated change that moves outcome AND tier, so the two
    agreement fields can be told apart."""
    triage = classify_submission(sub, facts)
    if sub.line_of_business == "cyber" and triage.outcome == "in_appetite":
        return triage.model_copy(
            update={"outcome": "borderline", "risk_tier": "elevated"}
        )
    return triage


def test_tier_agreement_is_judged_on_its_own_side():
    store = seed_store("SUB-1001")
    recorded = store.get("SUB-1001").result.triage
    assert recorded.risk_tier != "elevated"
    fake = ScriptedLLM(
        {
            "SUB-1001": {
                "outcome": "borderline",  # the rules' outcome…
                "risk_tier": recorded.risk_tier,  # …but the recorded tier
                "confidence": 0.9,
                "rationale": "scripted: split agreement",
            }
        }
    )
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tighten_outcome_and_tier
    )
    (check,) = report["provider"]["checks"]
    assert check["agrees_with"] == "rules"
    assert check["tier_agrees_with"] == "recorded"


def test_the_rules_report_is_returned_unrewritten():
    store = seed_store("SUB-1001", "SUB-1009")
    fake = ScriptedLLM(
        {
            "SUB-1001": {
                "outcome": "out_of_appetite",
                "risk_tier": "elevated",
                "confidence": 0.7,
                "rationale": "scripted: a third answer entirely",
            }
        }
    )
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tightened_appetite
    )
    rules_only = guideline_change_impact(
        store.all_records(), classifier=_tightened_appetite
    )
    assert report["rules"] == rules_only  # byte for byte the rules report
    (check,) = report["provider"]["checks"]
    assert check["agrees_with"] == "neither"
    # And the stored record was not touched by either report.
    assert store.get("SUB-1001").result.triage.outcome == "in_appetite"


# ---------------------------------------------------------------------------
# Bounded and costed
# ---------------------------------------------------------------------------


def test_the_cap_binds_and_names_the_lines_it_leaves_out():
    store = seed_store("SUB-1001", "SUB-1009")  # both move under _tighten_everything
    verdict = {
        "outcome": "borderline",
        "risk_tier": "moderate",
        "confidence": 0.9,
        "rationale": "scripted",
    }
    fake = ScriptedLLM({"SUB-1001": verdict, "SUB-1009": verdict})
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tighten_everything, cap=1
    )
    provider = report["provider"]
    assert provider["lines_moved"] == 2
    assert provider["lines_cross_checked"] == 1
    assert provider["cap"] == 1
    assert fake.calls == ["SUB-1001"]  # the second mover was never called
    assert [o["submission_id"] for o in provider["lines_over_cap"]] == ["SUB-1009"]


def test_usage_and_cost_are_accounted_from_the_backend():
    store = seed_store("SUB-1001")
    verdict = {
        "outcome": "borderline",
        "risk_tier": "moderate",
        "confidence": 0.9,
        "rationale": "scripted",
    }
    fake = ScriptedLLM({"SUB-1001": verdict})
    report = provider_change_impact(
        store.all_records(), fake, classifier=_tightened_appetite
    )
    usage = report["provider"]["usage"]
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 200
    # gpt-4o-mini at (0.15, 0.60) per 1M tokens — the telemetry table.
    assert usage["estimated_cost_usd"] == pytest.approx(0.00027)


def test_a_failing_provider_call_is_reported_not_fatal():
    store = seed_store("SUB-1001")

    class Boom(ScriptedLLM):
        def classify_with_llm(self, context):
            raise RuntimeError("provider exploded")

    report = provider_change_impact(
        store.all_records(), Boom({}), classifier=_tightened_appetite
    )
    provider = report["provider"]
    assert provider["available"] is True
    (check,) = provider["checks"]
    assert "provider exploded" in check["error"]
    assert report["rules"]["moved_count"] == 1  # the rules stand regardless


# ---------------------------------------------------------------------------
# Bundles: the moved line only, never the whole bundle
# ---------------------------------------------------------------------------


def test_bundle_moved_lines_are_checked_per_line():
    store = seed_store("SUB-1014")  # property + cyber bundle
    verdict = {
        "outcome": "borderline",
        "risk_tier": "moderate",
        "confidence": 0.9,
        "rationale": "scripted",
    }
    fake = ScriptedLLM({"SUB-1014-PROPERTY": verdict})
    report = provider_change_impact(
        store.all_records(), fake, classifier=_property_tightened
    )
    provider = report["provider"]
    assert provider["lines_moved"] == 1
    # Only the moved line's context reached the model — not the
    # unmoved cyber line, and not a flattened bundle context.
    assert fake.calls == ["SUB-1014-PROPERTY"]
    (check,) = provider["checks"]
    assert check["submission_id"] == "SUB-1014"
    assert check["line_submission_id"] == "SUB-1014-PROPERTY"
    assert check["line_of_business"] == "property"


# ---------------------------------------------------------------------------
# Offline: honestly unavailable, never substituted
# ---------------------------------------------------------------------------


def test_offline_backend_reports_the_pass_unavailable():
    store = seed_store("SUB-1001")
    report = provider_change_impact(
        store.all_records(), MockModelBackend(), classifier=_tightened_appetite
    )
    provider = report["provider"]
    assert provider["available"] is False
    assert "offline" in provider["reason"]
    assert provider["checks"] == []
    assert provider["usage"] is None
    # The rules report still stands in full.
    assert report["rules"]["moved_count"] == 1


def test_service_resolves_the_mock_to_unavailable():
    store = seed_store("SUB-1001")
    service = SubmissionService(store=store, checkpointer=False)
    report = service.guideline_change_impact_provider(
        classifier=_tightened_appetite
    )
    assert report["provider"]["available"] is False
    assert report["rules"]["moved_count"] == 1


def test_service_with_a_provider_runs_the_pass():
    store = seed_store("SUB-1001")
    recorded = store.get("SUB-1001").result.triage
    fake = ScriptedLLM(
        {
            "SUB-1001": {
                "outcome": recorded.outcome,
                "risk_tier": recorded.risk_tier,
                "confidence": 0.9,
                "rationale": "scripted",
            }
        }
    )
    service = SubmissionService(backend=fake, store=store, checkpointer=False)
    report = service.guideline_change_impact_provider(
        classifier=_tightened_appetite
    )
    assert report["provider"]["available"] is True
    assert fake.calls == ["SUB-1001"]


# ---------------------------------------------------------------------------
# API surface: the honest offline answer, rules intact
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def test_api_provider_pass_reports_unavailable_offline(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1001"])
    res = client.post("/guidelines/change-impact/provider", json={})
    assert res.status_code == 200
    body = res.json()
    assert body["rules"]["guidelines_version"]
    assert body["provider"]["available"] is False
    assert "offline" in body["provider"]["reason"]
