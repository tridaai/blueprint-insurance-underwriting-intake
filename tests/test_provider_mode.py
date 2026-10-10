"""Provider-mode behaviour with fake LLM backends (no network).

A fake backend implements the optional capability methods the graph
discovers with getattr — the same seam a real provider backend uses —
so these tests exercise the LLM paths end to end: extraction,
cross-check, agentic diagnosis, options, verify, review, repair, and
the token-budget degradation.
"""

from __future__ import annotations

from underwriting_agent.graph import run_submission
from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.samples import sample_submission_models

SAMPLES = {s.submission_id: s for s in sample_submission_models()}


class FakeLLM:
    """A cooperative fake: agrees with the rules, composes clean drafts."""

    name = "fake-llm"
    _model = "fake-1"

    def __init__(self) -> None:
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def usage_totals(self):
        return dict(self._usage)

    def reset_usage(self):
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def _tick(self, tokens=100):
        self._usage["calls"] += 1
        self._usage["input_tokens"] += tokens
        self._usage["output_tokens"] += tokens // 2

    def draft_documents(self, context):
        self._tick()
        if context.get("repair_feedback"):
            # The repaired draft drops the offending quote entirely.
            letter = (
                f"Dear {context['broker_name']} team,\n\nSubmission "
                f"{context['submission_id']} received. Next step: underwriter "
                "review. This letter is not a quotation and binds nothing."
            )
            return f"Submission {context['submission_id']}", letter, context_packet(context)
        return MockModelBackend().draft_documents(context)

    def classify_with_llm(self, context):
        self._tick()
        return {
            "outcome": context["outcome"] if "outcome" in context else "in_appetite",
            "risk_tier": "low",
            "confidence": 0.9,
            "rationale": "fake agreement",
            "backend": self.name,
        }

    def diagnose(self, context):
        self._tick()
        return {"summary": "Fake LLM risk summary.", "risk_factors": ["fake factor"]}

    def propose_options(self, context):
        self._tick()
        return [
            {"kind": "refer_to_senior_underwriter", "title": "Refer it", "description": "Senior eyes."},
            {"kind": "proceed_to_quote_prep", "title": "Proceed", "description": "Move on."},
        ]

    def verify_draft(self, context):
        self._tick()
        return {"grounded": True, "issues": [], "summary": "fake grounded"}

    def review_draft(self, context):
        self._tick()
        return {"verdict": "pass", "findings": [], "model": "fake-1"}


def context_packet(context) -> str:
    return (
        f"UNDERWRITER PACKET — submission {context['submission_id']}. "
        "Risk summary present. No premium is quoted and no coverage is bound."
    )


def test_fake_llm_paths_all_engage():
    result = run_submission(SAMPLES["SUB-1001"], backend=FakeLLM())
    assert result.cross_check is not None
    assert result.cross_check.resolution in ("agree", "rules_authoritative", "rules_only")
    assert result.diagnosis.source == "llm"
    assert result.diagnosis.summary == "Fake LLM risk summary."
    assert result.verification is not None and result.verification.source == "llm"
    assert result.review is not None and result.review.source == "llm"
    assert result.telemetry is not None
    assert result.telemetry.input_tokens is not None  # a (fake) model ran
    assert result.telemetry.model_calls > 1


def test_repair_loop_fixes_a_failing_llm_draft():
    # SUB-1011 with the cooperative fake: the first draft inherits the
    # bind trap (mock template), the repair redraft drops it, and the
    # run ends valid with the repair recorded.
    result = run_submission(SAMPLES["SUB-1011"], backend=FakeLLM())
    assert result.repair_attempted
    assert result.repaired
    assert result.validation.passed
    assert result.original_validation is not None
    assert not result.original_validation.passed


def test_repair_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_REPAIR", "off")
    result = run_submission(SAMPLES["SUB-1011"], backend=FakeLLM())
    assert not result.repair_attempted
    assert not result.validation.passed


def test_tool_loop_diagnosis_engages_and_logs_calls():
    class ToolFake(FakeLLM):
        def diagnose_with_tools(self, context, tool_specs, dispatch, max_tool_calls):
            self._tick()
            text, summary = dispatch("submission_facts", {})
            assert "triage:" in text
            text2, summary2 = dispatch("search_guidelines", {"query": "cyber appetite"})
            return {
                "summary": "Tool-composed summary.",
                "risk_factors": ["from tools"],
                "tool_calls": [
                    {"name": "submission_facts", "summary": summary},
                    {"name": "search_guidelines", "summary": summary2},
                ],
            }

    result = run_submission(SAMPLES["SUB-1001"], backend=ToolFake())
    assert result.diagnosis.source == "llm"
    assert len(result.diagnosis.tool_calls) == 2
    trace_text = " ".join(d for s in result.trace for d in s.details)
    assert "submission_facts" in trace_text


def test_broken_llm_degrades_everywhere_without_failing():
    class Boom(FakeLLM):
        def classify_with_llm(self, context):
            raise RuntimeError("provider exploded")

        def diagnose(self, context):
            raise RuntimeError("provider exploded")

        def propose_options(self, context):
            raise RuntimeError("provider exploded")

        def verify_draft(self, context):
            raise RuntimeError("provider exploded")

        def review_draft(self, context):
            raise RuntimeError("provider exploded")

    result = run_submission(SAMPLES["SUB-1001"], backend=Boom())
    assert result.triage.outcome == "in_appetite"  # rules carried it
    assert result.cross_check is not None and result.cross_check.resolution == "rules_only"
    assert result.diagnosis.source == "template"
    assert result.verification.source == "checklist"
    assert result.review.source == "checklist"
    assert result.validation.passed


def test_token_budget_degrades_provider_steps(monkeypatch):
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "10")  # tiny: the first calls exhaust it
    result = run_submission(SAMPLES["SUB-1001"], backend=FakeLLM())
    assert result.telemetry is not None and result.telemetry.budget is not None
    assert result.telemetry.budget["exceeded"]
    # The run still completes end to end on deterministic paths.
    assert result.status == "awaiting_underwriter"
    assert result.drafts is not None


def test_reviewer_block_flags_the_run():
    class BlockingReviewer(FakeLLM):
        def review_draft(self, context):
            self._tick()
            return {"verdict": "block", "findings": ["packet is missing the referral rationale"], "model": "fake-1"}

    result = run_submission(SAMPLES["SUB-1001"], backend=BlockingReviewer())
    assert result.reviewer_blocked
    assert not result.autonomy.eligible_for_auto_triage


def test_reviewer_can_be_disabled(monkeypatch):
    monkeypatch.setenv("REVIEWER", "off")
    result = run_submission(SAMPLES["SUB-1001"], backend=FakeLLM())
    assert result.review is not None and not result.review.enabled
    assert not result.reviewer_blocked
