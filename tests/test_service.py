"""Service layer: decisions, the feedback loop, memory, batch, webhook."""

from __future__ import annotations

import pytest

from underwriting_agent.samples import sample_submission_models
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s.submission_id: s for s in sample_submission_models()}


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def test_analyze_then_approve_records_decision():
    service = make_service()
    result = service.analyze(SAMPLES["SUB-1001"])
    assert result.status == "awaiting_underwriter"
    decided = service.approve("SUB-1001", approver="U. Writer", reason="clean file")
    assert decided.status == "approved"
    assert decided.decision == "approved"
    assert decided.decided_by == "U. Writer"
    assert decided.decision_reason == "clean file"
    assert decided.external_action_taken is False  # no webhook configured
    assert decided.dispatch_status is None


def test_approve_unknown_submission_raises():
    with pytest.raises(KeyError):
        make_service().approve("SUB-9999", approver="X")


def test_double_decision_raises():
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U")
    with pytest.raises(ValueError):
        service.approve("SUB-1001", approver="U")


def test_reject_records_decision():
    service = make_service()
    service.analyze(SAMPLES["SUB-1002"])
    decided = service.reject("SUB-1002", reviewer="S. Senior", reason="loss history too heavy")
    assert decided.status == "rejected"
    assert decided.decided_by == "S. Senior"


def test_failed_validation_cannot_be_approved():
    service = make_service()
    result = service.analyze(SAMPLES["SUB-1011"])  # bind trap: validation fails
    assert result.validation is not None and not result.validation.passed
    with pytest.raises(ValueError, match="guardrail"):
        service.approve("SUB-1011", approver="U")
    # …but it can still be rejected (the safe direction is never blocked).
    decided = service.reject("SUB-1011", reviewer="U", reason="draft unsafe")
    assert decided.status == "rejected"


def test_memory_second_submission_sees_the_first():
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    again = SAMPLES["SUB-1001"].model_copy(update={"submission_id": "SUB-2001"})
    result = service.analyze(again)
    evidence = " ".join(result.diagnosis.evidence)
    assert "prior submission(s) for this insured" in evidence


def test_feedback_loop_teaches_later_runs():
    service = make_service()
    service.analyze(SAMPLES["SUB-1002"])
    service.reject("SUB-1002", reviewer="S. Senior", reason="water damage controls unverified")
    follow_up = SAMPLES["SUB-1002"].model_copy(update={"submission_id": "SUB-2002"})
    result = service.analyze(follow_up)
    evidence = " ".join(result.diagnosis.evidence)
    assert "underwriter feedback" in evidence
    assert "water damage controls unverified" in evidence


def test_batch_isolates_a_bad_item():
    service = make_service()
    items = service.analyze_batch(
        [SAMPLES["SUB-1001"], {"submission_id": 123, "documents": "nope"}, SAMPLES["SUB-1009"]]
    )
    assert items[0].result is not None
    assert items[1].error is not None
    assert items[2].result is not None


def test_batch_concurrent_matches_sequential_outcomes():
    service = make_service()
    subs = [SAMPLES[k] for k in ("SUB-1001", "SUB-1002", "SUB-1009", "SUB-1012")]
    items = service.analyze_batch(subs, concurrency=3)
    assert [i.result.triage.outcome for i in items] == [
        "in_appetite",
        "borderline",
        "in_appetite",
        "borderline",
    ]


def test_approval_webhook_dispatch_success(monkeypatch):
    import underwriting_agent.service as service_module

    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["payload"] = request.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    decided = service.approve("SUB-1001", approver="U")
    assert decided.dispatch_status == "sent"
    assert decided.external_action_taken is True
    assert "SUB-1001" in captured["payload"]
    # The webhook carries the packet; it quotes no premium figure.
    assert "premium is $" not in captured["payload"].lower()


def test_approval_webhook_failure_keeps_the_approval(monkeypatch):
    import underwriting_agent.service as service_module

    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", boom)
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    decided = service.approve("SUB-1001", approver="U")
    assert decided.status == "approved"
    assert decided.dispatch_status == "failed"
    assert decided.external_action_taken is False
