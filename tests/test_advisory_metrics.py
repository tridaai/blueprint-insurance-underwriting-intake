"""Round 15: the letter-advisory trend in /metrics.

Since round 14 every run records its broker letter's borderline
phrasings (``letter_advisories`` — forms the blocking guardrails
deliberately miss but a judge still reads as commitments) and
whether the one steered redraft cleared them
(``advisory_repaired``). The records existed; nothing aggregated
them, so a drafting-quality regression — a model update quietly
producing more near-commitment letters — was visible only run by
run. The metrics now carry the family: runs with advisories,
the total, the per-run rate, the family mix, and the repair
clear rate, the way #3 watches forecast accuracy: a number the
team sees move before it becomes guardrail failures.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.insights import compute_metrics
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore


def _record(
    sid: str,
    advisories: list[dict] | None = None,
    *,
    repaired: bool = False,
) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            letter_advisories=advisories or [],
            advisory_repaired=repaired,
        ),
        submission={"insured_name": f"Insured {sid}"},
    )


def _advisory(family: str) -> dict:
    return {
        "family": family,
        "phrase": "will now move through review",
        "sentence": "The file will now move through review.",
        "problem": "a forward-motion commitment",
        "instead": "The submission is under review.",
    }


def test_the_trend_aggregates_runs_families_and_repairs():
    records = [
        _record(
            "A",
            [_advisory("forward_motion"), _advisory("soft_time")],
            repaired=True,
        ),
        _record("B", [_advisory("soft_time")]),
        _record("C"),
        _record(
            "D",
            [
                _advisory("forward_motion"),
                _advisory("forward_motion"),
                _advisory("review_commitment"),
            ],
            repaired=True,
        ),
    ]
    trend = compute_metrics(records)["letter_advisories"]
    assert trend["runs_with_advisories"] == 3
    assert trend["total"] == 6
    assert trend["per_run"] == 1.5
    assert trend["by_family"] == {
        "forward_motion": 3,
        "review_commitment": 1,
        "soft_time": 2,
    }
    assert trend["repaired_runs"] == 2
    assert trend["clear_rate"] == round(2 / 3, 3)


def test_the_trend_is_quiet_and_honest_on_an_empty_store():
    trend = compute_metrics([])["letter_advisories"]
    assert trend == {
        "runs_with_advisories": 0,
        "total": 0,
        "per_run": None,
        "by_family": {},
        "repaired_runs": 0,
        "clear_rate": None,
    }


def test_no_advisories_means_no_clear_rate_to_report():
    trend = compute_metrics([_record("A"), _record("B")])["letter_advisories"]
    assert trend["runs_with_advisories"] == 0
    assert trend["per_run"] == 0.0
    assert trend["clear_rate"] is None


def test_an_advisory_without_a_family_counts_as_unknown():
    # Older stored payloads predate the family field's
    # guarantee; the aggregate names the gap instead of
    # dropping the advisory.
    trend = compute_metrics([_record("A", [{"phrase": "be in touch"}])])[
        "letter_advisories"
    ]
    assert trend["by_family"] == {"unknown": 1}
    assert trend["total"] == 1


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    store = InMemoryStore()
    store.save(_record("A", [_advisory("soft_time")], repaired=True))
    store.save(_record("B"))
    api_module.service = SubmissionService(store=store, checkpointer=False)
    return TestClient(api_module.app)


def test_the_metrics_endpoint_carries_the_trend(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    trend = res.json()["letter_advisories"]
    assert trend["runs_with_advisories"] == 1
    assert trend["by_family"] == {"soft_time": 1}
    assert trend["clear_rate"] == 1.0
