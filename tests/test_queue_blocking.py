"""Round 7: blocking-line age in the queue view.

A bundle stranded on one line reads differently from a wholly
unready submission: queue items for bundles name the blocking
lines (round 6 records them on the gate) and how long the bundle
has waited on each — the age runs from the bundle's submission,
which is when the wait on that line started.
"""

from __future__ import annotations

from datetime import datetime, timezone

from underwriting_agent.insights import queue_aging
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import AgentResult, LineAnalysis, SubmissionFacts, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def _record(sid: str, submitted_at: str, **result_kwargs) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="awaiting_underwriter",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier="moderate"),
            **result_kwargs,
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": submitted_at,
        },
    )


def _items(view: dict) -> dict:
    return {
        item["submission_id"]: item
        for bucket in view["buckets"]
        for item in bucket["submissions"]
    }


def _gapped_bundle() -> dict:
    """SUB-1014 with the cyber line swapped for a D&O line missing
    its financials (the round-6 construction): exactly one line
    blocks the bundle."""
    import copy

    bundle = copy.deepcopy(SAMPLES["SUB-1014"])
    bundle["submission_id"] = "SUB-1016"
    bundle["lines"][1] = {
        "line_of_business": "directors_and_officers",
        "class_of_business": "hospitality management company",
        "documents": [
            {
                "doc_type": "loss runs",
                "document_id": "sub1016-dno-lr",
                "fields": {"loss_history_present": "yes"},
                "raw_text": "Loss runs 2023-2026: no claims reported in the period.",
            }
        ],
        "losses": [],
        "requested_limit": "2000000",
        "requested_deductible": "25000",
    }
    return bundle


def test_bundle_queue_item_names_blocking_lines_with_age():
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    result = service.analyze(_gapped_bundle())
    assert result.multi_line and result.blocking_lines == ["directors_and_officers"]
    view = service.queue()
    assert view["blocked_bundles"] == 1
    item = _items(view)["SUB-1016"]
    blocking = {b["line_of_business"]: b for b in item["blocking_lines"]}
    assert set(blocking) == {"directors_and_officers"}
    entry = blocking["directors_and_officers"]
    # The bundle has waited on the line since it was submitted:
    # the line's age is the bundle's queue age.
    assert entry["age_days"] == item["age_days"]
    assert entry["missing_items"]  # what to chase, per line
    assert entry["submission_id"].startswith("SUB-1016-")


def test_blocking_lines_empty_for_single_line_and_complete():
    line_result = AgentResult(
        submission_id="B1-PROPERTY",
        insured_name="Insured B1",
        facts=SubmissionFacts(readiness="ready", completeness_score=1.0),
        triage=Triage(outcome="in_appetite", risk_tier="low"),
    )
    bundle = _record(
        "B1",
        "2026-10-09",
        multi_line=True,
        blocking_lines=[],
        line_analyses=[
            LineAnalysis(
                line_of_business="property",
                submission_id="B1-PROPERTY",
                result=line_result,
            )
        ],
    )
    view = queue_aging([bundle], now=NOW)
    assert _items(view)["B1"]["blocking_lines"] == []
    single = queue_aging([_record("S1", "2026-10-09")], now=NOW)
    assert _items(single)["S1"]["blocking_lines"] == []
