"""Line-level information requests for bundles.

A bundle's gaps live on lines. When one line of a bundle is
incomplete, the broker is asked for that line's missing items —
under that line's heading, in the letter and in the information
request — the gate record names the blocking line(s), and a
complete line is never implied to owe anything.
"""

from __future__ import annotations

import copy

import pytest

from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def gapped_bundle() -> dict:
    """SUB-1014 with the cyber line swapped for a D&O line that is
    missing its financials: the bundle is incomplete for exactly
    one line, and the property line owes nothing."""
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


def test_the_gate_record_names_the_blocking_line():
    result = make_service().analyze(gapped_bundle())
    assert result.triage.outcome == "incomplete"  # driven by the D&O line
    assert result.blocking_lines == ["directors_and_officers"]
    by_line = {a.line_of_business: a.result for a in result.line_analyses}
    assert by_line["property"].triage.outcome == "in_appetite"
    assert by_line["directors_and_officers"].triage.outcome == "incomplete"


def test_the_information_request_is_per_line():
    result = make_service().analyze(gapped_bundle())
    info = result.information_request
    assert info is not None and info.needed
    assert len(info.line_requests) == 1
    line_request = info.line_requests[0]
    assert line_request.line_of_business == "directors_and_officers"
    assert line_request.submission_id == "SUB-1016-DIRECTORS-AND-OFFICERS"
    assert line_request.missing_items == [
        "the last two years of financial statements"
    ]
    # The flat list is the same union — nothing from the property line.
    assert info.missing_items == line_request.missing_items
    assert "For the directors_and_officers line:" in info.message
    assert "For the property line:" not in info.message
    assert "Lines not listed above have everything they need." in info.message


def test_the_broker_letter_asks_only_the_gapped_line():
    result = make_service().analyze(gapped_bundle())
    letter = result.drafts.broker_letter
    assert "Still needed for the directors_and_officers line:" in letter
    assert "the last two years of financial statements" in letter
    assert "Still needed for the property line" not in letter
    # The letter still passes the guardrails as a whole.
    assert result.validation is not None and result.validation.passed


def test_the_packet_names_the_blocking_line_and_its_gap():
    result = make_service().analyze(gapped_bundle())
    packet = result.drafts.underwriter_packet
    assert (
        "Lines blocking readiness: directors_and_officers "
        "(missing: financials)" in packet
    )


def test_a_complete_bundle_blocks_nothing_and_asks_nothing():
    result = make_service().analyze(SAMPLES["SUB-1014"])
    assert result.blocking_lines == []
    assert result.information_request is None
    assert "Still needed" not in result.drafts.broker_letter
    assert "Lines blocking readiness" not in result.drafts.underwriter_packet


def test_blocking_lines_ride_the_api(client):
    posted = client.post("/submissions/analyze", json=gapped_bundle())
    assert posted.status_code == 200
    body = posted.json()
    assert body["blocking_lines"] == ["directors_and_officers"]
    assert body["information_request"]["line_requests"][0][
        "line_of_business"
    ] == "directors_and_officers"
