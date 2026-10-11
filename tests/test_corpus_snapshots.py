"""Corpus snapshots per version: the guideline text in force when a
submission triaged is kept — immutably — so attribution can quote
the line as it read then, and two versions can be diffed line by
line. A version names one text forever: re-registering it with
different text is an explicit conflict, never a silent overwrite.
"""

from __future__ import annotations

import copy

import pytest

from underwriting_agent.corpus import (
    CorpusVersionConflict,
    corpus_content_hash,
    diff_snapshots,
    get_snapshot,
    load_snapshots,
    register_snapshot,
)
from underwriting_agent.guidelines_data import GUIDELINES, GUIDELINES_VERSION
from underwriting_agent.retriage import guideline_change_impact
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore
from underwriting_agent.triage import classify_submission
from underwriting_agent.schemas import Triage

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}

OLD_WORDING = "review of cause and remediation"
CURRENT_WORDING = "review of cause, remediation, and current protection"


def _old_corpus() -> list[dict]:
    """The corpus as it read under an earlier version: UW-PROP-01's
    severity sentence once asked for a narrower review."""
    corpus = copy.deepcopy(GUIDELINES)
    for guideline in corpus:
        if guideline["id"] == "UW-PROP-01":
            guideline["text"] = guideline["text"].replace(
                CURRENT_WORDING, OLD_WORDING
            )
    return corpus


def _v2_corpus() -> list[dict]:
    """A later corpus: the property severity figure tightens."""
    corpus = copy.deepcopy(GUIDELINES)
    for guideline in corpus:
        if guideline["id"] == "UW-PROP-01":
            guideline["text"] = guideline["text"].replace(
                "single claim above 250,000", "single claim above 100,000"
            )
    return corpus


def make_service(store: InMemoryStore | None = None) -> SubmissionService:
    return SubmissionService(store=store or InMemoryStore(), checkpointer=False)


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Registration: immutable per version
# ---------------------------------------------------------------------------


def test_registration_keeps_the_full_text_under_its_version():
    store = InMemoryStore()
    snapshot, created = register_snapshot(store)
    assert created is True
    assert snapshot["version"] == GUIDELINES_VERSION
    assert snapshot["content_hash"] == corpus_content_hash(GUIDELINES)
    assert snapshot["guideline_count"] == len(GUIDELINES)
    kept = get_snapshot(store, GUIDELINES_VERSION)
    assert kept is not None
    prop = next(g for g in kept["guidelines"] if g["id"] == "UW-PROP-01")
    assert "single claim above 250,000" in prop["text"]


def test_reregistering_identical_text_is_an_idempotent_noop():
    store = InMemoryStore()
    first, _ = register_snapshot(store)
    second, created = register_snapshot(store)
    assert created is False
    assert second == first
    assert list(load_snapshots(store)) == [GUIDELINES_VERSION]


def test_same_version_with_different_text_is_an_explicit_conflict():
    store = InMemoryStore()
    register_snapshot(store)
    with pytest.raises(CorpusVersionConflict) as excinfo:
        register_snapshot(store, guidelines=_v2_corpus(), version=GUIDELINES_VERSION)
    assert "bump GUIDELINES_VERSION" in str(excinfo.value)
    # The kept snapshot is untouched by the refused registration.
    kept = get_snapshot(store, GUIDELINES_VERSION)
    assert kept["content_hash"] == corpus_content_hash(GUIDELINES)


def test_a_new_version_registers_beside_the_old():
    store = InMemoryStore()
    register_snapshot(store, guidelines=_old_corpus(), version="2026.09.0")
    register_snapshot(store)  # the in-force corpus
    assert sorted(load_snapshots(store)) == ["2026.09.0", GUIDELINES_VERSION]
    old = get_snapshot(store, "2026.09.0")
    prop = next(g for g in old["guidelines"] if g["id"] == "UW-PROP-01")
    assert OLD_WORDING in prop["text"]


# ---------------------------------------------------------------------------
# As it read then
# ---------------------------------------------------------------------------


def test_triage_still_quotes_v1_after_v2_registers():
    service = make_service()
    result = service.analyze(SAMPLES["SUB-1009"])  # property; cites UW-PROP-01
    assert result.guidelines_version == GUIDELINES_VERSION
    # A later corpus registers under its own version.
    service.register_corpus_snapshot(guidelines=_v2_corpus(), version="2026.11.0")

    report = service.guideline_as_triaged("SUB-1009", "UW-PROP-01")
    assert report["triaged_under"] == GUIDELINES_VERSION
    assert report["snapshot_kept"] is True
    (entry,) = report["guidelines"]
    assert entry["source"] == "snapshot"
    # The line as it read when SUB-1009 triaged — the 250,000 figure,
    # not the tightened 100,000 the newer snapshot carries.
    assert "single claim above 250,000" in entry["text"]
    assert "single claim above 100,000" not in entry["text"]


def test_as_triaged_without_a_kept_snapshot_falls_back_to_the_record():
    store = InMemoryStore()
    service = make_service(store)
    service.analyze(SAMPLES["SUB-1009"])
    record = store.get("SUB-1009")
    record.result.guidelines_version = "1999.01.0"  # never registered
    store.save(record)

    report = service.guideline_as_triaged("SUB-1009", "UW-PROP-01")
    assert report["triaged_under"] == "1999.01.0"
    assert report["snapshot_kept"] is False
    (entry,) = report["guidelines"]
    assert entry["source"] == "record"  # named as the record's own copy
    cited = next(
        g for g in record.result.guidelines if g.guideline_id == "UW-PROP-01"
    )
    assert entry["text"] == cited.text


def test_unknown_submission_reads_as_none():
    service = make_service()
    assert service.guideline_as_triaged("SUB-9999") is None


# ---------------------------------------------------------------------------
# Change-impact attribution quotes then beside now
# ---------------------------------------------------------------------------


def _severity_referral(sub, facts) -> Triage:
    """A simulated corpus change: severity now refers at any size."""
    triage = classify_submission(sub, facts)
    if triage.outcome == "in_appetite":
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


def test_a_move_quotes_the_line_as_it_read_then():
    store = InMemoryStore()
    service = make_service(store)
    service.register_corpus_snapshot(guidelines=_old_corpus(), version="2026.09.0")
    service.analyze(SAMPLES["SUB-1001"])
    record = store.get("SUB-1001")
    record.result.guidelines_version = "2026.09.0"  # triaged under the old text
    store.save(record)

    report = service.guideline_change_impact(classifier=_severity_referral)
    reason = report["moved"][0]["reasons"][0]
    assert reason["guideline_id"] == "UW-PROP-01"
    assert CURRENT_WORDING in reason["guideline_line"]  # as it reads now
    assert OLD_WORDING in reason["guideline_line_then"]  # as it read then
    assert reason["guideline_line_then"] != reason["guideline_line"]


def test_no_then_line_without_an_older_snapshot():
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    report = service.guideline_change_impact(classifier=_severity_referral)
    reason = report["moved"][0]["reasons"][0]
    # Triaged under the in-force version: today's line is the line
    # as it read then, and no second quote is invented for it.
    assert "guideline_line_then" not in reason


# ---------------------------------------------------------------------------
# The diff across versions
# ---------------------------------------------------------------------------


def test_diff_names_the_changed_line_exactly():
    store = InMemoryStore()
    register_snapshot(store)
    register_snapshot(store, guidelines=_v2_corpus(), version="2026.11.0")
    diff = diff_snapshots(
        get_snapshot(store, GUIDELINES_VERSION), get_snapshot(store, "2026.11.0")
    )
    assert diff["from_version"] == GUIDELINES_VERSION
    assert diff["to_version"] == "2026.11.0"
    assert diff["added"] == [] and diff["removed"] == []
    assert diff["unchanged_count"] == len(GUIDELINES) - 1
    (change,) = diff["changed"]
    assert change["guideline_id"] == "UW-PROP-01"
    assert any("single claim above 250,000" in line for line in change["removed_lines"])
    assert any("single claim above 100,000" in line for line in change["added_lines"])


def test_diff_names_added_and_removed_guidelines():
    store = InMemoryStore()
    register_snapshot(store)
    v3 = [g for g in copy.deepcopy(GUIDELINES) if g["id"] != "UW-MAR-01"]
    v3.append(
        {
            "id": "UW-EPL-01",
            "title": "Employment practices notes",
            "text": "Employment practices submissions require the EEO-1 component.",
            "lines": ["employment_practices"],
        }
    )
    register_snapshot(store, guidelines=v3, version="2026.12.0")
    diff = diff_snapshots(
        get_snapshot(store, GUIDELINES_VERSION), get_snapshot(store, "2026.12.0")
    )
    assert diff["added"] == ["UW-EPL-01"]
    assert diff["removed"] == ["UW-MAR-01"]
    assert diff["changed"] == []


def test_service_diff_is_none_when_a_side_is_missing():
    service = make_service()
    assert service.corpus_diff(GUIDELINES_VERSION, "2027.01.0") is None


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_api_snapshot_flow(client):
    res = client.post("/guidelines/snapshots")
    assert res.status_code == 200
    body = res.json()
    assert body["version"] == GUIDELINES_VERSION and body["created"] is True

    res = client.post("/guidelines/snapshots")
    assert res.json()["created"] is False  # idempotent, not an error

    res = client.get("/guidelines/snapshots")
    listing = res.json()
    assert listing["current_version"] == GUIDELINES_VERSION
    assert [s["version"] for s in listing["snapshots"]] == [GUIDELINES_VERSION]

    res = client.get(f"/guidelines/snapshots/{GUIDELINES_VERSION}")
    assert res.status_code == 200
    assert res.json()["guideline_count"] == len(GUIDELINES)

    res = client.get("/guidelines/snapshots/1999.01.0")
    assert res.status_code == 404


def test_api_diff_and_as_triaged(client):
    client.post("/submissions/analyze", json=SAMPLES["SUB-1009"])
    client.post("/guidelines/snapshots")

    res = client.get(
        "/guidelines/snapshots/diff",
        params={"from_version": GUIDELINES_VERSION, "to_version": GUIDELINES_VERSION},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["changed"] == [] and body["unchanged_count"] == len(GUIDELINES)

    res = client.get(
        "/guidelines/snapshots/diff",
        params={"from_version": GUIDELINES_VERSION, "to_version": "2027.01.0"},
    )
    assert res.status_code == 404

    res = client.get("/submissions/SUB-1009/guidelines-as-triaged")
    assert res.status_code == 200
    body = res.json()
    assert body["triaged_under"] == GUIDELINES_VERSION
    assert body["snapshot_kept"] is True
    assert all(e["source"] == "snapshot" for e in body["guidelines"])

    res = client.get(
        "/submissions/SUB-1009/guidelines-as-triaged",
        params={"guideline_id": "UW-PROP-01"},
    )
    entries = res.json()["guidelines"]
    assert [e["guideline_id"] for e in entries] == ["UW-PROP-01"]

    res = client.get("/submissions/SUB-9999/guidelines-as-triaged")
    assert res.status_code == 404


def test_change_impact_still_reports_through_the_service():
    # The extended service read keeps the rules report's shape.
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    report = guideline_change_impact(
        service._get_store().all_records(), classifier=_severity_referral
    )
    assert report["moved_count"] == 1
