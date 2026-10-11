"""Cross-broker duplicate detection: the same risk, submitted twice.

The seeded corpus carries a designed duplicate pair — SUB-1015
(Kestrel Ridge Vineyards LLC, via Harborline) and SUB-1016
(Kestrel Ridge Vineyard, LLC, via Meridian Wholesale) — and a
near-miss, SUB-1017, which shares SUB-1009's class, line, and
effective date but is a different insured. The pair must flag
with its evidence; the near-miss must not reach the queue; and
detection must never touch either submission.
"""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.api import app
from underwriting_agent.duplicates import (
    name_similarity,
    normalize_insured_name,
    pair_key,
)
from underwriting_agent.samples import sample_submission_models
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, _record_to_dict

SAMPLES = {s.submission_id: s for s in sample_submission_models()}

DESIGNED_PAIR = pair_key("SUB-1015", "SUB-1016")
RENEWAL_PAIR = pair_key("SUB-1001", "SUB-1013")
NEAR_MISS_PAIR = pair_key("SUB-1009", "SUB-1017")


def make_service(*submission_ids: str) -> SubmissionService:
    service = SubmissionService(store=InMemoryStore(), checkpointer=False)
    for submission_id in submission_ids:
        service.analyze(SAMPLES[submission_id])
    return service


def seeded_service() -> SubmissionService:
    return make_service(
        "SUB-1001", "SUB-1013", "SUB-1009",
        "SUB-1015", "SUB-1016", "SUB-1017",
    )


# ---------------------------------------------------------------------------
# Name matching (deterministic normalisation, no model)
# ---------------------------------------------------------------------------


def test_name_normalisation_sets_aside_suffixes_punctuation_and_order():
    assert normalize_insured_name("Kestrel Ridge Vineyards LLC") == normalize_insured_name(
        "Vineyards, Kestrel Ridge"
    )
    assert name_similarity("Acme Widgets Inc", "Acme Widgets, LLC") == 1.0
    assert name_similarity("The Harborlight Marine Services LLC", "Harborlight Marine Services") == 1.0


def test_name_similarity_ranks_variants_above_strangers():
    variant = name_similarity("Kestrel Ridge Vineyards LLC", "Kestrel Ridge Vineyard, LLC")
    stranger = name_similarity("Elmwood Self Storage LP", "Alder Creek Storage Partners LLC")
    assert variant >= 0.9
    assert stranger < 0.6  # below the name gate


# ---------------------------------------------------------------------------
# The match pass
# ---------------------------------------------------------------------------


def test_designed_cross_broker_pair_is_flagged_with_evidence():
    report = seeded_service().duplicate_findings()
    findings = {f["pair_key"]: f for f in report["findings"]}
    assert DESIGNED_PAIR in findings
    finding = findings[DESIGNED_PAIR]
    assert finding["status"] == "open"
    assert finding["score"] >= report["policy"]["threshold"]
    assert set(finding["submission_ids"]) == {"SUB-1015", "SUB-1016"}
    # The two submissions arrived through different brokers — the
    # context says so, because that is the shape of the problem.
    assert finding["context"]["different_brokers"] is True
    assert {finding["context"]["broker_left"], finding["context"]["broker_right"]} == {
        "Harborline Brokers",
        "Meridian Wholesale Brokers",
    }
    # Field-by-field evidence: every scored field, both values,
    # its similarity, its weight, and the points it contributed.
    evidence = {e["field"]: e for e in finding["evidence"]}
    assert set(evidence) == {
        "insured_name", "class_of_business", "line_of_business",
        "effective_date", "revenue", "requested_limit",
    }
    assert evidence["insured_name"]["left"] == "Kestrel Ridge Vineyards LLC"
    assert evidence["insured_name"]["right"] == "Kestrel Ridge Vineyard, LLC"
    assert evidence["insured_name"]["similarity"] >= 0.9
    assert evidence["class_of_business"]["similarity"] == 1.0
    assert evidence["revenue"]["similarity"] >= 0.95
    points = sum(e["points"] for e in evidence.values())
    assert points == pytest.approx(finding["score"], abs=0.001)


def test_near_miss_pair_never_reaches_the_queue():
    report = seeded_service().duplicate_findings()
    keys = [f["pair_key"] for f in report["findings"]]
    assert NEAR_MISS_PAIR not in keys


def test_no_self_pairs_and_each_pair_once():
    report = seeded_service().duplicate_findings()
    keys = [f["pair_key"] for f in report["findings"]]
    assert len(keys) == len(set(keys))
    for finding in report["findings"]:
        left, right = finding["submission_ids"]
        assert left != right


def test_detection_alters_neither_submission():
    service = seeded_service()
    store = service._get_store()
    before = {
        sid: copy.deepcopy(_record_to_dict(store.get(sid)))
        for sid in ("SUB-1015", "SUB-1016")
    }
    service.duplicate_findings()
    service.resolve_duplicate(
        DESIGNED_PAIR, actor="U. Writer", decision="confirmed",
        reason="Same winery, two brokers — one risk.",
    )
    after = {
        sid: _record_to_dict(store.get(sid)) for sid in ("SUB-1015", "SUB-1016")
    }
    assert before == after


# ---------------------------------------------------------------------------
# Human resolution: confirm / dismiss, remembered per pair
# ---------------------------------------------------------------------------


def test_confirm_is_recorded_with_actor_and_reason():
    service = seeded_service()
    finding = service.resolve_duplicate(
        DESIGNED_PAIR, actor="U. Writer", decision="confirmed",
        reason="Same winery, two brokers — one risk.",
    )
    assert finding["status"] == "confirmed"
    assert finding["resolution"]["actor"] == "U. Writer"
    assert finding["resolution"]["reason"] == "Same winery, two brokers — one risk."
    assert finding["resolution"]["decided_at"]
    report = service.duplicate_findings()
    assert report["counts"] == {"open": 1, "confirmed": 1, "dismissed": 0}


def test_dismissal_is_remembered_and_the_pair_stays_out_of_the_open_queue():
    service = seeded_service()
    # The Northgate pair is a renewal, not a duplicate — the
    # classic dismissal, and the reason says which.
    service.resolve_duplicate(
        RENEWAL_PAIR, actor="S. Senior", decision="dismissed",
        reason="Renewal of SUB-1001, not a second submission of the risk.",
    )
    for _ in range(2):  # re-running detection never re-opens it
        report = service.duplicate_findings()
        by_key = {f["pair_key"]: f for f in report["findings"]}
        assert by_key[RENEWAL_PAIR]["status"] == "dismissed"
        assert by_key[RENEWAL_PAIR]["resolution"]["actor"] == "S. Senior"
    assert report["counts"]["open"] == 1  # only the Kestrel pair remains open


def test_resolution_requires_actor_reason_and_a_real_pair():
    service = seeded_service()
    with pytest.raises(ValueError, match="actor"):
        service.resolve_duplicate(
            DESIGNED_PAIR, actor="", decision="confirmed", reason="x"
        )
    with pytest.raises(ValueError, match="reason"):
        service.resolve_duplicate(
            DESIGNED_PAIR, actor="U. Writer", decision="dismissed", reason=" "
        )
    with pytest.raises(ValueError, match="verdict"):
        service.resolve_duplicate(
            DESIGNED_PAIR, actor="U. Writer", decision="merged", reason="x"
        )
    with pytest.raises(KeyError):
        service.resolve_duplicate(
            "SUB-1002::SUB-1003", actor="U. Writer",
            decision="confirmed", reason="x",
        )


def test_changing_a_verdict_keeps_the_earlier_one_in_history():
    service = seeded_service()
    service.resolve_duplicate(
        DESIGNED_PAIR, actor="U. Writer", decision="dismissed",
        reason="Looked like a renewal on first read.",
    )
    finding = service.resolve_duplicate(
        DESIGNED_PAIR, actor="S. Senior", decision="confirmed",
        reason="Broker confirmed both brokers were appointed — one risk.",
    )
    assert finding["status"] == "confirmed"
    history = finding["resolution"]["history"]
    assert len(history) == 1
    assert history[0]["status"] == "dismissed"
    assert history[0]["actor"] == "U. Writer"
    # Re-resolving with the standing verdict is a no-op read.
    again = service.resolve_duplicate(
        DESIGNED_PAIR, actor="S. Senior", decision="confirmed",
        reason="Broker confirmed both brokers were appointed — one risk.",
    )
    assert again["resolution"]["history"] == history


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINTS", "off")
    # Rebuild the module-level service against the in-memory
    # doubles (the test_api.py pattern), then analyze the four
    # relevant samples through the API itself.
    import underwriting_agent.api as api_module
    from underwriting_agent.samples import load_sample_submissions
    from underwriting_agent.wiring import build_service_from_env

    api_module.service = build_service_from_env()
    test_client = TestClient(app)
    raw = {s["submission_id"]: s for s in load_sample_submissions()}
    for submission_id in ("SUB-1009", "SUB-1015", "SUB-1016", "SUB-1017"):
        res = test_client.post("/submissions/analyze", json=raw[submission_id])
        assert res.status_code == 200
    return test_client


def test_duplicates_endpoints(client):
    report = client.get("/duplicates").json()
    keys = [f["pair_key"] for f in report["findings"]]
    assert DESIGNED_PAIR in keys
    assert NEAR_MISS_PAIR not in keys

    resolved = client.post(
        "/duplicates/resolve",
        json={
            "pair_key": DESIGNED_PAIR, "actor": "U. Writer",
            "decision": "dismissed",
            "reason": "Two locations of one insured, separately insured.",
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "dismissed"

    missing_reason = client.post(
        "/duplicates/resolve",
        json={"pair_key": DESIGNED_PAIR, "actor": "U. Writer",
              "decision": "confirmed", "reason": ""},
    )
    assert missing_reason.status_code == 422

    unknown = client.post(
        "/duplicates/resolve",
        json={"pair_key": "SUB-1002::SUB-1003", "actor": "U. Writer",
              "decision": "confirmed", "reason": "x"},
    )
    assert unknown.status_code == 404
