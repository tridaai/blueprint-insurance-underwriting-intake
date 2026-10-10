"""Round 23: citation repair through the rename log.

Round 21 freed a renamed label at once and pinned its
citations to an honest 404 — the right answer for a debate
that should update its citations, and a dead end for every
reader following one written before the move. The by-label
read now consults the rename history on a miss
(``proposals.calibration_snapshot_successor``): a label that
was renamed away answers with a **successor read** — the
report that now descends from the cited name, the label it
wears now, and the chain of renames between — a redirect in
data that names the move instead of pretending the old
address still names the report. The 404 survives exactly
where it belongs: a label that names nothing and never did,
and a label whose report a prune has removed (the rename
log's pruned stratum names that fate; there is no report
left to descend to). The repair is a by-label-read act only —
the diff's resolve step keeps refusing a stale label, so a
comparison is never silently re-pointed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from underwriting_agent.proposals import calibration_snapshot_successor
from underwriting_agent.schemas import AgentResult, Triage
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

BASE = date(2026, 9, 1)


def _decided(sid: str, tier: str, wait_days: int) -> ApprovalRecord:
    return ApprovalRecord(
        result=AgentResult(
            submission_id=sid,
            status="approved",
            insured_name=f"Insured {sid}",
            triage=Triage(outcome="borderline", risk_tier=tier),
            decided_at=(BASE + timedelta(days=wait_days)).isoformat(),
        ),
        submission={
            "insured_name": f"Insured {sid}",
            "broker_name": "Broker Q",
            "submitted_at": BASE.isoformat(),
        },
    )


def _service() -> SubmissionService:
    store = InMemoryStore()
    store.save(_decided("M-2", "moderate", 2))
    return SubmissionService(store=store, checkpointer=False)


def _renamed_twice() -> SubmissionService:
    """One kept report: pre-change → pre-budget-change →
    final-baseline."""
    service = _service()
    service.snapshot_sla_calibration("Ops Lead", label="pre-change")
    service.rename_sla_calibration_snapshot_label(
        "pre-change", "pre-budget-change", "Uma"
    )
    service.rename_sla_calibration_snapshot_label(
        "pre-budget-change", "final-baseline", "Ops Lead"
    )
    return service


# --- the pure chain-walk -------------------------------------------------------


def _snap(sid: str, label: str, history: list[dict]) -> dict:
    return {
        "snapshot_id": sid,
        "label": label,
        "label_history": history,
    }


def _move(frm: str, to: str, at: str, by: str = "Uma") -> dict:
    return {"from": frm, "to": to, "renamed_by": by, "renamed_at": at}


def test_the_chain_is_followed_whole_to_the_current_label():
    snapshots = [
        _snap(
            "cal-0001",
            "final-baseline",
            [
                _move("pre-change", "pre-budget-change",
                      "2026-10-03T09:00:00+00:00"),
                _move("pre-budget-change", "final-baseline",
                      "2026-10-04T09:00:00+00:00", by="Ops Lead"),
            ],
        )
    ]
    read = calibration_snapshot_successor(snapshots, "pre-change")
    assert read["snapshot"]["snapshot_id"] == "cal-0001"
    assert read["cited_label"] == "pre-change"
    assert read["current_label"] == "final-baseline"
    assert [m["from"] for m in read["rename_chain"]] == [
        "pre-change",
        "pre-budget-change",
    ]


def test_a_citation_of_the_middle_of_a_chain_walks_only_the_rest():
    snapshots = [
        _snap(
            "cal-0001",
            "final-baseline",
            [
                _move("pre-change", "pre-budget-change",
                      "2026-10-03T09:00:00+00:00"),
                _move("pre-budget-change", "final-baseline",
                      "2026-10-04T09:00:00+00:00"),
            ],
        )
    ]
    read = calibration_snapshot_successor(snapshots, "pre-budget-change")
    assert read["current_label"] == "final-baseline"
    assert len(read["rename_chain"]) == 1


def test_a_name_never_renamed_has_no_successor():
    snapshots = [_snap("cal-0001", "may", [])]
    assert calibration_snapshot_successor(snapshots, "nope") is None
    assert calibration_snapshot_successor(snapshots, "") is None
    assert calibration_snapshot_successor([], "may") is None


def test_a_retaken_name_descends_to_its_most_recent_fate():
    # "baseline" was freed by cal-0001's rename, taken by a
    # later keep, and renamed again — two chains start at the
    # cited name, and the later fate is the citation's answer.
    snapshots = [
        _snap(
            "cal-0001",
            "first-baseline",
            [_move("baseline", "first-baseline",
                   "2026-10-03T09:00:00+00:00")],
        ),
        _snap(
            "cal-0002",
            "second-baseline",
            [_move("baseline", "second-baseline",
                   "2026-10-05T09:00:00+00:00")],
        ),
    ]
    read = calibration_snapshot_successor(snapshots, "baseline")
    assert read["snapshot"]["snapshot_id"] == "cal-0002"
    assert read["current_label"] == "second-baseline"


# --- the service and API --------------------------------------------------------


def test_the_service_repairs_a_twice_renamed_citation():
    service = _renamed_twice()
    read = service.sla_calibration_snapshot_successor("pre-change")
    assert read["current_label"] == "final-baseline"
    assert len(read["rename_chain"]) == 2
    assert (
        service.sla_calibration_snapshot_by_label("pre-change") is None
    )


@pytest.fixture()
def client():
    import underwriting_agent.api as api_module

    api_module.service = _service()
    return TestClient(api_module.app)


def _keep_and_rename_twice(client) -> dict:
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "pre-change"},
    )
    for frm, to, actor in (
        ("pre-change", "pre-budget-change", "Uma"),
        ("pre-budget-change", "final-baseline", "Ops Lead"),
    ):
        res = client.post(
            "/queue/sla/calibration/snapshots/rename",
            json={"actor": actor, "from_label": frm, "to_label": to},
        )
        assert res.status_code == 200
    return res.json()


def test_the_by_label_read_answers_with_the_successor(client):
    kept = _keep_and_rename_twice(client)
    res = client.get(
        "/queue/sla/calibration/snapshots/by-label/pre-change"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["cited_label"] == "pre-change"
    assert body["current_label"] == "final-baseline"
    assert body["snapshot"]["snapshot_id"] == kept["snapshot_id"]
    assert body["snapshot"]["label"] == "final-baseline"
    assert [m["to"] for m in body["rename_chain"]] == [
        "pre-budget-change",
        "final-baseline",
    ]


def test_the_direct_read_is_unchanged_for_a_carried_label(client):
    _keep_and_rename_twice(client)
    res = client.get(
        "/queue/sla/calibration/snapshots/by-label/final-baseline"
    )
    assert res.status_code == 200
    body = res.json()
    # The report itself, in its standing shape — no successor
    # wrapper on a direct hit.
    assert body["label"] == "final-baseline"
    assert "cited_label" not in body
    assert "rename_chain" not in body


def test_a_label_that_never_named_anything_still_404s(client):
    _keep_and_rename_twice(client)
    res = client.get("/queue/sla/calibration/snapshots/by-label/nope")
    assert res.status_code == 404
    assert "nope" in res.json()["detail"]


def test_the_repair_does_not_resurrect_a_pruned_report(client):
    _keep_and_rename_twice(client)
    client.post(
        "/queue/sla/calibration/snapshots",
        json={"actor": "Ops Lead", "label": "later"},
    )
    client.post(
        "/queue/sla/calibration/snapshots/prune",
        json={"actor": "Uma", "keep": 1},
    )
    # The renamed report is gone; its fate is in the rename
    # log's pruned stratum, but there is no report to descend
    # to, so the citation 404s — honestly, as it names nothing
    # that stands.
    res = client.get(
        "/queue/sla/calibration/snapshots/by-label/pre-change"
    )
    assert res.status_code == 404
    log = client.get("/queue/sla/calibration/snapshots/renames").json()
    assert log["count"] == 2
    assert all(e["pruned"] for e in log["renames"])


def test_the_diff_still_refuses_a_stale_label(client):
    # The repair is a read act, not a silent re-pointing: a
    # diff naming the old label is refused for the standing
    # reason, so a comparison is never quietly re-aimed.
    _keep_and_rename_twice(client)
    res = client.get(
        "/queue/sla/calibration/snapshots/diff",
        params={"from_label": "pre-change", "to_label": "current"},
    )
    assert res.status_code == 422
    assert "pre-change" in res.json()["detail"]
