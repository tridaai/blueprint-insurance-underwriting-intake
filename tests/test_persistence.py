"""Persistence: SQLite store round-trips and the checkpointed gate."""

from __future__ import annotations

from underwriting_agent.checkpoints import SqliteCheckpointSaver
from underwriting_agent.samples import sample_submission_models
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore, SQLiteStore

SAMPLES = {s.submission_id: s for s in sample_submission_models()}


def test_sqlite_store_survives_reopen(tmp_path):
    path = tmp_path / "state.sqlite3"
    service = SubmissionService(store=SQLiteStore(path), checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U. Writer", reason="clean")

    reopened = SQLiteStore(path)
    record = reopened.get("SUB-1001")
    assert record is not None
    assert record.result.status == "approved"
    assert record.result.decided_by == "U. Writer"
    priors = reopened.prior_submissions(exclude_submission_id="SUB-OTHER")
    assert priors and priors[0]["insured_name"] == "Northgate Legal LLP"
    feedback = reopened.decision_feedback(exclude_submission_id="SUB-OTHER")
    assert feedback and feedback[0]["reason"] == "clean"


def test_inmemory_store_feedback_needs_a_reason():
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(SAMPLES["SUB-1001"])
    service.approve("SUB-1001", approver="U")  # no reason given
    assert store.decision_feedback(exclude_submission_id="X") == []


def test_checkpointed_gate_pauses_and_resumes(tmp_path):
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite3")
    store = InMemoryStore()
    service = SubmissionService(store=store, checkpointer=saver)
    result = service.analyze(SAMPLES["SUB-1001"])
    assert result.status == "awaiting_underwriter"
    # The thread is paused at the gate — a decision resumes it.
    decided = service.approve("SUB-1001", approver="U. Writer")
    assert decided.status == "approved"
    # Approving again is refused by the store (decision of record).
    try:
        service.approve("SUB-1001", approver="U. Writer")
        raise AssertionError("second approval should have raised")
    except ValueError:
        pass


def test_checkpointed_reject_completes_the_thread(tmp_path):
    saver = SqliteCheckpointSaver(tmp_path / "checkpoints.sqlite3")
    service = SubmissionService(store=InMemoryStore(), checkpointer=saver)
    service.analyze(SAMPLES["SUB-1002"])
    decided = service.reject("SUB-1002", reviewer="S. Senior", reason="too heavy")
    assert decided.status == "rejected"
