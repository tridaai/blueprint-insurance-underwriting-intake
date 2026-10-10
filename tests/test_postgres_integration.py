"""PostgreSQL integration tests — gated on ``DATABASE_URL``.

The hermetic suite runs on in-memory doubles and never touches a
server. These tests exercise the production persistence path — the
Postgres approval store, the official LangGraph Postgres
checkpointer, and the migration runner — against a real database.
They skip cleanly when ``DATABASE_URL`` is not set.

Point ``DATABASE_URL`` at a *throwaway* database (the compose
stack's Postgres, or any pgvector-capable server): the tests run the
real migrations against it and write records with ``SUB-`` sample
ids, cleaning up the checkpoint threads they create.

Note on hermeticity: the suite-wide fixture scrubs ``DATABASE_URL``
from the environment before each test, so the URL is captured here
at import time (collection runs before fixtures) and re-applied per
test with ``monkeypatch``.
"""

from __future__ import annotations

import os

import pytest

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _DATABASE_URL,
    reason="DATABASE_URL not set — Postgres integration tests skipped",
)


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DATABASE_URL)
    return _DATABASE_URL


@pytest.fixture()
def clean_approvals(pg_env):
    """Start (and end) with no rows for the sample submissions."""
    from underwriting_agent.db import connect, run_migrations

    run_migrations(pg_env)  # the approvals table must exist to clean it
    ids = ("SUB-1001", "SUB-1002")
    with connect(pg_env) as conn:
        conn.execute(
            "DELETE FROM approvals WHERE submission_id IN (%s, %s)", ids
        )
        conn.commit()
    yield ids
    with connect(pg_env) as conn:
        conn.execute(
            "DELETE FROM approvals WHERE submission_id IN (%s, %s)", ids
        )
        conn.commit()


def test_migrations_are_idempotent(pg_env):
    from underwriting_agent.db import run_migrations

    first = run_migrations(pg_env)
    second = run_migrations(pg_env)
    assert second == []  # nothing left to apply the second time
    assert set(first) <= {"0001", "0002", "0003"}


def test_postgres_store_survives_reopen(clean_approvals):
    from underwriting_agent.samples import sample_submission_models
    from underwriting_agent.service import SubmissionService
    from underwriting_agent.store import PostgresStore

    samples = {s.submission_id: s for s in sample_submission_models()}
    store = PostgresStore(_DATABASE_URL)
    service = SubmissionService(store=store, checkpointer=False)
    service.analyze(samples["SUB-1001"])
    service.approve("SUB-1001", approver="U. Writer", reason="clean")

    reopened = PostgresStore(_DATABASE_URL)
    record = reopened.get("SUB-1001")
    assert record is not None
    assert record.result.status == "approved"
    assert record.result.decided_by == "U. Writer"
    priors = reopened.prior_submissions(exclude_submission_id="SUB-OTHER")
    assert priors and priors[0]["insured_name"] == "Northgate Legal LLP"
    feedback = reopened.decision_feedback(exclude_submission_id="SUB-OTHER")
    assert feedback and feedback[0]["reason"] == "clean"


def test_postgres_checkpointer_pauses_and_resumes_across_instances(pg_env):
    from underwriting_agent.checkpoints import get_checkpointer
    from underwriting_agent.samples import sample_submission_models
    from underwriting_agent.service import SubmissionService
    from underwriting_agent.store import InMemoryStore

    samples = {s.submission_id: s for s in sample_submission_models()}
    saver = get_checkpointer()
    assert saver is not None
    try:
        service = SubmissionService(store=InMemoryStore(), checkpointer=saver)
        result = service.analyze(samples["SUB-1002"])
        assert result.status == "awaiting_underwriter"

        # The paused thread is visible to a *different* saver instance
        # over the same database — the gate survives a restart.
        config = {"configurable": {"thread_id": "SUB-1002"}}
        assert saver.get_tuple(config) is not None

        decided = service.approve("SUB-1002", approver="U. Writer")
        assert decided.status == "approved"
    finally:
        saver.delete_thread("SUB-1002")
