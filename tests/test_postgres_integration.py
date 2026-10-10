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


class _FakeEmbeddingsClient:
    """Deterministic stand-in for the provider embeddings client:
    bag-of-words hashing into 16 dimensions, so texts that share
    words land close together — enough to prove the pgvector path
    ranks by meaning-of-sorts, not by insertion order."""

    def __init__(self) -> None:
        self.corpus_embed_calls = 0

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        vector = [0.0] * 16
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            vector[digest[0] % 16] += 1.0
        return vector

    @property
    def embeddings(self):
        return self

    def create(self, model=None, input=None):
        from types import SimpleNamespace

        texts = list(input or [])
        if len(texts) > 1:
            self.corpus_embed_calls += 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=self._embed_one(t), index=i)
                for i, t in enumerate(texts)
            ]
        )


def test_pgvector_retrieval_round_trip(pg_env, monkeypatch):
    from underwriting_agent.db import connect, run_migrations
    from underwriting_agent.retriever import SemanticRetriever

    run_migrations(pg_env)
    guidelines = [
        {"id": "GL-CYBER", "title": "Cyber appetite", "text": "cyber ransomware controls and backups"},
        {"id": "GL-PROP", "title": "Property appetite", "text": "property sprinklers roof construction occupancy"},
        {"id": "GL-MARINE", "title": "Marine appetite", "text": "marine cargo vessel transit warehouse"},
    ]
    fake = _FakeEmbeddingsClient()
    monkeypatch.setattr(
        SemanticRetriever,
        "_build_embeddings_client",
        lambda self: (fake, "fake-embed-16", "fake", "http://fake"),
    )
    with connect(pg_env) as conn:
        conn.execute("DELETE FROM guideline_embeddings WHERE model = 'fake-embed-16'")
        conn.commit()

    retriever = SemanticRetriever(guidelines)
    results = retriever.retrieve("cyber ransomware backups", top_k=2)
    assert retriever.vector_store == "pgvector"
    assert results and results[0].guideline_id == "GL-CYBER"

    # Second retrieval: corpus vectors come from the table, not a
    # re-embed — the fake's corpus batch counter must not move.
    calls_after_first = fake.corpus_embed_calls
    again = retriever.retrieve("property sprinklers roof", top_k=1)
    assert again and again[0].guideline_id == "GL-PROP"
    assert fake.corpus_embed_calls == calls_after_first


def test_readiness_reports_postgres(pg_env):
    from fastapi.testclient import TestClient

    from underwriting_agent import api as api_module

    # Lifespan runs (migrations) because the client is used as a
    # context manager; readiness must name the real database.
    with TestClient(api_module.app) as client:
        response = client.get("/readiness")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "postgres"}


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
