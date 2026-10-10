"""Dispatch operations view: the webhook ledgers, read as one surface.

Per-submission ledgers answer for one approval at a time. Operations
needs the cross-record view: every failed delivery in the queue,
oldest first, which of them are due a retry under the recorded
backoff schedule, and a bulk retry that works the list — without
ever touching the approvals themselves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import underwriting_agent.service as service_module
from underwriting_agent.insights import dispatch_overview
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.schemas import AgentResult
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import ApprovalRecord, InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import underwriting_agent.api as api_module
    from underwriting_agent.api import app
    from underwriting_agent.wiring import build_service_from_env

    monkeypatch.setenv("CHECKPOINTS", "off")
    api_module.service = build_service_from_env()
    return TestClient(app)


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def attempt(at: datetime, status="failed", http_status=None, error="connection refused"):
    return {
        "attempt": 1,
        "attempted_at": at.isoformat(timespec="seconds"),
        "status": status,
        "http_status": http_status,
        "signature_id": None,
        "error": error if status == "failed" else None,
        "backoff_seconds": 0,
    }


def record(sid, insured, attempts, dispatch_status):
    return ApprovalRecord(
        result=AgentResult(submission_id=sid, insured_name=insured),
        submission={"insured_name": insured, "broker_name": "Broker One"},
        dispatch_status=dispatch_status,
        dispatch_attempts=attempts,
    )


# ---------------------------------------------------------------------------
# The overview projection
# ---------------------------------------------------------------------------


def test_overview_orders_failures_oldest_first_and_marks_due_retries():
    old = record(
        "SUB-OLD", "Old Insured",
        [attempt(NOW - timedelta(hours=3))], "failed",
    )
    fresh = record(
        "SUB-NEW", "New Insured",
        [attempt(NOW - timedelta(seconds=5))], "failed",
    )
    sent = record(
        "SUB-OK", "Fine Insured",
        [attempt(NOW - timedelta(hours=1), status="sent", error=None)], "sent",
    )
    untouched = record("SUB-IDLE", "Idle Insured", [], None)
    overview = dispatch_overview([fresh, sent, old, untouched], now=NOW)

    assert overview["attempts_total"] == 3
    assert overview["records_with_attempts"] == 3
    assert overview["sent_records"] == 1
    assert overview["failed_records"] == 2
    assert [f["submission_id"] for f in overview["failures"]] == ["SUB-OLD", "SUB-NEW"]
    assert overview["oldest_failure_at"] == old.dispatch_attempts[0]["attempted_at"]
    # The 3-hour-old failure is long past its 30s first-retry backoff;
    # the 5-second-old one is still inside it.
    due = {f["submission_id"] for f in overview["due_retries"]}
    assert due == {"SUB-OLD"}
    fresh_row = next(f for f in overview["failures"] if f["submission_id"] == "SUB-NEW")
    assert fresh_row["retry_due"] is False
    assert fresh_row["next_attempt"] == 2
    assert fresh_row["next_backoff_seconds"] == 30
    assert fresh_row["last_error"] == "connection refused"


def test_overview_backoff_grows_with_attempt_count():
    many = [attempt(NOW - timedelta(days=2)) for _ in range(4)]
    overview = dispatch_overview([record("SUB-X", "X", many, "failed")], now=NOW)
    row = overview["failures"][0]
    assert row["attempts"] == 4
    # Next attempt is #5: backoff min(30 * 2**3, 600) = 240.
    assert row["next_backoff_seconds"] == 240
    assert row["retry_due"] is True


def test_overview_of_an_empty_store_is_all_zeros():
    overview = dispatch_overview([], now=NOW)
    assert overview["attempts_total"] == 0
    assert overview["failures"] == [] and overview["due_retries"] == []
    assert overview["oldest_failure_at"] is None


# ---------------------------------------------------------------------------
# Failure aging: buckets by first failure, and the stuck flag
# ---------------------------------------------------------------------------


def test_failure_age_runs_from_the_first_failed_attempt():
    attempts = [
        attempt(NOW - timedelta(days=10)),
        attempt(NOW - timedelta(hours=1)),
    ]
    overview = dispatch_overview(
        [record("SUB-AGED", "Aged Insured", attempts, "failed")], now=NOW
    )
    row = overview["failures"][0]
    # Ten days since the delivery started failing — not one hour
    # since the latest attempt.
    assert row["first_attempt_at"] == attempts[0]["attempted_at"]
    assert row["failure_age_days"] == 10
    assert row["failure_age_bucket"] == "8-14 days"
    buckets = {b["bucket"]: b for b in overview["failure_age_buckets"]}
    assert buckets["8-14 days"]["submission_ids"] == ["SUB-AGED"]
    assert buckets["0-1 days"]["count"] == 0


def test_a_failure_older_than_the_threshold_with_budget_left_is_stuck():
    old = record(
        "SUB-STUCK", "Stuck Insured",
        [attempt(NOW - timedelta(days=3))], "failed",
    )
    overview = dispatch_overview([old], now=NOW)
    row = overview["failures"][0]
    assert row["attempts_remaining"] == 7  # default budget 8, one attempt made
    assert row["stuck"] is True
    assert overview["stuck_count"] == 1
    assert [f["submission_id"] for f in overview["stuck_failures"]] == ["SUB-STUCK"]


def test_a_recent_failure_is_not_stuck_however_due():
    fresh = record(
        "SUB-FRESH", "Fresh Insured",
        [attempt(NOW - timedelta(hours=2))], "failed",
    )
    overview = dispatch_overview([fresh], now=NOW)
    assert overview["failures"][0]["stuck"] is False
    assert overview["stuck_count"] == 0


def test_a_failure_past_its_attempt_budget_is_exhausted_not_stuck():
    attempts = [attempt(NOW - timedelta(days=5)) for _ in range(8)]
    overview = dispatch_overview(
        [record("SUB-SPENT", "Spent Insured", attempts, "failed")], now=NOW
    )
    row = overview["failures"][0]
    assert row["attempts_remaining"] == 0
    assert row["stuck"] is False  # old, but it needs a decision, not a retry
    assert overview["stuck_count"] == 0


def test_stuck_threshold_and_budget_are_configurable():
    rec = record(
        "SUB-CFG", "Configured Insured",
        [attempt(NOW - timedelta(hours=30))], "failed",
    )
    lenient = dispatch_overview([rec], now=NOW, stuck_age_hours=48.0)
    assert lenient["failures"][0]["stuck"] is False
    strict = dispatch_overview([rec], now=NOW, stuck_age_hours=12.0, attempt_budget=1)
    assert strict["failures"][0]["attempts_remaining"] == 0
    assert strict["failures"][0]["stuck"] is False
    strict = dispatch_overview([rec], now=NOW, stuck_age_hours=12.0, attempt_budget=4)
    assert strict["failures"][0]["stuck"] is True
    assert strict["stuck_age_hours"] == 12.0 and strict["attempt_budget"] == 4


# ---------------------------------------------------------------------------
# Service + API: bulk retry works the failure list
# ---------------------------------------------------------------------------


def _approve_two_with_failing_webhook(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    monkeypatch.setattr(service_module.urllib.request, "urlopen", boom)
    service = make_service()
    service.analyze(SAMPLES["SUB-1001"])
    service.analyze(SAMPLES["SUB-1009"])
    service.approve("SUB-1001", approver="U")
    service.approve("SUB-1009", approver="U")
    return service


def test_bulk_retry_recovers_every_failed_dispatch(monkeypatch):
    service = _approve_two_with_failing_webhook(monkeypatch)
    overview = service.dispatch_ops()
    assert overview["failed_records"] == 2 and overview["attempts_total"] == 2
    assert service.metrics()["dispatch"] == {
        "attempts": 2,
        "sent_records": 0,
        "failed_records": 2,
    }

    monkeypatch.setattr(
        service_module.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(200),
    )
    outcomes = service.retry_failed_dispatches()
    assert {o["submission_id"] for o in outcomes} == {"SUB-1001", "SUB-1009"}
    assert all(o["dispatch_status"] == "sent" for o in outcomes)
    overview = service.dispatch_ops()
    assert overview["failed_records"] == 0 and overview["sent_records"] == 2
    # Nothing left to retry: a second sweep is a no-op, not an error.
    assert service.retry_failed_dispatches() == []


def test_bulk_retry_reports_records_that_stay_failed(monkeypatch):
    service = _approve_two_with_failing_webhook(monkeypatch)

    def still_down(request, timeout=None):
        raise OSError("still down")

    monkeypatch.setattr(service_module.urllib.request, "urlopen", still_down)
    outcomes = service.retry_failed_dispatches()
    assert len(outcomes) == 2
    assert all(o["dispatch_status"] == "failed" for o in outcomes)
    assert all(o["attempts"] == 2 for o in outcomes)
    # The approvals were never rewritten by any of this.
    assert service.get("SUB-1001").status == "approved"


def test_dispatch_endpoints(client, monkeypatch):
    empty = client.get("/dispatch")
    assert empty.status_code == 200
    assert empty.json()["attempts_total"] == 0
    swept = client.post("/dispatch/retry-failed")
    assert swept.status_code == 200
    assert swept.json() == {"count": 0, "retried": []}
