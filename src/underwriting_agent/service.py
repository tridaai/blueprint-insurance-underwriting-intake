"""Service layer: run + underwriter-decision bookkeeping.

Approving a draft records the underwriter's decision and marks the
packet final. By default it performs NO external action (no email,
no broker portal call, nothing bound). The one opt-in exception is
output routing: when ``ACTION_WEBHOOK_URL`` is configured, the
approved packet is POSTed to that endpoint — the customer's system
of choice — and the outcome is recorded on the result (see
``dispatch_approval_webhook``).

Records live in an approval store (``store.py``): PostgreSQL when
``DATABASE_URL`` is set (the production system of record, shared
across replicas), so analyses and decisions survive restarts; the
in-memory store is the test double when it is not.

The underwriter gate is checkpointed (``checkpoints.py``, on unless
``CHECKPOINTS=off``): an analysis pauses its graph at the gate with
the state persisted under the submission id, and approve/reject
resume that thread with the decision. The split is deliberate — the
store is the record of decisions (and the only thing the decision
flow reads); the checkpointer holds graph state. A missing/finished
thread never blocks a decision.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .checkpoints import get_checkpointer
from .config import env_float, env_str, load_dotenv
from .graph import resume_approval, run_submission
from .model_backends import ModelBackend, get_backend
from .ports import Checkpointer, EventSink
from .retriever import Retriever, get_retriever
from .schemas import AgentResult, SubmissionInput
from .store import ApprovalRecord, broker_summary, default_store

__all__ = ["ApprovalRecord", "BatchItem", "SubmissionService", "Store"]


# Store protocol re-export for call sites that typed against the service.
from .ports import Store  # noqa: E402,F401


@dataclass
class BatchItem:
    """One outcome of a batch analysis: a result, or the error that
    stopped that one submission — never the whole batch."""

    submission_id: str
    result: AgentResult | None = None
    error: str | None = None


def dispatch_approval_webhook(record: ApprovalRecord) -> str | None:
    """POST the approved packet to ``ACTION_WEBHOOK_URL`` (off by default).

    This is the output-routing seam — the thin adapter between an
    approval and the customer's system of choice (their policy admin
    system, a work queue, an automation endpoint). Returns ``None``
    when no URL is configured (the default: no external action at
    all), ``"sent"`` on a 2xx response, and ``"failed"`` on any error
    or non-2xx — a failed dispatch never undoes the approval; it is
    recorded on the result for follow-up. Short timeout on purpose:
    an approval must not hang on a downstream system.
    """
    load_dotenv()
    url = env_str("ACTION_WEBHOOK_URL")
    if not url:
        return None
    result = record.result
    payload = {
        "submission_id": result.submission_id,
        "status": result.status,
        "decided_by": result.decided_by,
        "triage": result.triage.model_dump(mode="json"),
        "drafts": {
            "subject": result.drafts.subject if result.drafts else "",
            "broker_letter": result.drafts.broker_letter if result.drafts else "",
            "underwriter_packet": result.drafts.underwriter_packet if result.drafts else "",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = env_float("ACTION_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "sent" if 200 <= response.status < 300 else "failed"
    except Exception:  # dispatch failure is recorded, never raised
        return "failed"


@dataclass
class SubmissionService:
    backend: ModelBackend | None = None
    retriever: Retriever | None = None
    store: Store | None = None
    # The gate checkpointer: a saver instance, None (= resolve from
    # the environment, CHECKPOINTS, default on), or False (= disabled,
    # whatever the environment says — the CLI batch path uses this
    # because it persists nothing).
    checkpointer: Checkpointer | bool | None = None
    _resolved_store: Store | None = field(default=None, repr=False)
    _resolved_checkpointer: Checkpointer | None = field(default=None, repr=False)
    _checkpointer_resolved: bool = field(default=False, repr=False)

    def _get_store(self) -> Store:
        if self._resolved_store is None:
            self._resolved_store = self.store or default_store()
        return self._resolved_store

    def _get_checkpointer(self) -> Checkpointer | None:
        if not self._checkpointer_resolved:
            self._checkpointer_resolved = True
            if self.checkpointer is False:
                self._resolved_checkpointer = None
            elif self.checkpointer is not None:
                self._resolved_checkpointer = self.checkpointer  # type: ignore[assignment]
            else:
                self._resolved_checkpointer = get_checkpointer()
        return self._resolved_checkpointer

    def _resume_thread(self, submission_id: str, decision: dict) -> None:
        """Complete the checkpointed graph thread with a decision.

        Best-effort by design: the store record the caller just saved
        is the decision of record. A thread that is missing or already
        finished (the analysis ran with checkpoints off, modes were
        mixed) has nothing to resume, and a resume failure never
        unmakes a recorded human decision.
        """
        checkpointer = self._get_checkpointer()
        if checkpointer is None:
            return
        try:
            resume_approval(submission_id, decision, checkpointer)
        except Exception:
            pass

    def analyze(
        self, submission: SubmissionInput | dict, event_sink: EventSink | None = None
    ) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        # ``event_sink`` (optional) receives the run's structured events
        # (see events.py) — the API's streaming endpoint passes one.
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        model = (
            submission
            if isinstance(submission, SubmissionInput)
            else SubmissionInput.model_validate(submission)
        )
        return self._analyze_model(model, backend, self.retriever, event_sink=event_sink)

    def _analyze_model(
        self,
        model: SubmissionInput,
        backend: ModelBackend,
        retriever: Retriever,
        event_sink: EventSink | None = None,
    ) -> AgentResult:
        # Memory: what the store already knows about this insured and
        # this broker becomes diagnosis evidence for the new analysis.
        # The raw entries ride along too — the diagnosis tool loop
        # (provider mode) reads insured / broker history through them.
        priors = self._get_store().prior_submissions(
            exclude_submission_id=model.submission_id
        )
        history = self._history_summary(model, priors)
        # Feedback loop: what underwriters said about earlier cases
        # for this insured / broker (with their reasons) joins the
        # same evidence — the next diagnosis shows the pattern of
        # human oversight.
        feedback = self._feedback_for(model)
        if feedback:
            history = {**(history or {}), "feedback": feedback}
        result = run_submission(
            model,
            backend=backend,
            retriever=retriever,
            history=history,
            priors=priors,
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=model.submission_id,
        )
        self._get_store().save(
            ApprovalRecord(result=result, submission=model.model_dump(mode="json"))
        )
        return result

    def analyze_batch(
        self,
        submissions: list[SubmissionInput | dict],
        concurrency: int = 1,
    ) -> list[BatchItem]:
        """Analyse many submissions, in input order, optionally concurrently.

        Each item runs the same pipeline as :meth:`analyze` and gets
        its own result object. Injected backend/retriever are shared
        (test doubles are stateless); otherwise every item resolves
        fresh ones from the environment, so a provider backend's usage
        counters and a retriever's per-run stats never race across
        threads. The store is shared deliberately — batch items see
        each other as memory, like a real intake queue — and its
        writes are serialised by the store itself (SQLite: one
        connection per call; in-memory: a lock). A submission that
        fails (bad payload, provider down) lands in its item's
        ``error``; it never fails the batch.
        """

        def run_one(raw: SubmissionInput | dict) -> BatchItem:
            submission_id = (
                raw.get("submission_id", "unknown")
                if isinstance(raw, dict)
                else getattr(raw, "submission_id", "unknown")
            )
            try:
                model = (
                    raw
                    if isinstance(raw, SubmissionInput)
                    else SubmissionInput.model_validate(raw)
                )
                backend = self.backend or get_backend()
                retriever = self.retriever or get_retriever()
                result = self._analyze_model(model, backend, retriever)
                return BatchItem(submission_id=model.submission_id, result=result)
            except Exception as exc:  # one bad submission never fails the batch
                return BatchItem(
                    submission_id=str(submission_id),
                    error=f"{type(exc).__name__}: {exc}",
                )

        workers = max(1, int(concurrency))
        if workers == 1 or len(submissions) <= 1:
            return [run_one(item) for item in submissions]
        semaphore = threading.Semaphore(workers)

        def guarded(item: SubmissionInput | dict) -> BatchItem:
            with semaphore:
                return run_one(item)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(guarded, submissions))

    def _feedback_for(self, submission: SubmissionInput) -> list[dict]:
        """Recent underwriter feedback matching this insured or broker.

        Bounded to the last 3 matching decisions (the store returns
        newest first), reasons truncated — feedback is evidence, not
        a transcript. Each entry: {decision, reason, match} where
        match is "insured" or "broker" (insured wins when both match).
        """
        entries = self._get_store().decision_feedback(
            exclude_submission_id=submission.submission_id
        )
        if not entries:
            return []
        matched: list[dict] = []
        for entry in entries:
            if submission.insured_name and entry["insured_name"] == submission.insured_name:
                match = "insured"
            elif submission.broker_name and entry["broker_name"] == submission.broker_name:
                match = "broker"
            else:
                continue
            matched.append(
                {
                    "decision": entry["decision"],
                    "reason": entry["reason"][:160],
                    "match": match,
                }
            )
            if len(matched) == 3:
                break
        return matched

    def _history_summary(
        self, submission: SubmissionInput, priors: list[dict] | None = None
    ) -> dict | None:
        """Summarise prior analysed submissions for this insured + broker.

        Returns None when there is nothing to remember.
        """
        if priors is None:
            priors = self._get_store().prior_submissions(
                exclude_submission_id=submission.submission_id
            )
        if not priors:
            return None
        insured_hits = [
            e
            for e in priors
            if submission.insured_name and e["insured_name"] == submission.insured_name
        ]
        broker = broker_summary(priors, submission.broker_name)
        if not insured_hits and not broker["broker_count"]:
            return None
        return {
            "insured": submission.insured_name,
            "insured_count": len(insured_hits),
            "insured_recent_outcomes": [e["outcome"] for e in insured_hits[:3]],
            "broker": submission.broker_name,
            "broker_count": broker["broker_count"],
            "broker_outcome_counts": broker["broker_outcome_counts"],
        }

    def approve(
        self, submission_id: str, approver: str, reason: str = ""
    ) -> AgentResult:
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        if record.result.status != "awaiting_underwriter":
            raise ValueError(
                f"Submission {submission_id} is not awaiting an underwriter "
                f"(status: {record.result.status})"
            )
        if record.result.validation is not None and not record.result.validation.passed:
            failed = "; ".join(
                f"{c.name}: {c.detail}"
                for c in record.result.validation.checks
                if not c.passed
            )
            raise ValueError(
                "Draft failed guardrail validation and cannot be approved: " + failed
            )
        record.approved = True
        record.approver = approver
        record.approve_reason = reason
        record.result.status = "approved"
        record.result.decision = "approved"
        record.result.decided_by = approver
        record.result.decision_reason = reason or None
        # Output routing: by default there is NO external action. When
        # the operator configures ACTION_WEBHOOK_URL, the approved
        # packet is POSTed to that endpoint (the customer's system of
        # choice) and the outcome is recorded — a failed dispatch
        # leaves the approval standing and says so on the result.
        record.result.external_action_taken = False
        dispatch = dispatch_approval_webhook(record)
        if dispatch is not None:
            record.dispatch_status = dispatch
            record.result.dispatch_status = dispatch
            if dispatch == "sent":
                record.result.external_action_taken = True
        store.save(record)
        self._resume_thread(
            submission_id,
            {"decision": "approved", "actor": approver, "reason": reason},
        )
        return record.result

    def reject(self, submission_id: str, reviewer: str, reason: str = "") -> AgentResult:
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        if record.result.status != "awaiting_underwriter":
            raise ValueError(
                f"Submission {submission_id} is not awaiting an underwriter "
                f"(status: {record.result.status})"
            )
        record.rejected_by = reviewer
        record.reject_reason = reason
        record.result.status = "rejected"
        record.result.decision = "rejected"
        record.result.decided_by = reviewer
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            submission_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
        )
        return record.result

    def get(self, submission_id: str) -> AgentResult | None:
        record = self._get_store().get(submission_id)
        return record.result if record else None
