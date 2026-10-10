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

import hashlib
import hmac
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .autonomy import recommend_autonomy
from .bundles import (
    blocking_lines as bundle_blocking_lines,
    bundle_citations,
    compose_bundle_drafts,
    driving_line_index,
    is_bundle,
    line_slug,
    rollup_facts,
    rollup_triage,
    split_bundle,
)
from .checkpoints import get_checkpointer
from .clarify import build_bundle_information_request
from .config import env_float, env_str, load_dotenv
from .graph import resume_approval, run_submission
from .guardrails import validate_drafts, validate_status_note
from .guidelines_data import GUIDELINES_VERSION
from .insights import (
    all_scorecards,
    audit_rows,
    broker_quality_report,
    compose_status_note,
    compute_metrics,
    dispatch_overview,
    inbox_groups,
    queue_aging,
    queue_item_for,
    scorecard,
    scorecards_for,
)
from .model_backends import ModelBackend, get_backend
from .object_store import get_object_store
from .ports import Checkpointer, EventSink, ObjectStore
from .retriever import Retriever, get_retriever
from .reviewer import review_drafts
from .schemas import (
    AgentResult,
    Diagnosis,
    DispatchAttempt,
    LineAnalysis,
    RetrievedGuideline,
    RunTelemetry,
    SubmissionInput,
    TraceStep,
)
from .store import ApprovalRecord, broker_summary, default_store, history_entry
from .verify import verify_drafts

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


def _dispatch_once(record: ApprovalRecord) -> dict | None:
    """Perform one webhook dispatch attempt and return its ledger entry.

    This is the output-routing seam — the thin adapter between an
    approval and the customer's system of choice (their policy admin
    system, a work queue, an automation endpoint). Returns ``None``
    when no URL is configured (the default: no external action at
    all). Otherwise returns the attempt record for the delivery
    ledger: attempt number, timestamp, outcome (``"sent"`` on 2xx,
    ``"failed"`` otherwise), the receiver's HTTP status when it
    answered, the signature identifier when the dispatch was signed,
    the error when it was not, and the backoff the retry schedule
    assigned before this attempt. A failed dispatch never raises and
    never undoes the approval. Short timeout on purpose: an approval
    must not hang on a downstream system.
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
        "decided_at": result.decided_at,
        "triage": result.triage.model_dump(mode="json"),
        "drafts": {
            "subject": result.drafts.subject if result.drafts else "",
            "broker_letter": result.drafts.broker_letter if result.drafts else "",
            "underwriter_packet": result.drafts.underwriter_packet if result.drafts else "",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # Signed dispatch: when ACTION_WEBHOOK_SECRET is set the receiver
    # can verify the packet really came from this service — HMAC-SHA256
    # over the exact body bytes, sent as X-Trida-Signature. The ledger
    # keeps a truncated identifier, never the full signature.
    signature_id = None
    secret = env_str("ACTION_WEBHOOK_SECRET")
    if secret:
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        headers["X-Trida-Signature"] = f"sha256={signature}"
        signature_id = f"sha256={signature[:12]}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    timeout = env_float("ACTION_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    attempt_no = len(record.dispatch_attempts) + 1
    entry: dict = {
        "attempt": attempt_no,
        "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "failed",
        "http_status": None,
        "signature_id": signature_id,
        "error": None,
        # Retry schedule bookkeeping: the first attempt is immediate;
        # each retry waits twice the previous delay (30s, 60s, 120s,
        # ... capped at 600s). The operator-triggered retry does not
        # sleep — the value records what the schedule prescribed.
        "backoff_seconds": (
            0 if attempt_no == 1 else min(30 * 2 ** (attempt_no - 2), 600)
        ),
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            entry["http_status"] = response.status
            if 200 <= response.status < 300:
                entry["status"] = "sent"
            else:
                entry["error"] = f"receiver answered HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # urllib raises for non-2xx: the receiver DID answer, so its
        # status belongs in the ledger alongside the failure.
        entry["http_status"] = exc.code
        entry["error"] = f"receiver answered HTTP {exc.code}"
    except Exception as exc:  # dispatch failure is recorded, never raised
        entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return entry


def dispatch_approval_webhook(record: ApprovalRecord) -> str | None:
    """One dispatch attempt, as a status string (``None`` / ``"sent"``
    / ``"failed"``). The service drives dispatches through
    :func:`_dispatch_once` so each attempt also lands in the record's
    delivery ledger; this wrapper keeps the original contract."""
    entry = _dispatch_once(record)
    return entry["status"] if entry is not None else None


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
    # The document object store: an instance, None (= resolve from
    # the environment, S3_BUCKET), or False (= disabled).
    object_store: ObjectStore | bool | None = None
    _resolved_store: Store | None = field(default=None, repr=False)
    _resolved_checkpointer: Checkpointer | None = field(default=None, repr=False)
    _checkpointer_resolved: bool = field(default=False, repr=False)
    _resolved_object_store: ObjectStore | None = field(default=None, repr=False)
    _object_store_resolved: bool = field(default=False, repr=False)

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

    def _get_object_store(self) -> ObjectStore | None:
        if not self._object_store_resolved:
            self._object_store_resolved = True
            if self.object_store is False:
                self._resolved_object_store = None
            elif self.object_store is not None:
                self._resolved_object_store = self.object_store  # type: ignore[assignment]
            else:
                self._resolved_object_store = get_object_store()
        return self._resolved_object_store

    def _resolve_document_texts(self, model: SubmissionInput) -> SubmissionInput:
        """Fetch document text for key-only documents (production intake).

        An adapter that already stored the document submits just its
        ``object_key``; the pipeline needs the text, so it is fetched
        through the port here — the one place raw bytes enter.
        A failed fetch leaves the document text empty (the run
        proceeds; completeness scoring sees the gap), never raises.
        Bundle line documents resolve the same way as top-level ones.
        """
        store = self._get_object_store()
        if store is None:
            return model

        def resolve(documents):
            resolved, changed = [], False
            for doc in documents:
                if doc.object_key and not doc.raw_text:
                    try:
                        text = store.get(doc.object_key).decode(
                            "utf-8", errors="replace"
                        )
                    except Exception:
                        text = ""
                    if text:
                        doc = doc.model_copy(update={"raw_text": text})
                        changed = True
                resolved.append(doc)
            return resolved, changed

        documents, changed = resolve(model.documents)
        updates: dict = {}
        if changed:
            updates["documents"] = documents
        lines = []
        lines_changed = False
        for line in model.lines:
            line_documents, line_changed = resolve(line.documents)
            if line_changed:
                line = line.model_copy(update={"documents": line_documents})
                lines_changed = True
            lines.append(line)
        if lines_changed:
            updates["lines"] = lines
        if not updates:
            return model
        return model.model_copy(update=updates)

    def _archive_documents(self, model: SubmissionInput) -> dict:
        """Archive inline documents to the object store.

        Returns the submission payload to persist: the model's dump,
        with ``object_key`` filled in for every document that was
        written — bundle-level documents under the submission's
        prefix, line documents under their line's. Archiving is
        bookkeeping, not analysis — a failed put skips that
        document's key and never fails the run.
        """
        payload = model.model_dump(mode="json")
        store = self._get_object_store()
        if store is None:
            return payload

        def archive(documents, payload_documents, prefix) -> None:
            for index, doc in enumerate(documents):
                if not doc.raw_text or doc.object_key:
                    continue
                name = doc.document_id or f"doc-{index}"
                key = f"{prefix}/{name}.txt"
                try:
                    store.put(
                        key,
                        doc.raw_text.encode("utf-8"),
                        "text/plain; charset=utf-8",
                    )
                except Exception:
                    continue
                payload_documents[index]["object_key"] = key

        archive(
            model.documents,
            payload["documents"],
            f"submissions/{model.submission_id}/documents",
        )
        for line_index, line in enumerate(model.lines):
            archive(
                line.documents,
                payload["lines"][line_index]["documents"],
                f"submissions/{model.submission_id}/lines/"
                f"{line_slug(line.line_of_business)}/documents",
            )
        return payload

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
        model = self._resolve_document_texts(model)
        if is_bundle(model):
            return self._analyze_bundle(model, backend, retriever, event_sink)
        # Memory: what the store already knows about this insured and
        # this broker becomes diagnosis evidence for the new analysis.
        # The raw entries ride along too — the diagnosis tool loop
        # (provider mode) reads insured / broker history through them.
        prior_records = [
            record
            for record in self._get_store().all_records()
            if record.result.submission_id != model.submission_id
        ]
        priors = [history_entry(record) for record in prior_records]
        history = self._history_summary(model, priors)
        # Scorecards: the parties' whole track record (outcome mix,
        # approval rate, average completeness, top referral reasons)
        # joins the same evidence — the diagnosis cites it when the
        # insured or broker has history in the store.
        cards = scorecards_for(
            prior_records,
            insured_name=model.insured_name,
            broker_name=model.broker_name,
        )
        if cards:
            history = {**(history or {}), "scorecards": cards}
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
            prior_submission=self._prior_submission_for(model, prior_records),
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=model.submission_id,
        )
        self._get_store().save(
            ApprovalRecord(result=result, submission=self._archive_documents(model))
        )
        return result

    def _analyze_bundle(
        self,
        model: SubmissionInput,
        backend: ModelBackend,
        retriever: Retriever,
        event_sink: EventSink | None = None,
    ) -> AgentResult:
        """Analyse a multi-line bundle: one sub-analysis per line,
        one combined result, one record, one gate.

        Each line runs the ordinary pipeline (``run_submission``)
        with the bundle's shared insured context — the same memory,
        scorecards, and feedback — and a prior keyed to *that line*
        (a prior bundle's matching line, or a prior single-line
        submission of the same line), so renewal diffs stay per
        line. The lines do not gate individually: no checkpointer is
        attached to a sub-analysis, and the only record saved is the
        bundle's, under the bundle id — approve/reject decide the
        bundle, never a line.
        """
        subs = split_bundle(model)
        prior_records = [
            record
            for record in self._get_store().all_records()
            if record.result.submission_id != model.submission_id
        ]
        priors = [history_entry(record) for record in prior_records]
        history = self._history_summary(model, priors)
        cards = scorecards_for(
            prior_records,
            insured_name=model.insured_name,
            broker_name=model.broker_name,
        )
        if cards:
            history = {**(history or {}), "scorecards": cards}
        feedback = self._feedback_for(model)
        if feedback:
            history = {**(history or {}), "feedback": feedback}

        analyses: list[LineAnalysis] = []
        for sub in subs:
            line_result = run_submission(
                sub,
                backend=backend,
                retriever=retriever,
                history=history,
                priors=priors,
                prior_submission=self._prior_submission_for(sub, prior_records),
                event_sink=event_sink,
                checkpointer=None,
            )
            analyses.append(
                LineAnalysis(
                    line_of_business=sub.line_of_business,
                    submission_id=sub.submission_id,
                    result=line_result,
                )
            )

        triage = rollup_triage(analyses)
        facts = rollup_facts(analyses)
        driving = analyses[driving_line_index(analyses)]
        citations = bundle_citations(analyses)
        drafts = compose_bundle_drafts(model, analyses)
        guidelines: list[RetrievedGuideline] = []
        seen_ids: set[str] = set()
        for analysis in analyses:
            for guideline in analysis.result.guidelines:
                if guideline.guideline_id not in seen_ids:
                    seen_ids.add(guideline.guideline_id)
                    guidelines.append(guideline)
        lines_text = " + ".join(a.line_of_business for a in analyses)
        diagnosis = Diagnosis(
            summary=(
                f"Multi-line bundle for {model.insured_name} ({lines_text}): "
                + "; ".join(
                    f"{a.line_of_business} — {a.result.triage.outcome} / "
                    f"{a.result.triage.risk_tier}, readiness "
                    f"{a.result.facts.readiness}"
                    for a in analyses
                )
                + f". The bundle follows its most severe line "
                f"({driving.line_of_business}); the per-line diagnoses are "
                "in the packet sections."
            ),
            risk_factors=[
                f"[{a.line_of_business}] {factor}"
                for a in analyses
                for factor in a.result.diagnosis.risk_factors
            ],
            evidence=[
                f"[{a.line_of_business}] {line}"
                for a in analyses
                for line in a.result.diagnosis.evidence
            ],
            source="template",
            note="composed in code from the per-line diagnoses",
        )
        verification = verify_drafts(
            submission=model,
            triage=triage.model_dump(),
            facts=facts.model_dump(),
            drafts=drafts,
            retrieved_ids=citations,
            backend=backend,
        )
        review = review_drafts(
            submission=model,
            triage=triage.model_dump(),
            facts=facts.model_dump(),
            diagnosis=diagnosis,
            options=driving.result.options,
            drafts=drafts,
            retrieved_ids=citations,
            backend=backend,
        )
        reviewer_blocked = bool(review.enabled and review.verdict == "block")
        validation = validate_drafts(model, drafts, citations)
        autonomy = recommend_autonomy(
            triage=triage,
            facts=facts,
            validation=validation,
            repair_attempted=False,
            cross_check=driving.result.cross_check,
            review=review,
            reviewer_blocked=reviewer_blocked,
        )
        extractions = [e for a in analyses for e in a.result.extractions]
        # The bundle's information request is per line: only the
        # lines actually missing something are asked, each for its
        # own items (clarify.build_bundle_information_request).
        info = build_bundle_information_request(
            submission=model,
            triage_outcome=triage.outcome,
            analyses=analyses,
        )
        trace = [
            TraceStep(
                name="bundle",
                title="Multi-line bundle split",
                details=[
                    f"{len(analyses)} line(s) analysed under one bundle gate: "
                    + "; ".join(
                        f"{a.line_of_business} — {a.result.triage.outcome} / "
                        f"{a.result.triage.risk_tier}"
                        for a in analyses
                    )
                ],
            )
        ]
        for sub, analysis in zip(subs, analyses):
            slug = line_slug(sub.line_of_business).lower()
            trace.extend(
                TraceStep(
                    name=f"{slug}:{step.name}",
                    title=f"[{analysis.line_of_business}] {step.title}",
                    details=step.details,
                    duration_ms=step.duration_ms,
                )
                for step in analysis.result.trace
            )

        def _sum_optional(values: list) -> float | int | None:
            if all(v is None for v in values):
                return None
            return sum(v or 0 for v in values)

        telemetries = [a.result.telemetry for a in analyses]
        first_telemetry = telemetries[0] if telemetries else None
        telemetry = RunTelemetry(
            backend=first_telemetry.backend if first_telemetry else "mock",
            model=first_telemetry.model if first_telemetry else "",
            model_calls=sum(t.model_calls for t in telemetries if t),
            input_tokens=_sum_optional(
                [t.input_tokens if t else None for t in telemetries]
            ),
            output_tokens=_sum_optional(
                [t.output_tokens if t else None for t in telemetries]
            ),
            estimated_cost_usd=_sum_optional(
                [t.estimated_cost_usd if t else None for t in telemetries]
            ),
            latency_seconds=round(
                sum(t.latency_seconds for t in telemetries if t), 3
            ),
            budget=driving.result.telemetry.budget if driving.result.telemetry else None,
        )
        result = AgentResult(
            submission_id=model.submission_id,
            status="awaiting_underwriter",
            insured_name=model.insured_name,
            line_of_business=f"multi-line ({lines_text})",
            guidelines_version=GUIDELINES_VERSION,
            extractions=extractions,
            facts=facts,
            injection_flags=list(
                dict.fromkeys(
                    flag for a in analyses for flag in a.result.injection_flags
                )
            ),
            triage=triage,
            cross_check=driving.result.cross_check,
            guidelines=guidelines,
            vector_store=driving.result.vector_store,
            diagnosis=diagnosis,
            options=driving.result.options,
            drafts=drafts,
            verification=verification,
            review=review,
            reviewer_blocked=reviewer_blocked,
            validation=validation,
            autonomy=autonomy,
            information_request=info,
            needs_information=bool(info and info.needed),
            telemetry=telemetry,
            trace=trace,
            multi_line=True,
            line_analyses=analyses,
            blocking_lines=bundle_blocking_lines(analyses),
        )
        self._get_store().save(
            ApprovalRecord(result=result, submission=self._archive_documents(model))
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

    def _prior_submission_for(
        self, model: SubmissionInput, prior_records: list[ApprovalRecord]
    ) -> dict | None:
        """The insured's most recent prior stored submission **for
        this line**, as the renewal diffing consumes it: the stored
        payload plus the prior run's computed facts and triage
        (``diffing.py``).

        The match is keyed by line — a property renewal diffs against
        the last property submission, never against the account's
        cyber history. A prior *bundle* contributes its matching
        line (payload rebuilt by the same split that produced it,
        facts from that line's stored sub-analysis); a prior
        single-line submission contributes itself when its line
        matches. ``prior_records`` is store-ordered (oldest first)
        and already excludes this submission id, so the last match
        is the most recent prior. None when this insured has never
        submitted this line — a first submission has nothing to
        diff against.
        """
        from .facts import normalize_line

        if not model.insured_name:
            return None
        target = normalize_line(model.line_of_business)
        for record in reversed(prior_records):
            insured = record.result.insured_name or record.submission.get(
                "insured_name", ""
            )
            if insured != model.insured_name or not record.submission:
                continue
            payload = record.submission
            if payload.get("lines"):
                try:
                    prior_model = SubmissionInput.model_validate(payload)
                except Exception:
                    continue
                for sub, analysis in zip(
                    split_bundle(prior_model), record.result.line_analyses
                ):
                    if normalize_line(sub.line_of_business) != target:
                        continue
                    return {
                        "submission_id": analysis.submission_id or sub.submission_id,
                        "submitted_at": sub.submitted_at,
                        "outcome": analysis.result.triage.outcome,
                        "risk_tier": analysis.result.triage.risk_tier,
                        "submission": sub.model_dump(mode="json"),
                        "facts": analysis.result.facts.model_dump(mode="json"),
                    }
                continue
            if normalize_line(payload.get("line_of_business", "")) != target:
                continue
            return {
                "submission_id": record.result.submission_id,
                "submitted_at": payload.get("submitted_at", ""),
                "outcome": record.result.triage.outcome,
                "risk_tier": record.result.triage.risk_tier,
                "submission": payload,
                "facts": record.result.facts.model_dump(mode="json"),
            }
        return None

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
        record.result.decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Output routing: by default there is NO external action. When
        # the operator configures ACTION_WEBHOOK_URL, the approved
        # packet is POSTed to that endpoint (the customer's system of
        # choice) and the attempt is recorded in the delivery ledger
        # — a failed dispatch leaves the approval standing, says so
        # on the result, and can be retried (see retry_dispatch).
        record.result.external_action_taken = False
        entry = _dispatch_once(record)
        if entry is not None:
            record.dispatch_attempts.append(entry)
            self._sync_dispatch(record)
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
        record.result.decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            submission_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
        )
        return record.result

    def _sync_dispatch(self, record: ApprovalRecord) -> None:
        """Project the delivery ledger onto the record + result: the
        latest attempt's outcome is the dispatch status, and the full
        ledger rides on the result where operators read it."""
        if not record.dispatch_attempts:
            return
        latest = record.dispatch_attempts[-1]
        record.dispatch_status = latest["status"]
        record.result.dispatch_status = latest["status"]
        record.result.dispatch_attempts = [
            DispatchAttempt.model_validate(entry)
            for entry in record.dispatch_attempts
        ]
        if latest["status"] == "sent":
            record.result.external_action_taken = True

    def retry_dispatch(self, submission_id: str) -> AgentResult:
        """Retry a failed approval-webhook dispatch.

        Only an approved record whose latest dispatch failed has
        anything to retry; the retry is a new ledger attempt (with
        the schedule's next backoff recorded on it), and a successful
        retry flips ``external_action_taken`` on the result. The
        approval itself is untouched — output routing never rewrites
        a human decision.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        if record.result.status != "approved":
            raise ValueError(
                f"Submission {submission_id} is not approved "
                f"(status: {record.result.status}) — only an approved "
                "submission has a dispatch to retry"
            )
        if record.dispatch_status != "failed":
            raise ValueError(
                f"Dispatch for {submission_id} is not in a failed state "
                f"(status: {record.dispatch_status}) — nothing to retry"
            )
        entry = _dispatch_once(record)
        if entry is None:
            raise ValueError(
                "ACTION_WEBHOOK_URL is not configured — the failed "
                "dispatch cannot be retried until it is set"
            )
        record.dispatch_attempts.append(entry)
        self._sync_dispatch(record)
        store.save(record)
        return record.result

    def dispatch_ledger(self, submission_id: str) -> dict | None:
        """The delivery ledger for one submission, or None when the
        submission is unknown."""
        record = self._get_store().get(submission_id)
        if record is None:
            return None
        return {
            "submission_id": submission_id,
            "dispatch_status": record.dispatch_status,
            "attempts": record.dispatch_attempts,
        }

    def dispatch_ops(self) -> dict:
        """The dispatch operations view: every record's webhook
        ledger read as one surface — attempts, failures (oldest
        first), and which failures are due a retry under the
        recorded backoff schedule (``insights.dispatch_overview``)."""
        return dispatch_overview(self._get_store().all_records())

    def retry_failed_dispatches(self) -> list[dict]:
        """Retry every record whose latest dispatch failed.

        The bulk companion to :meth:`retry_dispatch`: one outcome
        per failed record — the new dispatch status, or the error
        that left it failed (receiver still down, webhook since
        unconfigured). Records that are not failed are untouched,
        and one record's failure never stops the rest.
        """
        outcomes: list[dict] = []
        for record in self._get_store().all_records():
            if record.dispatch_status != "failed":
                continue
            submission_id = record.result.submission_id
            try:
                result = self.retry_dispatch(submission_id)
            except ValueError as exc:
                outcomes.append({"submission_id": submission_id, "error": str(exc)})
                continue
            outcomes.append(
                {
                    "submission_id": submission_id,
                    "dispatch_status": result.dispatch_status,
                    "attempts": len(result.dispatch_attempts),
                }
            )
        return outcomes

    def get(self, submission_id: str) -> AgentResult | None:
        record = self._get_store().get(submission_id)
        return record.result if record else None

    def inbox(self) -> dict:
        """The intake queue grouped by insured and by broker.

        Each group carries its submission count and the latest
        submission's outcome, risk tier, readiness, and status —
        the scan an intake lead does to spot repeat insureds and
        brokers with several cases in flight.
        """
        records = self._get_store().all_records()
        return {
            "insureds": inbox_groups(records, "insured"),
            "brokers": inbox_groups(records, "broker"),
        }

    def scorecards(self) -> dict:
        """History scorecards for every insured and broker in the store."""
        return all_scorecards(self._get_store().all_records())

    def scorecard(self, kind: str, name: str) -> dict | None:
        """One party's scorecard (``kind`` is "insured" or "broker")."""
        return scorecard(self._get_store().all_records(), kind, name)

    def broker_quality_report(self, broker_name: str) -> dict | None:
        """The broker submission-quality report (trend, discrepancy
        rate, top missing elements) with its drafted text — composed
        from the store, never sent."""
        return broker_quality_report(self._get_store().all_records(), broker_name)

    def metrics(self) -> dict:
        """Operational aggregates over the store: run/decision counts,
        outcome mix, model usage, cost, and latency."""
        return compute_metrics(self._get_store().all_records())

    def queue(self) -> dict:
        """The awaiting queue by age bucket and outcome — how long
        undecided submissions have been sitting, and which one has
        been waiting longest."""
        return queue_aging(self._get_store().all_records())

    def draft_status_note(self, submission_id: str) -> dict:
        """Draft the underwriter status note for an SLA-breached case.

        The note exists for one moment: an undecided submission has
        sat past the SLA budget for its risk tier and somebody
        should pick it up. It is composed in code from the record
        and the queue projection (``insights.compose_status_note``),
        guardrail-checked like every other draft
        (``guardrails.validate_status_note``) — a note that fails is
        refused, never stored — and then kept on the record as an
        internal draft. Nothing is sent, to anyone, ever: the
        underwriter reads it in the console and decides.

        Re-drafting replaces the stored note (the case aged, the
        facts may have moved). Raises KeyError for an unknown id
        and ValueError when the submission is not awaiting a
        decision or is still within its SLA budget.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        item = queue_item_for(record)
        if item is None:
            raise ValueError(
                f"Submission {submission_id} is not awaiting an underwriter "
                f"(status: {record.result.status}) — status notes are for the queue"
            )
        if item["sla_state"] != "breached":
            raise ValueError(
                f"Submission {submission_id} is {item['sla_state']} its SLA "
                f"budget ({item['sla_budget_days']} day(s) for tier "
                f"{item['risk_tier']}, age {item['age_days']}) — status notes "
                "are drafted for breached cases"
            )
        now = datetime.now(timezone.utc)
        text = compose_status_note(record, item, now)
        validation = validate_status_note(text, submission_id)
        if not validation.passed:
            failed = "; ".join(
                f"{c.name}: {c.detail}" for c in validation.checks if not c.passed
            )
            raise ValueError(
                "Drafted status note failed guardrail validation and was "
                "not stored: " + failed
            )
        note = {
            "submission_id": submission_id,
            "text": text,
            "drafted_at": now.isoformat(timespec="seconds"),
            "sla": {
                "risk_tier": item["risk_tier"],
                "budget_days": item["sla_budget_days"],
                "age_days": item["age_days"],
                "days_over": item["sla_days_over"],
            },
            "validation": validation.model_dump(mode="json"),
        }
        record.status_note = note
        store.save(record)
        return note

    def status_note(self, submission_id: str) -> dict | None:
        """The stored status note for a submission, or None when no
        note has been drafted (or the submission is unknown)."""
        record = self._get_store().get(submission_id)
        return record.status_note if record else None

    def diff_trigger_calibration(self) -> dict:
        """The diff-trigger calibration report: the stored renewals
        replayed at candidate trigger thresholds (read-only
        what-if), alongside what actually fired at analysis time
        and what underwriters decided on those cases
        (``calibration.py``)."""
        from .calibration import trigger_calibration

        return trigger_calibration(self._get_store().all_records())

    def guideline_change_impact(self, classifier=None) -> dict:
        """Which awaiting submissions would re-triage differently
        under the current guidelines/rules (``retriage.py``).

        Report only — the store is read, never written. ``classifier``
        lets a caller (or a test) simulate a changed rule set.
        """
        from .retriage import guideline_change_impact

        return guideline_change_impact(
            self._get_store().all_records(), classifier=classifier
        )

    def audit_rows(self) -> list[dict]:
        """The audit trail: one flat row per stored submission,
        decisions and telemetry included."""
        return audit_rows(self._get_store().all_records())
