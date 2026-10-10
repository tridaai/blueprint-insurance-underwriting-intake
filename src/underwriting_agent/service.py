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
from datetime import datetime, timedelta, timezone

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
from .config import env_float, env_int, env_str, load_dotenv
from .graph import resume_approval, run_submission
from .guardrails import validate_drafts, validate_status_note
from .guidelines_data import GUIDELINES_VERSION
from .insights import (
    BROKER_QUALITY_DRAFTING_KEY,
    DIGEST_SNAPSHOTS_KEY,
    all_scorecards,
    audit_rows,
    broker_quality_report,
    build_digest_snapshot,
    compose_status_note,
    compute_metrics,
    digest_trend,
    dispatch_overview,
    follow_through_digest,
    follow_through_escalation_days,
    inbox_groups,
    latest_status_note_refusal,
    insured_loss_timeline,
    next_snapshot_id,
    ops_digest,
    prune_snapshot_list,
    queue_aging,
    queue_item_for,
    scorecard,
    scorecards_for,
    status_note_follow_through,
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


def _post_webhook(payload: dict, attempt_no: int) -> dict | None:
    """POST one payload through the output-routing seam and return
    its ledger entry.

    The seam itself: the customer's system of choice, reached at
    ``ACTION_WEBHOOK_URL``, signed with ``ACTION_WEBHOOK_SECRET``
    when set. Every event this service routes out (an approved
    packet, an SLA breach) goes through here, so the signing, the
    timeout, and the ledger-entry shape exist exactly once. Returns
    ``None`` when no URL is configured (the default: no external
    action at all). Otherwise returns the attempt record: attempt
    number, timestamp, outcome (``"sent"`` on 2xx, ``"failed"``
    otherwise), the receiver's HTTP status when it answered, the
    signature identifier when the dispatch was signed, the error
    when it was not, and the backoff the retry schedule assigned
    before this attempt. A failed dispatch never raises. Short
    timeout on purpose: a caller must not hang on a downstream
    system.
    """
    load_dotenv()
    url = env_str("ACTION_WEBHOOK_URL")
    if not url:
        return None
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


def _dispatch_once(record: ApprovalRecord) -> dict | None:
    """One approval-packet dispatch attempt, for the delivery ledger.

    The payload is the approved packet — the customer's system of
    choice takes it from there. The attempt mechanics (signing,
    posting, the ledger entry) live in :func:`_post_webhook`; this
    builds the approval event's body and numbers the attempt within
    the record's approval ledger.
    """
    result = record.result
    payload = {
        "event": "approval",
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
    return _post_webhook(payload, len(record.dispatch_attempts) + 1)


def _sla_breach_snapshot(item: dict) -> dict:
    """The breach facts a notice carries, taken from a queue item —
    the same projection the /queue view renders, so the webhook,
    the queue, and the status note can never disagree about a
    breach."""
    return {
        "submission_id": item["submission_id"],
        "insured_name": item["insured_name"],
        "broker_name": item["broker_name"],
        "outcome": item["outcome"],
        "risk_tier": item["risk_tier"],
        "sla": {
            "budget_days": item["sla_budget_days"],
            "age_days": item["age_days"],
            "days_over": item["sla_days_over"],
        },
    }


def _dispatch_sla_breach(
    record: ApprovalRecord, snapshot: dict, detected_at: str
) -> dict | None:
    """One SLA-breach dispatch attempt, for the record's breach ledger.

    The payload is the breach event — a case sitting undecided past
    its SLA budget — not a packet: nothing is decided, quoted, or
    bound by it; it tells the customer's system of choice that a
    queue case needs eyes. Returns the ledger entry, or ``None``
    when no webhook is configured (nothing fired, nothing recorded).
    """
    payload = {
        "event": "sla_breach",
        "status": "awaiting_underwriter",
        "detected_at": detected_at,
        **snapshot,
    }
    entry = _post_webhook(payload, len(record.sla_breach_attempts) + 1)
    if entry is None:
        return None
    record.sla_breach_attempts.append(entry)
    if record.sla_breach_notice is None:
        record.sla_breach_notice = {
            "first_observed_at": detected_at,
            **snapshot,
            "dispatch_status": entry["status"],
        }
    else:
        record.sla_breach_notice["dispatch_status"] = entry["status"]
    return entry


def _escalation_snapshot(entry: dict, wait_days: int) -> dict:
    """The escalation facts a notice carries, taken from a
    still-waiting entry — the same projection the ops digest and
    the queue render, so the webhook, the digest, and the queue
    can never disagree about an escalation: the case, the first
    call (the acknowledgement it carries), and the second rung
    it crossed."""
    return {
        "submission_id": entry["submission_id"],
        "insured_name": entry["insured_name"],
        "broker_name": entry["broker_name"],
        "risk_tier": entry.get("risk_tier") or "",
        "acknowledgement": entry.get("acknowledgement"),
        "acknowledged_age_days": entry.get("acknowledged_age_days"),
        "escalation_window_days": 2 * wait_days,
    }


def _dispatch_escalation(
    record: ApprovalRecord, snapshot: dict, detected_at: str
) -> dict | None:
    """One acknowledged-escalation dispatch attempt, for the
    record's escalation ledger.

    The payload is the escalation event — an acknowledged case
    still undecided a second window later — not a packet:
    nothing is decided, quoted, or bound by it; it tells the
    customer's system of choice that the second rung has been
    crossed. Returns the ledger entry, or ``None`` when no
    webhook is configured (nothing fired, nothing recorded).
    """
    payload = {
        "event": "acknowledged_escalation",
        "status": "awaiting_underwriter",
        "detected_at": detected_at,
        **snapshot,
    }
    entry = _post_webhook(payload, len(record.escalation_attempts) + 1)
    if entry is None:
        return None
    record.escalation_attempts.append(entry)
    if record.escalation_notice is None:
        record.escalation_notice = {
            "first_observed_at": detected_at,
            **snapshot,
            "dispatch_status": entry["status"],
        }
    else:
        record.escalation_notice["dispatch_status"] = entry["status"]
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
        # Status-note follow-through: an earlier case of this
        # insured breached its SLA and a note was drafted for it;
        # this resubmission is checked against what that note was
        # waiting on (packet section, composed in the graph) and the
        # note's own record is marked with the outcome.
        note_record = self._prior_status_note_record(model, prior_records)
        result = run_submission(
            model,
            backend=backend,
            retriever=retriever,
            history=history,
            priors=priors,
            prior_submission=self._prior_submission_for(model, prior_records),
            prior_status_note=note_record.status_note if note_record else None,
            diff_thresholds=self.effective_diff_thresholds(),
            loss_timeline=insured_loss_timeline(prior_records, model.insured_name),
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=model.submission_id,
        )
        self._get_store().save(
            ApprovalRecord(result=result, submission=self._archive_documents(model))
        )
        if note_record is not None:
            self._record_note_follow_through(note_record, result)
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
        # Status-note follow-through, as in the single-line path:
        # each line's sub-analysis checks the note's snapshot scoped
        # to its own line, and the note's record is marked from the
        # rolled-up bundle result once the bundle is saved.
        note_record = self._prior_status_note_record(model, prior_records)
        timeline = insured_loss_timeline(prior_records, model.insured_name)

        analyses: list[LineAnalysis] = []
        for sub in subs:
            line_result = run_submission(
                sub,
                backend=backend,
                retriever=retriever,
                history=history,
                priors=priors,
                prior_submission=self._prior_submission_for(sub, prior_records),
                prior_status_note=note_record.status_note if note_record else None,
                diff_thresholds=self.effective_diff_thresholds(),
                loss_timeline=timeline,
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
        if note_record is not None:
            self._record_note_follow_through(note_record, result)
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

    def _prior_status_note_record(
        self, model: SubmissionInput, prior_records: list[ApprovalRecord]
    ) -> ApprovalRecord | None:
        """The most recent prior record for this insured that carries
        a drafted status note, or None.

        The note was drafted because an earlier case of this insured
        breached its SLA; when the insured submits again, that note
        is followed through against the new submission (the packet
        section the graph composes, and the marking on the note's
        own record below). Most recent wins — an older note for the
        same insured was superseded by the newer case's note.
        """
        if not model.insured_name:
            return None
        for record in reversed(prior_records):
            if record.status_note is None:
                continue
            insured = record.result.insured_name or record.submission.get(
                "insured_name", ""
            )
            if insured == model.insured_name:
                return record
        return None

    def _record_note_follow_through(
        self, note_record: ApprovalRecord, result: AgentResult
    ) -> None:
        """Mark a followed-through status note on its own record.

        The new analysis checked the note's outstanding snapshot
        against its computed facts (the same computation the packet
        section reports — one function, two surfaces, so they
        cannot disagree). The marking records the state (answered /
        partially answered / still outstanding), the item lists,
        and which submission did the answering; the note's original
        text and snapshot are never rewritten, so a stale note can
        never quietly read as the current state of the case.
        """
        follow = status_note_follow_through(
            note_record.status_note or {},
            missing_elements=result.facts.missing_elements,
            blocking_lines=[
                {
                    "line_of_business": a.line_of_business,
                    "missing_items": list(a.result.facts.missing_elements),
                }
                for a in result.line_analyses
                if a.result.facts.missing_elements
            ],
        )
        note_record.status_note = {
            **(note_record.status_note or {}),
            "follow_through": {
                **follow,
                "by_submission_id": result.submission_id,
                "checked_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
        }
        self._get_store().save(note_record)

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
        recorded backoff schedule — plus the SLA-breach notices
        whose own ledger exhausted its attempts undelivered
        (``insights.dispatch_overview``)."""
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

    def notify_sla_breaches(self, now: datetime | None = None) -> dict:
        """Fire the ``sla_breach`` webhook for newly observed breaches.

        The queue flags breaches and the status note drafts for
        them, but until now nobody outside this service was told.
        This is the telling, through the same output-routing seam
        as approvals (``ACTION_WEBHOOK_URL``, HMAC-signed when
        ``ACTION_WEBHOOK_SECRET`` is set): the first time the queue
        projection observes a case past its SLA budget, one
        ``sla_breach`` event is POSTed and the notice is recorded
        on the record — the record of the notice is the dedupe, so
        re-checking the queue never re-fires for a submission,
        however long it sits breached. A failed firing is a ledger
        entry on the record's breach ledger, retryable by an
        operator (see :meth:`retry_sla_breach_notice`); it is never
        silently re-fired by a later check.

        The crossing is computed from the queue projection at call
        time — there is no timer in the service; an operator (or a
        scheduler hitting the endpoint) runs the check. With no
        webhook configured nothing fires and nothing is recorded:
        the breaches are reported as observed-but-unannounced, so
        configuring the webhook later still announces them.
        """
        now = now or datetime.now(timezone.utc)
        detected_at = now.isoformat(timespec="seconds")
        store = self._get_store()
        records = store.all_records()
        queue = queue_aging(
            records, now, sla_budgets_map=self.effective_sla_budgets()
        )
        by_id = {r.result.submission_id: r for r in records}
        fired: list[dict] = []
        already_notified: list[str] = []
        unannounced: list[str] = []
        for item in queue["breaches"]:
            record = by_id.get(item["submission_id"])
            if record is None:
                continue
            if record.sla_breach_notice is not None:
                already_notified.append(item["submission_id"])
                continue
            entry = _dispatch_sla_breach(
                record, _sla_breach_snapshot(item), detected_at
            )
            if entry is None:
                unannounced.append(item["submission_id"])
                continue
            store.save(record)
            fired.append(
                {
                    "submission_id": item["submission_id"],
                    "dispatch_status": entry["status"],
                }
            )
        return {
            "generated_at": detected_at,
            "breaches": queue["breach_count"],
            "fired": fired,
            "already_notified": already_notified,
            "unannounced_no_webhook": unannounced,
        }

    def retry_sla_breach_notice(self, submission_id: str) -> dict:
        """Retry a failed SLA-breach dispatch.

        The notice fired once (see :meth:`notify_sla_breaches`) and
        the receiver did not take it; the retry is a new attempt on
        the record's breach ledger carrying the original breach
        snapshot (the breach that was observed, at the age it was
        observed — a retry does not rewrite history with today's
        age). Only a notice whose latest attempt failed has
        anything to retry. The queue case itself is untouched —
        output routing never changes a case's state.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        notice = record.sla_breach_notice
        if notice is None:
            raise ValueError(
                f"No SLA breach notice has fired for {submission_id} — "
                "nothing to retry (run the breach check first)"
            )
        if not record.sla_breach_attempts or (
            record.sla_breach_attempts[-1]["status"] != "failed"
        ):
            raise ValueError(
                f"SLA breach notice for {submission_id} is not in a failed "
                f"state (status: {notice.get('dispatch_status')}) — nothing to retry"
            )
        snapshot = {
            key: notice[key]
            for key in (
                "submission_id",
                "insured_name",
                "broker_name",
                "outcome",
                "risk_tier",
                "sla",
            )
        }
        entry = _dispatch_sla_breach(
            record, snapshot, notice["first_observed_at"]
        )
        if entry is None:
            raise ValueError(
                "ACTION_WEBHOOK_URL is not configured — the failed breach "
                "notice cannot be retried until it is set"
            )
        store.save(record)
        return {
            "submission_id": submission_id,
            "notice": record.sla_breach_notice,
            "attempts": record.sla_breach_attempts,
        }

    def notify_acknowledged_escalations(
        self, now: datetime | None = None
    ) -> dict:
        """Fire the ``acknowledged_escalation`` webhook for cases
        on the second rung.

        The first rung told the customer's system of choice that
        a case breached; an operator acknowledged the case — the
        call happened — and a second follow-through window later
        the file still has not moved. Every in-app surface flags
        that (the ops digest's still-waiting section, the queue's
        escalated acknowledgements), but until now nothing left
        the building. This is the telling, through the same
        output-routing seam as approvals and breaches
        (``ACTION_WEBHOOK_URL``, HMAC-signed when
        ``ACTION_WEBHOOK_SECRET`` is set): the first time the
        still-waiting projection observes a case past the second
        rung, one ``acknowledged_escalation`` event is POSTed and
        the notice is recorded on the record — the record of the
        notice is the dedupe, one firing per acknowledgement
        episode (an acknowledgement is recorded once, so one
        episode per record), however often the check runs. A
        failed firing is a ledger entry on the record's
        escalation ledger, retryable by an operator (see
        :meth:`retry_escalation_notice`); its exhaustion joins
        the dispatch worklist beside the breach notices'.

        The crossing is computed from the still-waiting
        projection at call time — there is no timer in the
        service; an operator (or a scheduler hitting the
        endpoint) runs the check. With no webhook configured
        nothing fires and nothing is recorded: the escalations
        are reported as observed-but-unannounced, so configuring
        the webhook later still announces them.
        """
        from .insights import (
            _acknowledged_still_waiting,
            _exhausted_breach_notices,
        )

        now = now or datetime.now(timezone.utc)
        detected_at = now.isoformat(timespec="seconds")
        store = self._get_store()
        records = store.all_records()
        wait_days = follow_through_escalation_days()
        acknowledged = _exhausted_breach_notices(
            records,
            now,
            env_int("DISPATCH_ATTEMPT_BUDGET", 8),
            acknowledged=True,
        )
        escalated = [
            entry
            for entry in _acknowledged_still_waiting(
                acknowledged, now, wait_days
            )
            if entry["escalated"]
        ]
        by_id = {r.result.submission_id: r for r in records}
        fired: list[dict] = []
        already_notified: list[str] = []
        unannounced: list[str] = []
        for entry in escalated:
            record = by_id.get(entry["submission_id"])
            if record is None:
                continue
            if record.escalation_notice is not None:
                already_notified.append(entry["submission_id"])
                continue
            attempt = _dispatch_escalation(
                record, _escalation_snapshot(entry, wait_days), detected_at
            )
            if attempt is None:
                unannounced.append(entry["submission_id"])
                continue
            store.save(record)
            fired.append(
                {
                    "submission_id": entry["submission_id"],
                    "dispatch_status": attempt["status"],
                }
            )
        return {
            "generated_at": detected_at,
            "escalations": len(escalated),
            "fired": fired,
            "already_notified": already_notified,
            "unannounced_no_webhook": unannounced,
        }

    def notify_sla_notifications(
        self, now: datetime | None = None
    ) -> dict:
        """One sweep for both tellings: breaches, then escalations.

        Operators run the breach check and the escalation check
        together every morning — they are the two rungs of one
        ladder (a case nobody told about; a case told, worked,
        and still waiting) — so the morning sweep runs both and
        reports both firings in one response: the breach report
        as :meth:`notify_sla_breaches` returns it, with the
        escalation report (see
        :meth:`notify_acknowledged_escalations`) riding under
        ``escalations``. Each telling keeps its own dedupe and
        ledger — the breach notice on its record, the escalation
        notice on its — and the sweep is exactly the two checks
        run back to back on the same clock, nothing merged: a
        case that crosses both rungs between sweeps is announced
        on both, in ladder order. The separate escalation
        endpoint stays for schedulers that want only that rung.
        """
        now = now or datetime.now(timezone.utc)
        report = self.notify_sla_breaches(now)
        report["escalations"] = self.notify_acknowledged_escalations(now)
        return report

    def retry_escalation_notice(self, submission_id: str) -> dict:
        """Retry a failed acknowledged-escalation dispatch.

        The notice fired once (see
        :meth:`notify_acknowledged_escalations`) and the receiver
        did not take it; the retry is a new attempt on the
        record's escalation ledger carrying the original
        escalation snapshot (the crossing that was observed, at
        the acknowledgement age it was observed — a retry does
        not rewrite the episode with today's age). Only a notice
        whose latest attempt failed has anything to retry. The
        queue case itself is untouched — output routing never
        changes a case's state.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(
                f"Unknown submission_id: {submission_id} (analyze it first)"
            )
        notice = record.escalation_notice
        if notice is None:
            raise ValueError(
                f"No escalation notice has fired for {submission_id} — "
                "nothing to retry (run the escalation check first)"
            )
        if not record.escalation_attempts or (
            record.escalation_attempts[-1]["status"] != "failed"
        ):
            raise ValueError(
                f"Escalation notice for {submission_id} is not in a "
                f"failed state (status: {notice.get('dispatch_status')}) "
                "— nothing to retry"
            )
        snapshot = {
            key: notice[key]
            for key in (
                "submission_id",
                "insured_name",
                "broker_name",
                "risk_tier",
                "acknowledgement",
                "acknowledged_age_days",
                "escalation_window_days",
            )
        }
        entry = _dispatch_escalation(
            record, snapshot, notice["first_observed_at"]
        )
        if entry is None:
            raise ValueError(
                "ACTION_WEBHOOK_URL is not configured — the failed "
                "escalation notice cannot be retried until it is set"
            )
        store.save(record)
        return {
            "submission_id": submission_id,
            "notice": record.escalation_notice,
            "attempts": record.escalation_attempts,
        }

    def acknowledged_escalation(self, submission_id: str) -> dict | None:
        """The escalation notice + delivery ledger for one
        submission, or None when the submission is unknown."""
        record = self._get_store().get(submission_id)
        if record is None:
            return None
        return {
            "submission_id": submission_id,
            "notice": record.escalation_notice,
            "attempts": record.escalation_attempts,
        }

    def acknowledge_exhausted_breach_notice(
        self,
        submission_id: str,
        actor: str,
        note: str = "",
        now: datetime | None = None,
    ) -> dict:
        """Acknowledge an exhausted SLA-breach notice as worked.

        The exhaustion worklist's whole point is the human call
        (see ``insights._exhausted_breach_notices``): the notice
        never landed, the case is still undecided, and somebody
        has to pick it up outside the webhook. This records that
        somebody did — who, when, and their note — on the record
        as ``breach_notice_acknowledgement``. The acknowledgement
        marks the notice *worked*: it leaves the exhaustion
        worklist and the digest's count, and stays readable on
        the acknowledged list with the record. It changes nothing
        else: the delivery ledger is never rewritten (the webhook
        still never landed, and the record keeps saying so), the
        case's queue state is untouched, and nothing is sent.

        Only a notice that is exhausted *now* can be acknowledged
        — the same computation the worklist reads, so what can be
        acknowledged is exactly what the worklist shows. A notice
        already acknowledged cannot be acknowledged again: the
        first acknowledgement is the audit, and a second would
        blur whose call it was.
        """
        from .insights import _exhausted_breach_notices

        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(
                f"Unknown submission_id: {submission_id} (analyze it first)"
            )
        if record.breach_notice_acknowledgement is not None:
            ack = record.breach_notice_acknowledgement
            raise ValueError(
                f"The breach notice for {submission_id} was already "
                f"acknowledged by {ack.get('acknowledged_by')} at "
                f"{ack.get('acknowledged_at')} — the first "
                "acknowledgement is the record of the call"
            )
        now = now or datetime.now(timezone.utc)
        exhausted = _exhausted_breach_notices(
            [record], now, env_int("DISPATCH_ATTEMPT_BUDGET", 8)
        )
        if not exhausted:
            raise ValueError(
                f"The breach notice for {submission_id} is not "
                "exhausted — acknowledgement records the human "
                "pickup of a notice whose delivery ledger spent "
                "its attempts on a case still awaiting a decision, "
                "and this notice is not in that state"
            )
        acknowledgement = {
            "acknowledged_by": actor,
            "acknowledged_at": now.isoformat(timespec="seconds"),
            "note": note,
        }
        record.breach_notice_acknowledgement = acknowledgement
        store.save(record)
        return {
            "submission_id": submission_id,
            "acknowledgement": acknowledgement,
            "notice": record.sla_breach_notice,
            "attempts": record.sla_breach_attempts,
        }

    def mark_escalated_note_worked(
        self,
        submission_id: str,
        actor: str,
        note: str = "",
        now: datetime | None = None,
    ) -> dict:
        """Record that an escalated follow-through note was worked.

        The follow-through escalation's whole point is the human
        touch (see ``insights.follow_through_digest``): a status
        note's loop has been open past the escalation window and
        somebody has to chase what it is waiting on. This records
        that somebody did — who, when, and their note — on the
        note itself, as the latest of its ``escalation_touches``.
        The touch **quiets** the escalation for one escalation
        window from the touch: the note leaves the digest's
        escalated subset and count, the ops digest's
        follow-through count, and the queue's escalated-note
        flags — every surface reads the one computation — and
        rides in the digest's ``worked`` subset with the touch
        attached. It changes nothing else: the note's text, its
        outstanding snapshot, and its follow-through marking are
        untouched, the underlying case is untouched, and nothing
        is sent.

        Only a note that is escalated *now* can be marked worked
        — the same computation the digest reads, so what can be
        worked is exactly what the surfaces flag. A note whose
        escalation an earlier touch is still quieting cannot be
        touched again until it re-escalates: while the window
        runs, the latest touch is the record of the work, and a
        second stamp would blur whose touch quieted it. When the
        window lapses with the loop still open, the note
        escalates again and a further touch can be recorded —
        the worked state must not become a quieter place to rot.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(
                f"Unknown submission_id: {submission_id} (analyze it first)"
            )
        if not record.status_note:
            raise ValueError(
                f"No status note has been drafted for {submission_id} "
                "— there is no follow-through loop to work"
            )
        now = now or datetime.now(timezone.utc)
        digest = follow_through_digest([record], now)
        entry = digest["entries"][0] if digest["entries"] else None
        if entry is None or not entry["escalated"]:
            if entry is not None and entry["quieted_by_work"]:
                touch = entry["worked"] or {}
                raise ValueError(
                    f"The follow-through note for {submission_id} was "
                    f"already worked by {touch.get('worked_by')} at "
                    f"{touch.get('worked_at')} — that touch quiets "
                    "the escalation for one window; a further touch "
                    "can be recorded once the note re-escalates"
                )
            state = entry["state"] if entry else "unknown"
            raise ValueError(
                f"The follow-through note for {submission_id} is not "
                f"escalated (state: {state}) — a worked touch is "
                "recorded against an escalated note, and this "
                "note's loop is either answered or still inside "
                "its escalation window"
            )
        touch = {
            "worked_by": actor,
            "worked_at": now.isoformat(timespec="seconds"),
            "note": note,
        }
        touches = list(record.status_note.get("escalation_touches") or [])
        touches.append(touch)
        record.status_note["escalation_touches"] = touches
        store.save(record)
        return {
            "submission_id": submission_id,
            "touch": touch,
            "touches": touches,
            "quiets_until": (
                now.date() + timedelta(days=digest["escalation_days"])
            ).isoformat(),
            "state": entry["state"],
            "note_age_days": entry["note_age_days"],
        }

    def sla_breach(self, submission_id: str) -> dict | None:
        """The SLA-breach notice + delivery ledger for one submission,
        or None when the submission is unknown."""
        record = self._get_store().get(submission_id)
        if record is None:
            return None
        return {
            "submission_id": submission_id,
            "notice": record.sla_breach_notice,
            "attempts": record.sla_breach_attempts,
        }

    def get(self, submission_id: str) -> AgentResult | None:
        record = self._get_store().get(submission_id)
        return record.result if record else None

    def inbox(self) -> dict:
        """The intake queue grouped by insured and by broker.

        Each group carries its submission count and the latest
        submission's outcome, risk tier, readiness, and status —
        the scan an intake lead does to spot repeat insureds and
        brokers with several cases in flight. The inbox also
        carries the **follow-through digest**: every drafted
        status note by state (answered / partially / still
        outstanding, oldest outstanding first), so a resubmission
        visibly closes — or fails to close — the loop a breached
        case's note opened.
        """
        records = self._get_store().all_records()
        return {
            "insureds": inbox_groups(records, "insured"),
            "brokers": inbox_groups(records, "broker"),
            "follow_through": follow_through_digest(records),
        }

    def scorecards(self) -> dict:
        """History scorecards for every insured and broker in the store."""
        return all_scorecards(self._get_store().all_records())

    def scorecard(self, kind: str, name: str) -> dict | None:
        """One party's scorecard (``kind`` is "insured" or "broker")."""
        return scorecard(self._get_store().all_records(), kind, name)

    def loss_timeline(self, insured_name: str) -> dict | None:
        """One insured's loss-development timeline: their stored
        per-line renewal diffs chained into per-claim event streams
        (appeared / developed / no longer listed) — the whole arc,
        not just the last pair (``insights.insured_loss_timeline``)."""
        return insured_loss_timeline(self._get_store().all_records(), insured_name)

    def broker_quality_report(self, broker_name: str) -> dict | None:
        """The broker submission-quality report (trend, discrepancy
        rate, top missing elements) with its drafted text — composed
        from the store, never sent.

        The draft is composed under the drafting discipline (see
        ``insights._discipline_broker_draft``): guardrail-checked,
        steered once on a wording-only failure, refused on
        substance. Each drafting is recorded as an event on the
        ``broker_quality_drafting`` document — clean, repaired,
        or refused, with the families and failed checks — so the
        report's repairs and refusals join the drafting-repair
        family in ``/metrics`` beside the letter advisories and
        the status-note repairs, named as their own source. The
        report is drafted on read, so each read is one drafting
        and one event; the computed sections are unaffected.
        """
        report = broker_quality_report(
            self._get_store().all_records(), broker_name
        )
        if report is not None:
            self._record_broker_quality_drafting(report)
        return report

    def _record_broker_quality_drafting(self, report: dict) -> None:
        """Append one drafting event for a broker quality report
        (see :meth:`broker_quality_report`): the outcome the
        discipline reached — refused if the report carries a
        refusal, repaired if it carries a repair, clean
        otherwise — with the repair families and failed checks
        the aggregate counts."""
        refusal = report.get("draft_refusal") or {}
        repair = report.get("draft_repair") or {}
        outcome = (
            "refused" if refusal else "repaired" if repair else "clean"
        )
        store = self._get_store()
        document = store.get_document(BROKER_QUALITY_DRAFTING_KEY) or {}
        events = list(document.get("events") or [])
        events.append(
            {
                "broker": report.get("broker", ""),
                "drafted_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "outcome": outcome,
                "repair_families": (
                    repair.get("repair_families")
                    or refusal.get("repair_families")
                    or []
                ),
                "failed_checks": refusal.get("failed_checks") or [],
                "steered": bool(refusal.get("steered")),
            }
        )
        document["events"] = events
        store.save_document(BROKER_QUALITY_DRAFTING_KEY, document)

    def broker_quality_drafting_events(self) -> list[dict]:
        """Every recorded broker-quality drafting event, in the
        order drafted (see :meth:`_record_broker_quality_drafting`)."""
        document = (
            self._get_store().get_document(BROKER_QUALITY_DRAFTING_KEY)
            or {}
        )
        return list(document.get("events") or [])

    def broker_quality_drafting_prunes(self) -> list[dict]:
        """Every prune recorded on the drafting-events document,
        in the order they happened."""
        document = (
            self._get_store().get_document(BROKER_QUALITY_DRAFTING_KEY)
            or {}
        )
        return list(document.get("prunes") or [])

    def prune_broker_quality_drafting_events(
        self, actor: str, keep: int
    ) -> dict:
        """Prune the drafting events to the newest ``keep`` — the
        snapshot documents' retention discipline on the one
        document that grows per read rather than per ask. The
        events are the broker drafting counts' only source, so
        the prune is recorded on the document itself (who, when,
        what was asked, how many left and the range they spanned,
        what survived) before the survivors are written back: a
        step down in the counts afterwards explains itself. A
        prune that would remove nothing writes nothing and
        records nothing — it just reports. ``keep`` below 1 is
        refused by the shared computation
        (``insights.prune_drafting_event_list``)."""
        from .insights import (
            drafting_event_prune_entry,
            prune_drafting_event_list,
        )

        store = self._get_store()
        document = store.get_document(BROKER_QUALITY_DRAFTING_KEY) or {}
        events = list(document.get("events") or [])
        kept, pruned = prune_drafting_event_list(events, keep)
        entry = None
        if pruned:
            entry = drafting_event_prune_entry(
                actor=actor,
                keep=keep,
                pruned=pruned,
                kept=kept,
                pruned_at=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            )
            document["events"] = kept
            document["prunes"] = list(document.get("prunes") or []) + [
                entry
            ]
            store.save_document(BROKER_QUALITY_DRAFTING_KEY, document)
        return {
            "kept": len(kept),
            "pruned_count": len(pruned),
            "pruned_oldest": (
                pruned[0].get("drafted_at", "") if pruned else ""
            ),
            "pruned_newest": (
                pruned[-1].get("drafted_at", "") if pruned else ""
            ),
            "surviving_first": (
                kept[0].get("drafted_at", "") if kept else ""
            ),
            "surviving_last": (
                kept[-1].get("drafted_at", "") if kept else ""
            ),
            "prune": entry,
        }

    def preview_prune_broker_quality_drafting_events(
        self, keep: int
    ) -> dict:
        """Preview a drafting-event prune (see
        :meth:`prune_broker_quality_drafting_events`): how many
        events a keep-last-N would remove and the drafted-at
        ranges on both sides of the cut, before anything is
        written — the document behind the drafting counts is
        pruned against evidence, not memory."""
        from .insights import drafting_event_prune_preview

        return drafting_event_prune_preview(
            self.broker_quality_drafting_events(), keep
        )

    def metrics(self) -> dict:
        """Operational aggregates over the store: run/decision counts,
        outcome mix, the letter-advisory trend (borderline drafting
        quality, aggregated from the runs' recorded advisories),
        the drafting-repair family (status-note repairs and
        refusals, with the broker quality report's nested as
        their own source), the proposal re-ask family (drafts
        carrying a pruned earlier decision, counted from the
        proposals document, and how the second decision went),
        model usage, cost, and latency."""
        from .proposals import load_proposals

        return compute_metrics(
            self._get_store().all_records(),
            broker_drafting_events=self.broker_quality_drafting_events(),
            proposals=load_proposals(self._get_store()),
        )

    def queue(self) -> dict:
        """The awaiting queue by age bucket and outcome — how long
        undecided submissions have been sitting, and which one has
        been waiting longest. SLA states are measured against the
        budgets in force (the settings record an accepted SLA
        proposal wrote, over the environment). When a digest
        snapshot has been kept, the breach list is trend-aware:
        each breached item is flagged new or continuing against
        the latest snapshot's breached ids, and the view carries
        the movement (new / continuing / cleared) — the worklist
        itself distinguishes a new fire from an old one."""
        view = queue_aging(
            self._get_store().all_records(),
            sla_budgets_map=self.effective_sla_budgets(),
            digest_snapshots=self.ops_digest_snapshots(),
        )
        view["sla_budgets_source"] = self.sla_budget_settings()["source"]
        return view

    def ops_digest(self) -> dict:
        """The operator digest (``insights.ops_digest``): stuck
        dispatches, exhausted breach notices, breached queue
        cases, outstanding/escalated follow-through, and
        calibration proposals nearing or past expiry — one
        composed read over the store, measured against the SLA
        budgets in force. Computed, never sent; reading it
        changes nothing. When a digest snapshot has been kept,
        the digest is trend-aware: its ``movement`` block reads
        the live counts against the latest snapshot's, and the
        headline says what moved."""
        snapshots = self.ops_digest_snapshots()
        return ops_digest(
            self._get_store().all_records(),
            self.calibration_proposals(),
            sla_budgets_map=self.effective_sla_budgets(),
            latest_snapshot=snapshots[-1] if snapshots else None,
            broker_drafting_events=self.broker_quality_drafting_events(),
        )

    def snapshot_ops_digest(self, actor: str) -> dict:
        """Take a stored snapshot of the ops digest as it stands
        now (``insights.build_digest_snapshot``): the headline,
        the counts, and the ids behind each section, kept under
        the taker's name. Snapshots are operator-taken, never on a
        timer — they are what the digest trend reads. Taking one
        changes nothing but the snapshot list itself (the
        document's prune record rides along untouched), and its id
        runs one past the highest number still kept, so an id a
        prune removed is never reissued."""
        store = self._get_store()
        document = store.get_document(DIGEST_SNAPSHOTS_KEY) or {}
        snapshots = list(document.get("snapshots") or [])
        snapshot = build_digest_snapshot(
            self.ops_digest(), next_snapshot_id(snapshots, "snap-"), actor
        )
        snapshots.append(snapshot)
        document["snapshots"] = snapshots
        store.save_document(DIGEST_SNAPSHOTS_KEY, document)
        return snapshot

    def ops_digest_snapshots(self) -> list[dict]:
        """Every stored ops-digest snapshot, in the order taken."""
        document = self._get_store().get_document(DIGEST_SNAPSHOTS_KEY) or {}
        return list(document.get("snapshots") or [])

    def ops_digest_snapshot_prunes(self) -> list[dict]:
        """Every prune recorded on the digest-snapshot document,
        in the order they happened."""
        document = self._get_store().get_document(DIGEST_SNAPSHOTS_KEY) or {}
        return list(document.get("prunes") or [])

    def prune_ops_digest_snapshots(self, actor: str, keep: int) -> dict:
        """Prune the kept digest snapshots to the newest ``keep``
        (see :meth:`_prune_kept_snapshots` — the one retention
        discipline for both kept-report documents)."""
        return self._prune_kept_snapshots(DIGEST_SNAPSHOTS_KEY, actor, keep)

    def preview_prune_ops_digest_snapshots(self, keep: int) -> dict:
        """Preview a digest-snapshot prune (see
        :meth:`_preview_prune_kept_snapshots`): which snapshots a
        keep-last-N would remove and what range would survive,
        before anything is written."""
        return self._preview_prune_kept_snapshots(DIGEST_SNAPSHOTS_KEY, keep)

    def _preview_prune_kept_snapshots(self, key: str, keep: int) -> dict:
        """The prune's dry run, shared by both kept documents.

        Runs the prune's own split over the document as it
        stands (``insights.snapshot_prune_preview``) and reports
        the plan — the ids that would leave, the surviving
        range — writing nothing and recording nothing: no prune
        entry is appended, because a preview pruned nothing. A
        prune that follows on the same document produces exactly
        the previewed removals. ``keep`` below 1 is refused by
        the shared computation, as it is for the prune itself.
        """
        from .insights import snapshot_prune_preview

        document = self._get_store().get_document(key) or {}
        snapshots = list(document.get("snapshots") or [])
        return snapshot_prune_preview(snapshots, keep)

    def _prune_kept_snapshots(self, key: str, actor: str, keep: int) -> dict:
        """Keep-last-N retention for a kept-snapshot document.

        Snapshots accumulate one per operator ask, unboundedly;
        this is the only way one ever leaves. The prune is
        recorded on the document itself (who, when, what was
        asked, what left, the surviving range) before the
        survivors are written back, so a gap in a trend explains
        itself. A prune that would remove nothing writes nothing
        and records nothing — it just reports. ``keep`` below 1
        is refused by the shared computation
        (``insights.prune_snapshot_list``).
        """
        from .insights import snapshot_prune_entry

        store = self._get_store()
        document = store.get_document(key) or {}
        snapshots = list(document.get("snapshots") or [])
        kept, pruned = prune_snapshot_list(snapshots, keep)
        entry = None
        if pruned:
            entry = snapshot_prune_entry(
                actor=actor,
                keep=keep,
                pruned=pruned,
                kept=kept,
                pruned_at=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            )
            document["snapshots"] = kept
            document["prunes"] = list(document.get("prunes") or []) + [entry]
            store.save_document(key, document)
        return {
            "kept": len(kept),
            "pruned_count": len(pruned),
            "pruned_ids": [s.get("snapshot_id", "") for s in pruned],
            "surviving_first": kept[0].get("snapshot_id", "") if kept else "",
            "surviving_last": kept[-1].get("snapshot_id", "") if kept else "",
            "prune": entry,
        }

    def ops_digest_trend(self) -> dict:
        """Movement across the stored digest snapshots
        (``insights.digest_trend``): the headline counts as a
        series, their change from first to latest and since the
        previous snapshot, and which ids entered or cleared each
        section between the latest two. The drafting-events
        document's prune record rides along, so a broker-count
        step down that a prune explains is annotated as
        retention rather than left to read as a drafting
        collapse (see ``insights._prune_step_annotations``)."""
        return digest_trend(
            self.ops_digest_snapshots(),
            drafting_prunes=self.broker_quality_drafting_prunes(),
        )

    def effective_sla_budgets(self) -> dict:
        """The queue SLA budgets in force: the environment budgets,
        overridden per tier by the settings record an accepted SLA
        proposal wrote (``proposals.effective_sla_budgets``). The
        queue projection, the breach check, and the status-note
        drafting all measure cases against these — accepting an
        SLA proposal changes the very next queue read."""
        from .proposals import effective_sla_budgets

        return effective_sla_budgets(self._get_store())

    def sla_budget_settings(self) -> dict:
        """The SLA budgets in force and their provenance: the
        values, whether they come from the environment or an
        accepted proposal, and the settings record's history."""
        from .proposals import sla_budget_settings_view

        return sla_budget_settings_view(self._get_store())

    def sla_budget_calibration(self) -> dict:
        """The SLA-budget calibration report
        (``proposals.sla_budget_calibration``): per tier, the
        decided cases' waits as a distribution and a sweep of
        candidate budgets — breach counts, decision splits, and
        the moved set against the budget in force, marked when it
        is the one-sided evidence an SLA proposal would be drafted
        from. The budgets are the ones in force (settings record
        over environment), each tier's source named. Read-only —
        reading it changes nothing."""
        from .proposals import sla_budget_calibration, sla_budget_sources

        store = self._get_store()
        return sla_budget_calibration(
            store.all_records(),
            self.effective_sla_budgets(),
            budget_sources=sla_budget_sources(store),
        )

    def draft_sla_budget_proposals(self) -> dict:
        """Draft SLA-budget proposals from the decided cases' waits
        (``proposals.draft_sla_proposals``): candidate budgets
        whose moved cases carry one-sided decision evidence become
        stored proposal documents in the same proposals list as
        the threshold proposals. Drafting changes no budget and no
        record — a human still decides."""
        from .proposals import draft_sla_proposals

        store = self._get_store()
        return draft_sla_proposals(store, store.all_records())

    def draft_sla_proposal_from_row(
        self, tier: str, value: int, actor: str
    ) -> dict:
        """Draft one SLA-budget proposal from a chosen calibration
        row (``proposals.draft_sla_proposal_from_row``): the row
        is recomputed against the store as it stands and, when it
        is proposal-ready, stored as a draft carrying the row's
        evidence verbatim, stamped with its origin and its
        drafter. The scan's discipline binds it — one open
        question per tier, a rejected direction never re-asked,
        swept candidates only — and drafting changes no budget:
        a human still decides the proposal it produces."""
        from .proposals import annotate_proposal, draft_sla_proposal_from_row

        store = self._get_store()
        proposal = draft_sla_proposal_from_row(
            store, store.all_records(), tier, value, actor
        )
        return annotate_proposal(proposal)

    def _check_calibration_label(
        self, snapshots: list[dict], label: str
    ) -> None:
        """The label rules every calibration keep answers to.

        A label is an address, so it must name one report: the
        reserved ``current`` is refused outright (it names the
        live side of a diff, a moving target no keep may wear),
        and a label a kept snapshot already carries is refused
        naming the holder and the labels in use. Unlabelled
        keeps are unlimited and checked no further.
        """
        from .proposals import (
            CURRENT_CALIBRATION_LABEL,
            calibration_snapshot_by_label,
            calibration_snapshot_labels,
        )

        if label == CURRENT_CALIBRATION_LABEL:
            raise ValueError(
                "The label 'current' is reserved for the live "
                "side of a calibration diff — it names the "
                "report as it stands now, so no kept snapshot "
                "may carry it."
            )
        if label:
            holder = calibration_snapshot_by_label(snapshots, label)
            if holder is not None:
                in_use = ", ".join(
                    f"'{name}'"
                    for name in calibration_snapshot_labels(snapshots)
                )
                raise ValueError(
                    f"Label '{label}' is already carried by "
                    f"{holder.get('snapshot_id', '')} — kept "
                    "calibration labels are unique, so a citation "
                    f"names one report. Labels in use: {in_use}."
                )

    def snapshot_sla_calibration(self, actor: str, label: str = "") -> dict:
        """Keep a snapshot of the SLA calibration report as it
        stands now (``proposals.build_calibration_snapshot``):
        the budgets in force, the decided evidence weighed, and
        per tier which candidate rows were proposal-ready — dated
        and under the taker's name, with the taker's optional
        ``label`` ("post-budget-change") so a later citation has
        a handle beyond the snapshot id. Operator-kept, never on
        a timer; keeping one changes nothing but the snapshot
        list, and the snapshots are what the calibration diff
        reads.

        A label is an address, so it must name one report: a
        keep whose label a kept snapshot already carries is
        refused, naming the holder and the labels in use —
        round 16 stored labels verbatim and never deduped them,
        which is sound for a display string and unsound for a
        citation; addressing is what changed. Unlabelled keeps
        are unlimited, and a prune that removes the holder frees
        its label, as it frees the id range it pruned. The label
        ``current`` is refused outright: it is reserved for the
        live side of a diff (see
        :meth:`sla_calibration_snapshot_diff`), and a kept report
        wearing it would make the citation name a moving target."""
        from .proposals import (
            CALIBRATION_SNAPSHOTS_KEY,
            build_calibration_snapshot,
        )

        store = self._get_store()
        document = store.get_document(CALIBRATION_SNAPSHOTS_KEY) or {}
        snapshots = list(document.get("snapshots") or [])
        self._check_calibration_label(snapshots, label)
        snapshot = build_calibration_snapshot(
            self.sla_budget_calibration(),
            next_snapshot_id(snapshots, "cal-"),
            actor,
            label=label,
        )
        snapshots.append(snapshot)
        document["snapshots"] = snapshots
        store.save_document(CALIBRATION_SNAPSHOTS_KEY, document)
        return snapshot

    def keep_calibration_current(self, actor: str, label: str) -> dict:
        """Keep the live side of a calibration diff, from the diff.

        A diff against ``current`` that shows movement worth
        citing used to end with the operator keeping the live
        report by hand and re-running the diff to cite it — a
        second computation the citation could not be sure
        matched the one they read. This act closes that loop:
        the live report is shaped once, in the snapshot's own
        shape (``proposals.live_calibration_snapshot``), and
        that very shape is what gets kept — re-stamped with the
        next snapshot id, the keeper's name, the keep moment,
        and the label the operator gives at that moment, and
        marked ``kept_from: "current"`` so the list can tell a
        diff-born keep from a plain one. The label rules are
        every keep's (see :meth:`_check_calibration_label`):
        ``current`` itself refused, a label already carried
        refused. A label is *required* here, where a plain keep
        allows none: this act exists to make a citation, and a
        citation needs its handle at the moment it is made.
        """
        from .proposals import (
            CALIBRATION_SNAPSHOTS_KEY,
            live_calibration_snapshot,
        )

        if not label:
            raise ValueError(
                "Keeping the current side of a diff needs the "
                "label the citation will use — the act exists to "
                "name the live report the operator just read. "
                "For an unlabelled keep, use the plain snapshot."
            )
        store = self._get_store()
        document = store.get_document(CALIBRATION_SNAPSHOTS_KEY) or {}
        snapshots = list(document.get("snapshots") or [])
        self._check_calibration_label(snapshots, label)
        live = live_calibration_snapshot(self.sla_budget_calibration())
        snapshot = {
            **live,
            "snapshot_id": next_snapshot_id(snapshots, "cal-"),
            "taken_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "taken_by": actor,
            "label": label,
            "kept_from": "current",
        }
        snapshots.append(snapshot)
        document["snapshots"] = snapshots
        store.save_document(CALIBRATION_SNAPSHOTS_KEY, document)
        return snapshot

    def rename_sla_calibration_snapshot_label(
        self, from_label: str, to_label: str, actor: str
    ) -> dict:
        """Rename a kept calibration report's label, on the
        snapshot itself.

        Labels are addresses (round 17), and a debate that
        renames its milestone — "pre-change" becomes
        "pre-budget-change" once the team settles its words —
        used to have two bad options: cite the wrong name, or
        re-keep the report under the new label and fork the
        citation (two kept reports, one evidence base, and the
        diff reading a keep against itself). The rename is the
        third: the snapshot's ``label`` changes in place and
        the move is appended to its ``label_history`` (from, to,
        who, when), so the report's evidence — budgets, ready
        rows, its keeper and moment — is untouched and its
        addressing history is auditable. The old label is freed
        at once: a by-label read of it names nothing (404), a
        keep may take it, and the new label resolves to this
        report in the list, the by-label read, and the diff.

        The refusals are the addressing discipline's: the
        ``from`` label must name a kept report (an unknown one
        is refused naming the labels in use); the ``to`` label
        must be a label — renaming to nothing would un-address
        the report, a different act this endpoint does not
        perform; renaming to the label already carried changes
        nothing and is refused as such; and the new label
        answers to every keep's rules
        (:meth:`_check_calibration_label`): ``current`` refused
        as reserved, a label another kept report carries refused
        naming the holder."""
        from .proposals import (
            CALIBRATION_SNAPSHOTS_KEY,
            calibration_snapshot_by_label,
            calibration_snapshot_labels,
        )

        store = self._get_store()
        document = store.get_document(CALIBRATION_SNAPSHOTS_KEY) or {}
        snapshots = list(document.get("snapshots") or [])
        if not from_label:
            raise ValueError(
                "A rename names the label the kept report "
                "carries now — an unlabelled report is cited by "
                "its snapshot id and has no address to move."
            )
        holder = calibration_snapshot_by_label(snapshots, from_label)
        if holder is None:
            in_use = calibration_snapshot_labels(snapshots)
            names = (
                ", ".join(f"'{name}'" for name in in_use)
                if in_use
                else "none"
            )
            raise ValueError(
                f"No kept calibration snapshot carries the "
                f"label '{from_label}'. Labels in use: {names}."
            )
        if not to_label:
            raise ValueError(
                "A rename lands on a new label — renaming "
                f"'{from_label}' to nothing would un-address "
                "the report, and un-addressing is not a rename."
            )
        if to_label == from_label:
            raise ValueError(
                f"{holder.get('snapshot_id', '')} already "
                f"carries the label '{to_label}' — a rename "
                "changes the address, and this one would not."
            )
        self._check_calibration_label(snapshots, to_label)
        holder["label"] = to_label
        history = list(holder.get("label_history") or [])
        history.append(
            {
                "from": from_label,
                "to": to_label,
                "renamed_by": actor,
                "renamed_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        holder["label_history"] = history
        document["snapshots"] = snapshots
        store.save_document(CALIBRATION_SNAPSHOTS_KEY, document)
        return holder

    def sla_calibration_snapshots(
        self, label: str | None = None
    ) -> list[dict]:
        """Every stored SLA-calibration snapshot, in the order
        kept — or, with ``label``, only the kept reports
        carrying it: the citation read (labels are unique among
        kept snapshots, so at most one)."""
        from .proposals import CALIBRATION_SNAPSHOTS_KEY

        document = self._get_store().get_document(CALIBRATION_SNAPSHOTS_KEY) or {}
        snapshots = list(document.get("snapshots") or [])
        if label is None:
            return snapshots
        return [
            s for s in snapshots if (s.get("label") or "") == label
        ]

    def sla_calibration_snapshot_by_label(self, label: str) -> dict | None:
        """The kept snapshot a label cites, or None when no kept
        snapshot carries it — the report must be addressable by
        the handle the debate uses, not only by its id."""
        from .proposals import calibration_snapshot_by_label

        return calibration_snapshot_by_label(
            self.sla_calibration_snapshots(), label
        )

    def sla_calibration_snapshot_prunes(self) -> list[dict]:
        """Every prune recorded on the calibration-snapshot
        document, in the order they happened."""
        from .proposals import CALIBRATION_SNAPSHOTS_KEY

        document = self._get_store().get_document(CALIBRATION_SNAPSHOTS_KEY) or {}
        return list(document.get("prunes") or [])

    def prune_sla_calibration_snapshots(self, actor: str, keep: int) -> dict:
        """Prune the kept calibration snapshots to the newest
        ``keep`` — the same retention discipline as the digest
        snapshots (see :meth:`_prune_kept_snapshots`), on the
        document a budget debate cites."""
        from .proposals import CALIBRATION_SNAPSHOTS_KEY

        return self._prune_kept_snapshots(
            CALIBRATION_SNAPSHOTS_KEY, actor, keep
        )

    def preview_prune_sla_calibration_snapshots(self, keep: int) -> dict:
        """Preview a calibration-snapshot prune (see
        :meth:`_preview_prune_kept_snapshots`): which kept
        reports a keep-last-N would remove and what range would
        survive, before anything is written — the document a
        budget debate cites is pruned against evidence, not
        memory."""
        from .proposals import CALIBRATION_SNAPSHOTS_KEY

        return self._preview_prune_kept_snapshots(
            CALIBRATION_SNAPSHOTS_KEY, keep
        )

    def sla_calibration_snapshot_diff(
        self, from_label: str = "", to_label: str = ""
    ) -> dict:
        """The proposal-ready movement between two kept
        calibration snapshots (``proposals.calibration_snapshot_diff``):
        per tier, which candidate rows became proposal-ready and
        which stopped being, beside the budget and decided-count
        movement. Fewer than two snapshots is not a diff — the
        report says so instead of inventing movement.

        With no selectors the two sides are the two latest kept,
        as always. A debate, though, cites labels, so either side
        may instead be named by its ``from_label`` / ``to_label``:
        a named side resolves to the latest kept snapshot
        carrying that label (an unknown label is refused, naming
        the labels in use), and an unnamed side defaults to its
        positional partner — ``to`` the latest kept, ``from``
        the one kept just before the resolved ``to``. The pair
        then walks in keep order whichever way the selectors
        named them: the diff's older side is the one kept first.
        Both selectors naming the same snapshot is refused; a
        lone selector resolving to the only side there is reads
        as no diff, the same honesty as fewer than two kept.

        ``to_label="current"`` is the reserved selector for the
        debates' commonest question — what moved since a kept
        report, against the report as it stands *now*: the newer
        side is the live calibration report shaped in memory
        (``proposals.live_calibration_snapshot``), cited in the
        diff as ``current``, and nothing is stored to answer it.
        The kept side is the ``from_label`` report, or the
        latest kept when unnamed; with nothing kept there is no
        diff, exactly as with fewer than two kept. ``current``
        can only be the newer side — it names the report as it
        stands, the newest thing there is — so a
        ``from_label="current"`` is refused."""
        from .proposals import (
            CURRENT_CALIBRATION_LABEL,
            calibration_snapshot_by_label,
            calibration_snapshot_diff,
            calibration_snapshot_labels,
            live_calibration_snapshot,
        )

        snapshots = self.sla_calibration_snapshots()
        if from_label == CURRENT_CALIBRATION_LABEL:
            raise ValueError(
                "The label 'current' names the live calibration "
                "report — the newest side there is — so it can "
                "only be a diff's 'to' side, never its 'from'."
            )
        if to_label == CURRENT_CALIBRATION_LABEL:
            if not snapshots:
                return {"snapshot_count": 0, "diff": None}
            if from_label:
                older = calibration_snapshot_by_label(
                    snapshots, from_label
                )
                if older is None:
                    in_use = calibration_snapshot_labels(snapshots)
                    names = (
                        ", ".join(f"'{name}'" for name in in_use)
                        if in_use
                        else "none"
                    )
                    raise ValueError(
                        f"No kept calibration snapshot carries "
                        f"the from label '{from_label}'. Labels "
                        f"in use: {names}."
                    )
            else:
                older = snapshots[-1]
            return {
                "snapshot_count": len(snapshots),
                "diff": calibration_snapshot_diff(
                    older,
                    live_calibration_snapshot(
                        self.sla_budget_calibration()
                    ),
                ),
            }
        if not from_label and not to_label:
            if len(snapshots) < 2:
                return {"snapshot_count": len(snapshots), "diff": None}
            older, newer = snapshots[-2], snapshots[-1]
        else:
            if not snapshots:
                return {"snapshot_count": 0, "diff": None}

            def resolve(label: str, role: str) -> dict:
                found = calibration_snapshot_by_label(snapshots, label)
                if found is None:
                    in_use = calibration_snapshot_labels(snapshots)
                    names = (
                        ", ".join(f"'{name}'" for name in in_use)
                        if in_use
                        else "none"
                    )
                    raise ValueError(
                        f"No kept calibration snapshot carries the "
                        f"{role} label '{label}'. Labels in use: "
                        f"{names}."
                    )
                return found

            positions = {
                s.get("snapshot_id", ""): i
                for i, s in enumerate(snapshots)
            }
            newer = (
                resolve(to_label, "to") if to_label else snapshots[-1]
            )
            if from_label:
                older = resolve(from_label, "from")
            else:
                before = positions[newer.get("snapshot_id", "")] - 1
                older = snapshots[before] if before >= 0 else None
            if older is None:
                return {
                    "snapshot_count": len(snapshots),
                    "diff": None,
                }
            if older.get("snapshot_id") == newer.get("snapshot_id"):
                if from_label and to_label:
                    raise ValueError(
                        f"Both labels name {older.get('snapshot_id', '')} "
                        "— a diff needs two different kept reports."
                    )
                return {
                    "snapshot_count": len(snapshots),
                    "diff": None,
                }
            if positions[older.get("snapshot_id", "")] > positions[
                newer.get("snapshot_id", "")
            ]:
                older, newer = newer, older
        return {
            "snapshot_count": len(snapshots),
            "diff": calibration_snapshot_diff(older, newer),
        }

    def draft_status_note(self, submission_id: str) -> dict:
        """Draft the underwriter status note for an SLA-breached case.

        The note exists for one moment: an undecided submission has
        sat past the SLA budget for its risk tier and somebody
        should pick it up. It is composed in code from the record
        and the queue projection (``insights.compose_status_note``),
        guardrail-checked like every other draft
        (``guardrails.validate_status_note``), and then kept on the
        record as an internal draft. Nothing is sent, to anyone,
        ever: the underwriter reads it in the console and decides.

        A note that fails on **wording alone** — record prose the
        composition interpolated (a gap source, a missing-element
        label) carrying a forward-motion, review-commitment,
        soft-time, or authority phrasing — gets one steered redraft
        before refusal (``letter_repair.steer_status_note``: the
        offending sentence rewritten into the family's approved
        note shape, the rewrite revalidated, the repair recorded
        on the note with the original validation, the guidance
        that steered it, and the families it rewrote). A note
        failing anything else — bind language, a premium figure,
        a promise, PII — is refused outright, never stored,
        exactly as before: steering is for wording, refusal is
        for substance. Either way the refusal is recorded on the
        record (``status_note_refusals``: the failed checks, and
        whether a steered candidate was produced and still
        failed), because the metrics aggregate the repairs and
        the refusals as one drafting-quality family — a refusal
        that evaporated into the raised error would be a signal
        nobody could trend.

        Re-drafting replaces the stored note (the case aged, the
        facts may have moved). Raises KeyError for an unknown id
        and ValueError when the submission is not awaiting a
        decision or is still within its SLA budget.
        """
        store = self._get_store()
        record = store.get(submission_id)
        if record is None:
            raise KeyError(f"Unknown submission_id: {submission_id} (analyze it first)")
        item = queue_item_for(record, budgets=self.effective_sla_budgets())
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
        repair: dict = {}
        steered_candidate = False
        if not validation.passed:
            # One steered redraft for a wording-only failure (see
            # the docstring): the steering quotes the offending
            # sentence and rewrites it into the approved shape;
            # the rewrite ships only if it passes the same gate.
            from .guardrails import letter_wording_hits
            from .letter_repair import (
                status_note_repair_guidance,
                steer_status_note,
            )

            guidance = status_note_repair_guidance(validation, text)
            steered = steer_status_note(text, validation) if guidance else None
            if steered is not None:
                steered_candidate = True
                steered_validation = validate_status_note(steered, submission_id)
                if steered_validation.passed:
                    repair = {
                        "repair_attempted": True,
                        "repaired": True,
                        "original_validation": validation.model_dump(
                            mode="json"
                        ),
                        "repair_guidance": guidance,
                        # The families the steering rewrote —
                        # the vocabulary the metrics aggregate
                        # counts repairs in (see
                        # insights._status_note_repair_aggregate).
                        "repair_families": sorted(
                            family
                            for family, hit in letter_wording_hits(
                                text
                            ).items()
                            if hit
                        ),
                    }
                    text, validation = steered, steered_validation
        if not validation.passed:
            # The note is refused and never stored — but the
            # refusal itself is recorded on the record before
            # raising: a refused draft means record prose the
            # composition interpolated carried a forbidden
            # phrasing, and until the refusal was kept it
            # evaporated into the error, invisible to the
            # drafting-quality metrics that watch exactly this
            # signal (see insights._status_note_repair_aggregate).
            from .guardrails import letter_wording_hits

            failed_checks = [
                c.name for c in validation.checks if not c.passed
            ]
            refusals = list(record.status_note_refusals or [])
            refusals.append(
                {
                    "refused_at": now.isoformat(timespec="seconds"),
                    "failed_checks": failed_checks,
                    "steered": steered_candidate,
                    "repair_families": sorted(
                        family
                        for family, hit in letter_wording_hits(
                            text
                        ).items()
                        if hit
                    )
                    if "no_forward_motion_in_letter" in failed_checks
                    else [],
                }
            )
            record.status_note_refusals = refusals
            store.save(record)
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
            **repair,
            "sla": {
                "risk_tier": item["risk_tier"],
                "budget_days": item["sla_budget_days"],
                "age_days": item["age_days"],
                "days_over": item["sla_days_over"],
            },
            # What this case was waiting on when the note was
            # drafted — the snapshot a later resubmission's
            # follow-through is checked against (see
            # insights.status_note_follow_through). The note's prose
            # says it in words; this says it in data, so the check
            # is computed, not re-read from text.
            "outstanding": {
                "missing_elements": list(record.result.facts.missing_elements),
                "gap_sources": list(record.result.facts.gap_sources),
                "blocking_lines": [
                    {
                        "line_of_business": entry["line_of_business"],
                        "missing_items": list(entry["missing_items"]),
                    }
                    for entry in item["blocking_lines"]
                ],
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

    def status_note_refusal(self, submission_id: str) -> dict | None:
        """The refused status-note drafts for one submission, read
        on the case itself.

        Returns None when the submission is unknown (the same
        contract as the breach read). Otherwise the case's
        refusal view: the latest refusal in full (when, which
        checks, steered or outright, the wording families — see
        ``insights.latest_status_note_refusal``), the refusal
        count, and the full refusal history behind it. Until
        this read existed, a refused draft was visible only as
        an aggregate in ``/metrics`` and a raised error at draft
        time: the underwriter who opened the case saw that no
        note existed, not that one was attempted and refused —
        and the refusal is usually about prose on this very
        record, which is what they can fix.
        """
        record = self._get_store().get(submission_id)
        if record is None:
            return None
        return {
            "submission_id": submission_id,
            "refusal_count": len(record.status_note_refusals or []),
            "latest_refusal": latest_status_note_refusal(record),
            "refusals": list(record.status_note_refusals or []),
        }

    def effective_diff_thresholds(self) -> dict:
        """The diff-trigger thresholds in force: the environment
        configuration, overridden per dimension by the settings
        record an accepted calibration proposal wrote
        (``proposals.effective_thresholds``). Every analysis this
        service runs passes these into the pipeline explicitly, so
        an accepted proposal changes the very next renewal's
        triggers — no restart, no redeploy."""
        from .proposals import effective_thresholds

        return effective_thresholds(self._get_store())

    def diff_threshold_settings(self) -> dict:
        """The thresholds in force and their provenance: the values,
        whether they come from the environment or an accepted
        proposal, and the settings record's change history."""
        from .proposals import threshold_settings_view

        return threshold_settings_view(self._get_store())

    def diff_trigger_calibration(self) -> dict:
        """The diff-trigger calibration report: the stored renewals
        replayed at candidate trigger thresholds (read-only
        what-if), alongside what actually fired at analysis time
        and what underwriters decided on those cases
        (``calibration.py``). The replay centres on the thresholds
        in force — the settings record's, when a proposal has been
        accepted — and says which source they came from."""
        from .calibration import trigger_calibration

        report = trigger_calibration(
            self._get_store().all_records(),
            thresholds=self.effective_diff_thresholds(),
        )
        report["threshold_source"] = self.diff_threshold_settings()["source"]
        return report

    def draft_calibration_proposals(self) -> dict:
        """Draft threshold-change proposals from the calibration
        evidence (``proposals.draft_proposals``): candidate moves
        whose gained/lost referrals carry one-sided decision
        evidence become stored proposal documents. Drafting changes
        no threshold and no record — a human still decides."""
        from .proposals import draft_proposals

        store = self._get_store()
        return draft_proposals(store, store.all_records())

    def calibration_proposals(self) -> list[dict]:
        """Every stored calibration proposal, oldest first — each
        annotated with its evidence expiry (``expires_at`` /
        ``expired``), resolved at read time."""
        from .proposals import annotate_proposal, load_proposals

        return [
            annotate_proposal(p) for p in load_proposals(self._get_store())
        ]

    def refresh_calibration_proposal(self, proposal_id: str, actor: str) -> dict:
        """Refresh one expired proposal: re-run its evidence scan
        against the store as it stands (``proposals.refresh_proposal``).
        The proposal is renewed with a fresh snapshot, or withdrawn
        when the evidence no longer carries the move — either way
        the refresh is on the proposal's audit list."""
        from .proposals import annotate_proposal, refresh_proposal

        store = self._get_store()
        proposal = refresh_proposal(
            store, store.all_records(), proposal_id, actor
        )
        return annotate_proposal(proposal)

    def refresh_expired_calibration_proposals(self, actor: str) -> dict:
        """Refresh every expired calibration draft in one call
        (``proposals.refresh_expired_proposals``): each draft's
        evidence scan re-runs individually — renewed with a fresh
        snapshot or withdrawn with its reason — and the report
        lists the two outcomes side by side. Nothing is decided
        by the sweep."""
        from .proposals import refresh_expired_proposals

        store = self._get_store()
        return refresh_expired_proposals(store, store.all_records(), actor)

    def preview_refresh_calibration_proposal(
        self, proposal_id: str
    ) -> dict:
        """Preview one expired draft's refresh
        (``proposals.preview_refresh_proposal``): its evidence
        scan runs and the outcome it would produce is reported —
        renewed with a named fresh snapshot, or withdrawn with
        its reason — while nothing is written."""
        from .proposals import preview_refresh_proposal

        store = self._get_store()
        return preview_refresh_proposal(
            store, store.all_records(), proposal_id
        )

    def preview_refresh_expired_calibration_proposals(self) -> dict:
        """Preview the refresh-all sweep
        (``proposals.preview_refresh_expired_proposals``): every
        expired draft scanned individually and reported as
        would-renew / would-withdraw, without a single write —
        the shape of the drift before anyone commits to it."""
        from .proposals import preview_refresh_expired_proposals

        store = self._get_store()
        return preview_refresh_expired_proposals(
            store, store.all_records()
        )

    def revert_diff_threshold_settings(self, actor: str) -> dict:
        """Revert the threshold settings record's latest accepted
        change (``proposals.revert_threshold_settings``): the
        prior value returns as a new audited history entry naming
        the proposal it undoes — history is never rewritten."""
        from .proposals import revert_threshold_settings

        revert_threshold_settings(self._get_store(), actor)
        return self.diff_threshold_settings()

    def revert_sla_budget_settings(self, actor: str) -> dict:
        """Revert the SLA-budget settings record's latest accepted
        change, the same append-only way."""
        from .proposals import revert_sla_budget_settings

        revert_sla_budget_settings(self._get_store(), actor)
        return self.sla_budget_settings()

    def restore_diff_threshold_settings(
        self, actor: str, history_index: int, note: str = ""
    ) -> dict:
        """Restore the threshold settings record to the state in
        force after its history entry ``history_index``
        (``proposals.restore_threshold_settings``): one new audited
        history entry naming the entry restored to, however many
        dimensions move. The way back from a bad sequence of
        accepts is one decision, not one revert per change. The
        restorer's optional ``note`` is kept on the restore entry
        as its decision note — the "why" beside the "what"."""
        from .proposals import restore_threshold_settings

        restore_threshold_settings(
            self._get_store(), history_index, actor, note=note
        )
        return self.diff_threshold_settings()

    def restore_sla_budget_settings(
        self, actor: str, history_index: int, note: str = ""
    ) -> dict:
        """Restore the SLA-budget settings record to a chosen
        history entry's state, the same append-only way, with the
        same optional decision note on the restore entry."""
        from .proposals import restore_sla_budget_settings

        restore_sla_budget_settings(
            self._get_store(), history_index, actor, note=note
        )
        return self.sla_budget_settings()

    def preview_restore_diff_threshold_settings(
        self, history_index: int, note: str = ""
    ) -> dict:
        """Preview a threshold-settings restore
        (``proposals.preview_restore_threshold_settings``): the
        values in force now, the values as of the chosen history
        entry, and exactly which dimensions would move — a dry
        run that writes nothing and appends nothing. A drafted
        decision ``note`` is echoed back, stored nowhere."""
        from .proposals import preview_restore_threshold_settings

        return preview_restore_threshold_settings(
            self._get_store(), history_index, note=note
        )

    def preview_restore_sla_budget_settings(
        self, history_index: int, note: str = ""
    ) -> dict:
        """Preview an SLA-budget settings restore, the same
        dry-run way, with the same drafted-note echo."""
        from .proposals import preview_restore_sla_budget_settings

        return preview_restore_sla_budget_settings(
            self._get_store(), history_index, note=note
        )

    def decide_calibration_proposal(
        self,
        proposal_id: str,
        actor: str,
        *,
        accept: bool,
        note: str = "",
    ) -> dict:
        """Record a human's decision on a calibration proposal.

        Accepting writes the proposed threshold into the settings
        record the classifier reads (with the change in the
        record's audit history); rejecting leaves every threshold
        exactly where it was. Either way the decision — who, when,
        the note — is kept on the proposal itself."""
        from .proposals import decide_proposal

        return decide_proposal(
            self._get_store(), proposal_id, actor, accept=accept, note=note
        )

    def calibration_proposal_prunes(self) -> list[dict]:
        """Every prune recorded on the proposals document, in the
        order they happened."""
        from .proposals import PROPOSALS_KEY

        document = self._get_store().get_document(PROPOSALS_KEY) or {}
        return list(document.get("prunes") or [])

    def prune_calibration_proposals(self, actor: str, keep: int) -> dict:
        """Prune the proposals document's decided history to the
        newest ``keep`` — the retention discipline the snapshot
        documents and the drafting events already carry, on the
        document that grows with every drafted question. Drafts
        are never pruned (an open question is not history); the
        prune is recorded on the document itself before the
        survivors are written back, beside the proposals list —
        which is why :func:`proposals._save_proposals` preserves
        the document's other keys. The record matters beyond
        tidiness: the drafting scans read the decided history (a
        rejected direction is never re-asked), so a prune thins
        what the scans remember, and the document says when,
        under whom, and what left. A prune that would remove
        nothing writes nothing and records nothing. ``keep``
        below 1 is refused by the shared computation
        (``proposals.prune_proposal_history``)."""
        from .proposals import (
            PROPOSALS_KEY,
            PROPOSAL_HISTORY_STATUSES,
            load_proposals,
            proposal_prune_entry,
            prune_proposal_history,
        )

        store = self._get_store()
        proposals = load_proposals(store)
        kept, pruned = prune_proposal_history(proposals, keep)
        entry = None
        if pruned:
            entry = proposal_prune_entry(
                actor=actor,
                keep=keep,
                pruned=pruned,
                kept=kept,
                pruned_at=datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            )
            document = store.get_document(PROPOSALS_KEY) or {}
            document["proposals"] = kept
            document["prunes"] = list(document.get("prunes") or []) + [
                entry
            ]
            store.save_document(PROPOSALS_KEY, document)
        kept_history = [
            p for p in kept if p.get("status") in PROPOSAL_HISTORY_STATUSES
        ]
        return {
            "kept": len(kept),
            "kept_history": len(kept_history),
            "kept_drafts": len(kept) - len(kept_history),
            "pruned_count": len(pruned),
            "pruned_ids": [p.get("proposal_id", "") for p in pruned],
            "surviving_first": (
                kept_history[0].get("proposal_id", "")
                if kept_history
                else ""
            ),
            "surviving_last": (
                kept_history[-1].get("proposal_id", "")
                if kept_history
                else ""
            ),
            "prune": entry,
        }

    def preview_prune_calibration_proposals(self, keep: int) -> dict:
        """Preview a proposals prune (see
        :meth:`prune_calibration_proposals`): which decided
        proposals a keep-last-N over the decided history would
        remove and what decided range would survive, before
        anything is written — the scans' memory is pruned
        against evidence, not memory."""
        from .proposals import load_proposals, proposal_prune_preview

        return proposal_prune_preview(
            load_proposals(self._get_store()), keep
        )

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
