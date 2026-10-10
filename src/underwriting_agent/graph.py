"""LangGraph pipeline for the underwriting submission intake agent.

    extract → ingest → screen → classify → ┌ retrieve ┐
                                            ├ discrepancies ┤ (fan-out)
                                            └ history ┘
                          → diagnose → options → draft → verify → review
                          → validate → gate → approval_gate → END

The three evidence branches (guideline retrieval, extraction
cross-check discrepancies, insured/broker memory + underwriter
feedback) are independent reads over state the earlier nodes already
produced, so they run as a parallel fan-out; the diagnosis joins them.
The joined state is identical to a sequential run — the fan-out is an
execution detail, not a behaviour change.

Responsibilities are split on purpose:

- **Code computes**: facts (completeness, readiness, loss summary),
  triage signals, option scores, guardrail checks. The model never
  invents these numbers.
- **The model drafts**: document extraction, an independent triage
  (cross-checked against the rules), risk prose, option wording,
  letters, self-verification, independent review — in provider mode.
  In the default mode deterministic templates take those paths.
- **Humans decide**: the graph ends at a checkpointed gate. Nothing
  is quoted, bound, sent, or filed by any node; approval is recorded
  by the service layer and only then may an opt-in webhook route the
  packet to the customer's system of choice.

Resilience: provider-backed nodes run under a per-node timeout with
bounded retries for the steps that are safe to repeat (see
``resilience.py``); drafting and the gate are never retried.

Token budget: with ``RUN_TOKEN_BUDGET`` set, provider steps degrade
to their deterministic paths once cumulative usage crosses the budget
— the run completes on templates instead of failing (see
``_BudgetGuard``).
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .autonomy import recommend_autonomy
from .clarify import build_information_request
from .config import env_int, env_str, load_dotenv
from .crosscheck import resolve_triage
from .diagnosis import build_diagnosis, format_loss_summary, memory_evidence_lines
from .diffing import compute_submission_diff, diff_evidence_lines, diff_packet_lines
from .errors import ProviderError
from .events import NullSink, RunEvent
from .extractor import extract_documents, extraction_discrepancies
from .facts import compute_facts
from .guardrails import validate_drafts
from .model_backends import DraftContext, estimate_cost_usd
from .options import build_options
from .resilience import ResilientBackend
from .retriever import get_retriever
from .reviewer import review_drafts
from .schemas import (
    AgentResult,
    CrossCheck,
    Diagnosis,
    DocumentExtraction,
    Drafts,
    RecoveryOption,
    RetrievedGuideline,
    ReviewResult,
    SubmissionDiff,
    SubmissionFacts,
    SubmissionInput,
    TraceStep,
    Triage,
    Validation,
    VerifyResult,
)
from .screening import sanitized_notes, screen_submission
from .tools_agent import DiagnosisToolBox
from .triage import classify_submission
from .verify import verify_drafts

# Backend method -> the pipeline node it serves. The budget guard
# degrades per node: once the budget is exceeded, that node's provider
# calls stop and its deterministic path runs instead.
_METHOD_NODE = {
    "extract_document_fields": "extract",
    "classify_with_llm": "classify",
    "diagnose": "diagnose",
    "diagnose_with_tools": "diagnose",
    "propose_options": "options",
    "draft_documents": "draft",
    "verify_draft": "verify",
    "review_draft": "review",
    "compose_information_request": "gate",
}

# Nodes whose provider steps may degrade under a token budget. The
# gate node itself (autonomy, the human decision) never degrades —
# only its optional model call (the information-request wording) does.
_PROVIDER_STEP_NODES = {
    "extract", "classify", "diagnose", "options", "draft", "verify", "review", "gate",
}


class _BudgetGuard:
    """Backend proxy enforcing ``RUN_TOKEN_BUDGET`` per run.

    Wraps the run's backend: before each provider call in a degradable
    node it checks the cumulative usage reported so far; once usage
    exceeds the budget, the optional capability methods (extraction,
    triage cross-check, diagnosis, options, verify, review, info-
    request wording) disappear — the nodes read a missing method as
    "use the deterministic path", exactly like the mock backend — and
    ``draft_documents`` degrades to the template backend. The run
    completes; it never fails for budget reasons. ``status()`` reports
    the final accounting for telemetry.
    """

    def __init__(self, backend, template_backend, limit: int | None) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_template", template_backend)
        object.__setattr__(self, "_limit", limit)
        object.__setattr__(self, "_exceeded", False)

    def _over_budget(self) -> bool:
        if self._limit is None:
            return False
        if self._exceeded:
            return True
        usage = self._backend.usage_totals()
        used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        if used > self._limit:
            object.__setattr__(self, "_exceeded", True)
            return True
        return False

    def status(self) -> dict | None:
        if self._limit is None:
            return None
        usage = self._backend.usage_totals()
        used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        return {"used": used, "limit": self._limit, "exceeded": used > self._limit}

    def __getattr__(self, name: str):
        node = _METHOD_NODE.get(name)
        if node is not None and node in _PROVIDER_STEP_NODES and self._over_budget():
            if name == "draft_documents":
                return getattr(self._template, name)
            return None  # capability absent -> deterministic path
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)


def _token_budget() -> int | None:
    load_dotenv()
    value = env_int("RUN_TOKEN_BUDGET", 0)
    return value if value and value > 0 else None


def _wrap_backend(backend):
    """Apply the per-node resilience policy (timeout + safe retries)."""
    return ResilientBackend(backend)


def _guardrail_repair_enabled() -> bool:
    load_dotenv()
    return env_str("GUARDRAIL_REPAIR", "on").strip().lower() not in {"off", "0", "false", "no"}


def _guardrail_repair_max_attempts() -> int:
    load_dotenv()
    return min(max(env_int("GUARDRAIL_REPAIR_MAX_ATTEMPTS", 1), 0), 3)


class AgentState(TypedDict, total=False):
    submission: dict[str, Any]
    extractions: list[dict[str, Any]]
    injection_flags: list[str]
    facts: dict[str, Any]
    triage: dict[str, Any]
    rule_triage: dict[str, Any]
    cross_check: Optional[dict[str, Any]]
    guidelines: list[dict[str, Any]]
    retrieval_info: str
    discrepancies: list[str]
    history_evidence: list[str]
    history: Optional[dict[str, Any]]
    priors: list[dict[str, Any]]
    prior_submission: Optional[dict[str, Any]]
    submission_diff: Optional[dict[str, Any]]
    diagnosis: dict[str, Any]
    options: list[dict[str, Any]]
    options_notes: str
    drafts: dict[str, Any]
    verification: dict[str, Any]
    review: dict[str, Any]
    reviewer_blocked: bool
    validation: dict[str, Any]
    repair_attempted: bool
    repaired: bool
    repair_attempts: int
    original_validation: Optional[dict[str, Any]]
    autonomy: dict[str, Any]
    needs_information: bool
    information_request: Optional[dict[str, Any]]
    token_budget: Optional[dict[str, Any]]
    gate_status: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _screen_text(sub: SubmissionInput) -> str:
    """The broker's text as prompts may see it: flagged sentences removed."""
    return sanitized_notes(sub)


def _classify_context(sub: SubmissionInput, facts: SubmissionFacts) -> DraftContext:
    return DraftContext(
        submission_id=sub.submission_id,
        insured_name=sub.insured_name,
        broker_name=sub.broker_name,
        line_of_business=sub.line_of_business,
        class_of_business=sub.class_of_business,
        completeness=facts.completeness_score,
        readiness=facts.readiness,
        missing=facts.missing_elements,
        loss_summary=format_loss_summary(facts.loss_summary.model_dump()),
        discrepancies=facts.discrepancies,
        broker_text=_screen_text(sub),
        requested_limit=sub.requested_limit,
        requested_deductible=sub.requested_deductible,
        revenue=sub.revenue,
        years_in_business=sub.years_in_business,
        guideline_details=[],
    )


def _draft_context(
    state: AgentState, sub: SubmissionInput, *, repair_feedback: str | None = None
) -> DraftContext:
    triage = Triage.model_validate(state["triage"])
    facts = SubmissionFacts.model_validate(state["facts"])
    guidelines = [RetrievedGuideline.model_validate(g) for g in state.get("guidelines", [])]
    options = [RecoveryOption.model_validate(o) for o in state.get("options", [])]
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    recommended = next((o for o in options if o.recommended), None)
    context = DraftContext(
        submission_id=sub.submission_id,
        insured_name=sub.insured_name,
        broker_name=sub.broker_name,
        line_of_business=sub.line_of_business,
        class_of_business=sub.class_of_business,
        outcome=triage.outcome,
        risk_tier=triage.risk_tier,
        confidence=triage.confidence,
        rationale=triage.rationale,
        referral_triggers=triage.referral_triggers,
        completeness=facts.completeness_score,
        readiness=facts.readiness,
        missing=facts.missing_elements,
        loss_summary=format_loss_summary(facts.loss_summary.model_dump()),
        discrepancies=facts.discrepancies,
        broker_text=_screen_text(sub),
        diagnosis_summary=diagnosis.summary,
        diagnosis_risk_factors=diagnosis.risk_factors,
        guideline_details=[g.model_dump() for g in guidelines],
        citations=[g.guideline_id for g in guidelines],
        submission_changes=(
            diff_packet_lines(SubmissionDiff.model_validate(state["submission_diff"]))
            if state.get("submission_diff")
            else []
        ),
        options=[o.model_dump() for o in options],
        recommended_option_kind=recommended.kind if recommended else "",
        recommended_option_text=(
            f"{recommended.title} — {recommended.description}" if recommended else ""
        ),
    )
    if repair_feedback:
        context["repair_feedback"] = repair_feedback
    return context


def _failed_checks_text(validation: Validation) -> str:
    return "\n".join(
        f"- {check.name}: {check.detail}" for check in validation.checks if not check.passed
    )


def _build_graph(backend, retriever, sink, checkpointer=None):
    """Compile the pipeline graph around one run's collaborators."""

    def node(name: str, title: str):
        """Wrap a step function as a graph node that times itself,
        emits its run event, and accumulates ``duration_ms``."""

        def wrap(fn):
            def run(state: AgentState) -> dict[str, Any]:
                started = time.perf_counter()
                update = fn(state)
                duration_ms = round((time.perf_counter() - started) * 1000, 1)
                merged = dict(state.get("_durations") or {})
                merged[name] = duration_ms
                sink.emit(
                    RunEvent(
                        type="node_completed",
                        submission_id=state["submission"]["submission_id"],
                        node=name,
                        duration_ms=duration_ms,
                    )
                )
                return {**update, "_durations": merged}

            run.__name__ = f"node_{name}"
            return run

        return wrap

    @node("extract", "Document extraction + cross-check")
    def _extract(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        extractions = extract_documents(sub, backend)
        dumps = [e.model_dump() for e in extractions]
        return {"extractions": dumps}

    @node("ingest", "Computed facts (completeness, readiness, loss summary)")
    def _ingest(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        extractions = [DocumentExtraction.model_validate(e) for e in state["extractions"]]
        facts = compute_facts(sub, extractions)
        return {"facts": facts.model_dump()}

    @node("screen", "Injection screening of broker text")
    def _screen(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        return {"injection_flags": screen_submission(sub)}

    @node("classify", "Triage: rules + LLM cross-check")
    def _classify(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        facts = SubmissionFacts.model_validate(state["facts"])
        rule = classify_submission(sub, facts)
        update: dict[str, Any] = {"rule_triage": rule.model_dump()}
        classify_fn = getattr(backend, "classify_with_llm", None)
        if classify_fn is None:
            update["triage"] = rule.model_dump()
            update["cross_check"] = None
            return update
        try:
            llm = classify_fn(_classify_context(sub, facts))
        except Exception as exc:
            # A provider failure in the cross-check degrades to
            # rules-only — the exception travels in state, not up.
            final, cross = resolve_triage(rule, None)
            cross.note = f"LLM cross-check failed ({exc}) — rules only."
            update["triage"] = final.model_dump()
            update["cross_check"] = cross.model_dump()
            return update
        final, cross = resolve_triage(rule, llm, getattr(backend, "name", ""))
        update["triage"] = final.model_dump()
        update["cross_check"] = cross.model_dump()
        return update

    @node("retrieve", "Guideline retrieval")
    def _retrieve(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        triage = Triage.model_validate(state["triage"])
        facts = SubmissionFacts.model_validate(state["facts"])
        query = " ".join(
            part
            for part in [
                sub.line_of_business,
                sub.class_of_business,
                triage.outcome.replace("_", " "),
                triage.risk_tier,
                " ".join(triage.referral_triggers),
                " ".join(facts.missing_elements),
                "underwriting guideline appetite referral authority",
            ]
            if part
        )
        guidelines = retriever.retrieve(query, top_k=4)
        return {
            "guidelines": [g.model_dump() for g in guidelines],
            "retrieval_info": getattr(retriever, "vector_store", "") or "",
        }

    @node("discrepancies", "Extraction cross-check discrepancies")
    def _discrepancies(state: AgentState) -> dict[str, Any]:
        extractions = [DocumentExtraction.model_validate(e) for e in state["extractions"]]
        return {"discrepancies": extraction_discrepancies(extractions)}

    @node("history", "Insured/broker memory + underwriter feedback")
    def _history(state: AgentState) -> dict[str, Any]:
        update: dict[str, Any] = {
            "history_evidence": memory_evidence_lines(state.get("history"))
        }
        # Renewal diffing rides this branch: when the service found a
        # prior submission for this insured, diff it against the new
        # one in code — the deltas join the diagnosis evidence and
        # the packet, and land on the result.
        prior = state.get("prior_submission")
        if prior:
            sub = SubmissionInput.model_validate(state["submission"])
            facts = SubmissionFacts.model_validate(state["facts"])
            diff = compute_submission_diff(sub, facts, prior)
            if diff is not None:
                update["submission_diff"] = diff.model_dump()
        return update

    @node("diagnose", "Risk diagnosis (bounded tool loop in provider mode)")
    def _diagnose(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        history_lines = list(state.get("history_evidence", []))
        if state.get("submission_diff"):
            history_lines += diff_evidence_lines(
                SubmissionDiff.model_validate(state["submission_diff"])
            )
        toolbox = DiagnosisToolBox(
            submission=sub,
            triage=state["triage"],
            facts=state["facts"],
            discrepancies=state.get("discrepancies", []),
            retriever=retriever,
            priors=state.get("priors") or [],
        )
        diagnosis = build_diagnosis(
            submission=sub,
            triage=state["triage"],
            facts=state["facts"],
            extractions=[DocumentExtraction.model_validate(e) for e in state["extractions"]],
            guidelines=state.get("guidelines", []),
            backend=backend,
            history=state.get("history"),
            toolbox=toolbox,
            discrepancies=state.get("discrepancies", []),
            history_lines=history_lines,
        )
        return {"diagnosis": diagnosis.model_dump()}

    @node("options", "Next-step options, scored in code")
    def _options(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        options, notes = build_options(
            submission=sub,
            triage=state["triage"],
            facts=state["facts"],
            diagnosis_summary=state["diagnosis"]["summary"],
            backend=backend,
        )
        update: dict[str, Any] = {"options": [o.model_dump() for o in options]}
        if notes:
            update["options_notes"] = notes
        return update

    @node("draft", "Draft broker letter + underwriter packet")
    def _draft(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        subject, letter, packet = backend.draft_documents(_draft_context(state, sub))
        return {
            "drafts": Drafts(
                subject=subject, broker_letter=letter, underwriter_packet=packet
            ).model_dump()
        }

    @node("verify", "Self-verification of the drafts")
    def _verify(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        drafts = Drafts.model_validate(state["drafts"])
        retrieved_ids = [g["guideline_id"] for g in state.get("guidelines", [])]
        result = verify_drafts(
            submission=sub,
            triage=state["triage"],
            facts=state["facts"],
            drafts=drafts,
            retrieved_ids=retrieved_ids,
            backend=backend,
        )
        return {"verification": result.model_dump()}

    @node("review", "Independent review of the drafts")
    def _review(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        drafts = Drafts.model_validate(state["drafts"])
        retrieved_ids = [g["guideline_id"] for g in state.get("guidelines", [])]
        result = review_drafts(
            submission=sub,
            triage=state["triage"],
            facts=state["facts"],
            diagnosis=Diagnosis.model_validate(state["diagnosis"]),
            options=[RecoveryOption.model_validate(o) for o in state.get("options", [])],
            drafts=drafts,
            retrieved_ids=retrieved_ids,
            backend=backend,
        )
        return {
            "review": result.model_dump(),
            "reviewer_blocked": bool(result.enabled and result.verdict == "block"),
        }

    @node("validate", "Guardrail checks (+ bounded repair)")
    def _validate(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        citations = [g["guideline_id"] for g in state.get("guidelines", [])]
        drafts = Drafts.model_validate(state["drafts"])
        validation = validate_drafts(sub, drafts, citations)
        update: dict[str, Any] = {
            "validation": validation.model_dump(),
            "repair_attempted": False,
            "repaired": False,
            "repair_attempts": 0,
            "original_validation": None,
        }
        if validation.passed or not _guardrail_repair_enabled():
            return update
        max_attempts = _guardrail_repair_max_attempts()
        original = validation
        attempts = 0
        while attempts < max_attempts and not validation.passed:
            attempts += 1
            feedback = _failed_checks_text(validation)
            try:
                subject, letter, packet = backend.draft_documents(
                    _draft_context(state, sub, repair_feedback=feedback)
                )
            except Exception:
                break  # a failed redraft keeps the original drafts + failures
            candidate = Drafts(subject=subject, broker_letter=letter, underwriter_packet=packet)
            candidate_validation = validate_drafts(sub, candidate, citations)
            if candidate_validation.passed or _failed_count(candidate_validation) < _failed_count(validation):
                drafts, validation = candidate, candidate_validation
        update.update(
            {
                "drafts": drafts.model_dump(),
                "validation": validation.model_dump(),
                "repair_attempted": attempts > 0,
                "repaired": validation.passed,
                "repair_attempts": attempts,
                "original_validation": original.model_dump(),
            }
        )
        return update

    @node("gate", "Autonomy recommendation + information request")
    def _gate(state: AgentState) -> dict[str, Any]:
        sub = SubmissionInput.model_validate(state["submission"])
        triage = Triage.model_validate(state["triage"])
        facts = SubmissionFacts.model_validate(state["facts"])
        validation = Validation.model_validate(state["validation"])
        cross = (
            CrossCheck.model_validate(state["cross_check"])
            if state.get("cross_check")
            else None
        )
        review = (
            ReviewResult.model_validate(state["review"]) if state.get("review") else None
        )
        autonomy = recommend_autonomy(
            triage=triage,
            facts=facts,
            validation=validation,
            repair_attempted=bool(state.get("repair_attempted")),
            cross_check=cross,
            review=review,
            reviewer_blocked=bool(state.get("reviewer_blocked")),
        )
        info = build_information_request(
            submission=sub,
            triage_outcome=triage.outcome,
            facts=facts,
            extractions=[DocumentExtraction.model_validate(e) for e in state["extractions"]],
            backend=backend,
        )
        return {
            "autonomy": autonomy.model_dump(),
            "needs_information": bool(info and info.needed),
            "information_request": info.model_dump() if info else None,
            "gate_status": "awaiting_underwriter",
        }

    def _approval_gate(state: AgentState) -> dict[str, Any]:
        """The human gate: pause the thread until a decision resumes it.

        With a checkpointer the run pauses here (``interrupt``) and the
        service's approve/reject resumes the same thread with the
        decision payload. Without one this node is a pass-through and
        the decision lives only in the store — same outcome, less
        ceremony. Either way no node after this one exists: END.
        """
        if checkpointer is None:
            return {"gate_status": state.get("gate_status", "awaiting_underwriter")}
        triage = state.get("triage") or {}
        validation = state.get("validation") or {}
        decision = interrupt(
            {
                "submission_id": state["submission"]["submission_id"],
                "status": "awaiting_underwriter",
                "outcome": triage.get("outcome"),
                "risk_tier": triage.get("risk_tier"),
                "validation_passed": validation.get("passed"),
                "external_action_taken": False,
            }
        )
        if isinstance(decision, dict) and decision.get("decision") in (
            "approved",
            "rejected",
        ):
            return {"gate_status": decision["decision"]}
        return {"gate_status": "awaiting_underwriter"}

    builder = StateGraph(AgentState)
    for name, fn in [
        ("extract", _extract),
        ("ingest", _ingest),
        ("screen", _screen),
        ("classify", _classify),
        ("retrieve", _retrieve),
        ("discrepancies", _discrepancies),
        ("history", _history),
        ("diagnose", _diagnose),
        ("options", _options),
        ("draft", _draft),
        ("verify", _verify),
        ("review", _review),
        ("validate", _validate),
        ("gate", _gate),
        ("approval_gate", _approval_gate),
    ]:
        builder.add_node(name, fn)
    builder.set_entry_point("extract")
    builder.add_edge("extract", "ingest")
    builder.add_edge("ingest", "screen")
    builder.add_edge("screen", "classify")
    # Evidence fan-out: three independent branches off classify,
    # joined by diagnose.
    builder.add_edge("classify", "retrieve")
    builder.add_edge("classify", "discrepancies")
    builder.add_edge("classify", "history")
    builder.add_edge("retrieve", "diagnose")
    builder.add_edge("discrepancies", "diagnose")
    builder.add_edge("history", "diagnose")
    builder.add_edge("diagnose", "options")
    builder.add_edge("options", "draft")
    builder.add_edge("draft", "verify")
    builder.add_edge("verify", "review")
    builder.add_edge("review", "validate")
    builder.add_edge("validate", "gate")
    builder.add_edge("gate", "approval_gate")
    builder.add_edge("approval_gate", END)
    return builder.compile(checkpointer=checkpointer)


def _failed_count(validation: Validation) -> int:
    return sum(1 for check in validation.checks if not check.passed)


def _durations(state: AgentState) -> dict[str, float]:
    return dict(state.get("_durations") or {})


def _build_trace(state: AgentState) -> list[TraceStep]:
    """Assemble the human-readable step trace from the final state."""
    d = _durations(state)
    sub = SubmissionInput.model_validate(state["submission"])
    triage = Triage.model_validate(state["triage"])
    rule = Triage.model_validate(state["rule_triage"]) if state.get("rule_triage") else triage
    facts = SubmissionFacts.model_validate(state["facts"])
    diagnosis = Diagnosis.model_validate(state["diagnosis"])
    validation = Validation.model_validate(state["validation"])
    guidelines = [RetrievedGuideline.model_validate(g) for g in state.get("guidelines", [])]
    options = [RecoveryOption.model_validate(o) for o in state.get("options", [])]
    verification = (
        VerifyResult.model_validate(state["verification"])
        if state.get("verification")
        else None
    )
    review = ReviewResult.model_validate(state["review"]) if state.get("review") else None

    def step(name: str, title: str, details: list[str]) -> TraceStep:
        return TraceStep(name=name, title=title, details=details, duration_ms=d.get(name, 0.0))

    extraction_details = [
        f"{e['document_id'] or e['doc_type']}: {len(e['fields'])} field(s) "
        f"({e['source']})"
        + (
            " — "
            + ", ".join(
                f"{f['field']}={f['status']}" for f in e["fields"] if f["status"] != "match"
            )
            if any(f["status"] != "match" for f in e["fields"])
            else ""
        )
        for e in state.get("extractions", [])
    ] or ["no documents on the submission"]
    if state.get("discrepancies"):
        extraction_details.append(
            f"{len(state['discrepancies'])} cross-check discrepanc"
            f"{'y' if len(state['discrepancies']) == 1 else 'ies'} recorded"
        )

    classify_details = [
        f"rules: {rule.outcome} / {rule.risk_tier} (confidence {rule.confidence})"
    ]
    cross = state.get("cross_check")
    if cross:
        classify_details.append(f"cross-check: {cross['resolution']} — {cross['note']}")
        if cross.get("llm_outcome"):
            classify_details.append(
                f"LLM: {cross['llm_outcome']} / {cross['llm_tier']} "
                f"(confidence {cross['llm_confidence']})"
            )
    classify_details.append(
        f"final: {triage.outcome} / {triage.risk_tier} (confidence {triage.confidence})"
    )
    classify_details.extend(f"trigger: {t}" for t in triage.referral_triggers)

    steps = [
        step("extract", "Document extraction + cross-check", extraction_details),
        step(
            "ingest",
            "Computed facts",
            [
                f"completeness {facts.completeness_score} · readiness {facts.readiness}",
                f"loss history: {format_loss_summary(facts.loss_summary.model_dump())}",
                *(
                    [f"missing: {', '.join(facts.missing_elements)}"]
                    if facts.missing_elements
                    else []
                ),
                *(
                    [f"warning: {facts.document_pair_warning}"]
                    if facts.document_pair_warning
                    else []
                ),
                f"submission age: {facts.submission_age_days} day(s)",
            ],
        ),
        step(
            "screen",
            "Injection screening",
            state.get("injection_flags")
            or ["no injection patterns found in broker text"],
        ),
        step("classify", "Triage (rules + LLM cross-check)", classify_details),
        step(
            "retrieve",
            "Guideline retrieval",
            [
                f"[{g.guideline_id}] {g.title} (score {g.score}, via "
                f"{'+'.join(g.sources) or 'keyword'})"
                for g in guidelines
            ]
            or ["no guidelines retrieved"],
        ),
        step(
            "diagnose",
            "Risk diagnosis",
            [
                f"{diagnosis.summary} (source: {diagnosis.source})",
                *( [f"note: {diagnosis.note}"] if diagnosis.note else [] ),
                *[f"tool: {tc.get('name')} — {tc.get('summary')}" for tc in diagnosis.tool_calls],
                *diagnosis.evidence,
            ],
        ),
        step(
            "options",
            "Next-step options (scored in code)",
            [
                f"[{o.kind}] {o.title} — score {o.score}"
                + (" ← recommended" if o.recommended else "")
                for o in options
            ]
            + ([f"note: {state['options_notes']}"] if state.get("options_notes") else []),
        ),
        step(
            "draft",
            "Drafts composed",
            [
                "broker letter + underwriter packet drafted — nothing sent, "
                "nothing bound, no premium quoted"
            ],
        ),
    ]
    if verification is not None:
        steps.append(
            step(
                "verify",
                "Self-verification",
                [
                    f"grounded: {verification.grounded} (source: {verification.source})",
                    *[f"issue: {i}" for i in verification.issues],
                    *([f"note: {verification.note}"] if verification.note else []),
                ],
            )
        )
    if review is not None:
        steps.append(
            step(
                "review",
                "Independent review",
                [
                    f"verdict: {review.verdict} (source: {review.source}, "
                    f"enabled: {review.enabled})",
                    *[f"finding: {f}" for f in review.findings],
                    *(
                        ["⛔ reviewer blocked — flagged for the underwriter"]
                        if state.get("reviewer_blocked")
                        else []
                    ),
                ],
            )
        )
    validate_details = [
        f"[{'PASS' if c.passed else 'FAIL'}] {c.name} — {c.detail}"
        for c in validation.checks
    ]
    if state.get("repair_attempted"):
        validate_details.append(
            f"repair: {state.get('repair_attempts')} attempt(s) — "
            f"{'repaired' if state.get('repaired') else 'still failing'}"
        )
    steps.append(step("validate", "Guardrail checks", validate_details))
    autonomy = state.get("autonomy") or {}
    gate_details = [
        f"autonomy recommendation: "
        f"{'eligible for auto-triage' if autonomy.get('eligible_for_auto_triage') else 'human decision required'}"
        " (recommendation only — never acted on)",
        *autonomy.get("reasons", []),
    ]
    if state.get("needs_information"):
        gate_details.append(
            "information request composed for the broker (not sent) — missing: "
            + "; ".join((state.get("information_request") or {}).get("missing_items", []))
        )
    steps.append(step("gate", "Underwriter gate", gate_details))
    steps.append(
        step(
            "approval_gate",
            "Human decision",
            [
                f"status: {state.get('gate_status', 'awaiting_underwriter')} · "
                "external action taken: False (nothing is quoted, bound, sent, "
                "or filed by the agent)"
            ],
        )
    )
    return steps


def _run_telemetry(backend, started: float, budget_status: dict | None) -> dict:
    """Assemble the run's telemetry block from the backend's accounting.

    The mock backend reports zero tokens for its render calls — those
    are template renders, not model usage — so mock runs record token
    counts as None ("no model ran") rather than dressing them up.
    """
    usage = backend.usage_totals()
    is_mock = getattr(backend, "name", "") == "mock"
    input_tokens = None if is_mock else int(usage.get("input_tokens", 0))
    output_tokens = None if is_mock else int(usage.get("output_tokens", 0))
    model = getattr(backend, "_model", "") or ""
    return {
        "backend": getattr(backend, "name", "unknown"),
        "model": model,
        "model_calls": int(usage.get("calls", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimate_cost_usd(model, input_tokens, output_tokens),
        "latency_seconds": round(time.perf_counter() - started, 3),
        "budget": budget_status,
    }


def run_submission(
    submission: SubmissionInput,
    backend=None,
    retriever=None,
    history: dict | None = None,
    priors: list[dict] | None = None,
    prior_submission: dict | None = None,
    event_sink=None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> AgentResult:
    """Run one submission through the pipeline to the underwriter gate.

    ``backend``/``retriever`` default to the environment-configured
    ones; ``history``/``priors`` carry the store's memory for this
    insured/broker (the service layer supplies them);
    ``prior_submission`` is the insured's most recent prior stored
    submission (payload + computed facts) for renewal diffing;
    ``event_sink`` receives structured run events; ``checkpointer``
    + ``thread_id`` pause the run at the gate for a later resume.
    """
    from .model_backends import MockModelBackend, get_backend

    started = time.perf_counter()
    sink = event_sink or NullSink()
    raw_backend = backend or get_backend()
    if hasattr(raw_backend, "reset_usage"):
        raw_backend.reset_usage()
    budget = _BudgetGuard(raw_backend, MockModelBackend(), _token_budget())
    wrapped = _wrap_backend(budget)
    retriever = retriever or get_retriever()
    graph = _build_graph(wrapped, retriever, sink, checkpointer=checkpointer)
    initial: AgentState = {
        "submission": submission.model_dump(),
        "history": history,
        "priors": priors or [],
        "prior_submission": prior_submission,
    }
    sink.emit(
        RunEvent(
            type="run_started",
            submission_id=submission.submission_id,
            detail={"insured_name": submission.insured_name},
        )
    )
    config = (
        {"configurable": {"thread_id": thread_id or submission.submission_id}}
        if checkpointer is not None
        else None
    )
    try:
        if config is not None:
            state = graph.invoke(initial, config)
            interrupts = state.get("__interrupt__") or []
            if interrupts:
                # Paused at the gate: read the working state back from
                # the checkpoint — it is the state the decision will
                # resume, and the store records it as awaiting.
                snapshot = graph.get_state(config)
                state = dict(snapshot.values)
        else:
            state = graph.invoke(initial)
    except ProviderError:
        raise
    except Exception as exc:
        sink.emit(
            RunEvent(
                type="run_failed",
                submission_id=submission.submission_id,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
        )
        raise
    result = AgentResult(
        submission_id=submission.submission_id,
        status="awaiting_underwriter",
        insured_name=submission.insured_name,
        line_of_business=submission.line_of_business,
        submission_diff=(
            SubmissionDiff.model_validate(state["submission_diff"])
            if state.get("submission_diff")
            else None
        ),
        extractions=[DocumentExtraction.model_validate(e) for e in state.get("extractions", [])],
        facts=SubmissionFacts.model_validate(state["facts"]),
        injection_flags=state.get("injection_flags", []),
        triage=Triage.model_validate(state["triage"]),
        cross_check=(
            CrossCheck.model_validate(state["cross_check"])
            if state.get("cross_check")
            else None
        ),
        guidelines=[RetrievedGuideline.model_validate(g) for g in state.get("guidelines", [])],
        vector_store=state.get("retrieval_info", ""),
        diagnosis=Diagnosis.model_validate(state["diagnosis"]),
        options=[RecoveryOption.model_validate(o) for o in state.get("options", [])],
        drafts=Drafts.model_validate(state["drafts"]) if state.get("drafts") else None,
        verification=(
            VerifyResult.model_validate(state["verification"])
            if state.get("verification")
            else None
        ),
        review=ReviewResult.model_validate(state["review"]) if state.get("review") else None,
        reviewer_blocked=bool(state.get("reviewer_blocked")),
        validation=(
            Validation.model_validate(state["validation"])
            if state.get("validation")
            else None
        ),
        repair_attempted=bool(state.get("repair_attempted")),
        repaired=bool(state.get("repaired")),
        repair_attempts=int(state.get("repair_attempts", 0)),
        original_validation=(
            Validation.model_validate(state["original_validation"])
            if state.get("original_validation")
            else None
        ),
        information_request=None,
        needs_information=bool(state.get("needs_information")),
        telemetry=None,
        trace=_build_trace(state),
    )
    if state.get("information_request"):
        from .schemas import InformationRequest

        result.information_request = InformationRequest.model_validate(
            state["information_request"]
        )
    from .schemas import AutonomyRecommendation, RunTelemetry

    if state.get("autonomy"):
        result.autonomy = AutonomyRecommendation.model_validate(state["autonomy"])
    result.telemetry = RunTelemetry.model_validate(
        _run_telemetry(raw_backend, started, budget.status())
    )
    sink.emit(
        RunEvent(
            type="run_completed",
            submission_id=submission.submission_id,
            detail={"status": result.status},
        )
    )
    return result


def resume_approval(
    submission_id: str,
    decision: dict,
    checkpointer: BaseCheckpointSaver,
    backend=None,
    retriever=None,
) -> bool:
    """Resume a gate-paused thread with the underwriter's decision.

    Returns True when a paused thread was found and completed, False
    when there is nothing to resume (checkpoints off during analysis,
    thread already finished). The decision of record lives in the
    store; this only closes out the graph thread.
    """
    from .model_backends import get_backend

    sink = NullSink()
    graph = _build_graph(
        _wrap_backend(backend or get_backend()),
        retriever or get_retriever(),
        sink,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": submission_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.next:
        return False
    from langgraph.types import Command

    graph.invoke(Command(resume=decision), config)
    return True
