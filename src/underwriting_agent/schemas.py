"""Typed schemas for the underwriting submission intake agent.

Every boundary in the pipeline is a Pydantic model: the submission that
arrives, the facts computed in code, the triage decision, the drafts, and
the final result that waits at the underwriter's gate. The model never
invents the numbers on these objects — computed fields are computed in
code (see ``facts.py``); the model only does language work over them.

Intake is messy in the real world, so normalisation happens here, at the
boundary: document-type aliases are folded to a small vocabulary and
numeric field values are coerced to strings instead of rejected.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Intake normalisation
# ---------------------------------------------------------------------------

DOC_TYPE_ALIASES = {
    "acord_form": "acord_form",
    "acord": "acord_form",
    "submission_form": "acord_form",
    "application": "acord_form",
    "application_form": "acord_form",
    "loss_run": "loss_run",
    "loss_runs": "loss_run",
    "loss_history": "loss_run",
    "loss history": "loss_run",
    "claims_history": "loss_run",
    "financial_statement": "financials",
    "financials": "financials",
    "financial_statement(s)": "financials",
    "financial statements": "financials",
    "statement_of_values": "sov",
    "sov": "sov",
    "statement of values": "sov",
    "schedule_of_values": "sov",
    "broker_email": "broker_email",
    "email": "broker_email",
    "broker email": "broker_email",
    "broker_letter": "broker_email",
    "supplemental": "supplemental",
    "supplemental_application": "supplemental",
    "risk_control_report": "supplemental",
}


def normalize_doc_type(raw: str) -> str:
    key = " ".join(str(raw or "").strip().lower().replace("_", " ").split())
    key_us = key.replace(" ", "_")
    return DOC_TYPE_ALIASES.get(key) or DOC_TYPE_ALIASES.get(key_us) or key_us


def _coerce_field_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


class DocumentInput(BaseModel):
    doc_type: str
    document_id: str = ""
    raw_text: str = ""
    # Object-storage reference: when the document already lives in
    # the bucket (the production intake shape), the adapter submits
    # the key and the service fetches the text through the
    # ObjectStore port; after analysis the service archives inline
    # documents and records their keys here on the stored payload.
    object_key: str = ""
    fields: dict[str, str] = Field(default_factory=dict)

    doc_type_provided: str = ""
    doc_type_flagged: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            raw = str(data.get("doc_type", ""))
            norm = normalize_doc_type(raw)
            data["doc_type_provided"] = raw
            data["doc_type"] = norm
            data["doc_type_flagged"] = norm not in {
                "acord_form", "loss_run", "financials", "sov", "broker_email", "supplemental",
            }
            fields = data.get("fields") or {}
            if isinstance(fields, dict):
                data["fields"] = {str(k): _coerce_field_value(v) for k, v in fields.items()}
        return data


class LossRecord(BaseModel):
    date: str = ""  # ISO date or year
    amount_incurred: float = 0.0
    description: str = ""
    status: str = "closed"
    # The carrier's claim identifier, when the loss run carries one.
    # Loss development matching (diffing.py) keys on it first and
    # falls back to date + description — a claim keeps its identity
    # across loss runs even when its wording is edited.
    claim_id: str = ""

    @field_validator("amount_incurred", mode="before")
    @classmethod
    def _num(cls, v: Any) -> float:
        if isinstance(v, str):
            v = v.replace(",", "").replace("$", "").strip()
        return float(v or 0)


def _coerce_optional_number(v: Any) -> Any:
    if isinstance(v, str):
        v = v.replace(",", "").replace("$", "").strip()
        if v == "":
            return None
        return float(v)
    return v


class LineInput(BaseModel):
    """One line of business inside a multi-line (bundled) submission.

    Real broker submissions bundle lines — property + cyber on one
    account. The bundle carries the shared context (insured, broker,
    email, notes, submitted date); each line carries what is
    line-specific: its own limit/deductible asks, its own documents,
    its own loss run. Fields left empty fall back to the bundle-level
    values when the service splits the bundle (see ``bundles.py``).
    """

    line_of_business: str
    class_of_business: str = ""
    requested_limit: str = ""
    requested_deductible: str = ""
    effective_date_requested: str = ""
    revenue: Optional[float] = None
    years_in_business: Optional[float] = None
    prior_carrier: str = ""
    documents: list[DocumentInput] = Field(default_factory=list)
    losses: list[LossRecord] = Field(default_factory=list)

    @field_validator("revenue", "years_in_business", mode="before")
    @classmethod
    def _num_opt(cls, v: Any) -> Any:
        return _coerce_optional_number(v)


class SubmissionInput(BaseModel):
    submission_id: str
    insured_name: str = ""
    broker_name: str = ""
    broker_email_text: str = ""
    notes: str = ""
    line_of_business: str = ""
    class_of_business: str = ""
    submitted_at: str = ""  # ISO datetime/date
    effective_date_requested: str = ""
    revenue: Optional[float] = None
    years_in_business: Optional[float] = None
    prior_carrier: str = ""
    requested_limit: str = ""
    requested_deductible: str = ""
    documents: list[DocumentInput] = Field(default_factory=list)
    losses: list[LossRecord] = Field(default_factory=list)
    # Multi-line bundles: when non-empty, this submission is a
    # bundle and ``lines`` is the intake of record — the service
    # splits it into one sub-analysis per line (sharing the insured
    # context above) and rolls the lines into one combined packet
    # with one underwriter gate. The bundle-level line/limit/loss
    # fields then act as fallbacks for lines that leave them empty.
    lines: list[LineInput] = Field(default_factory=list)

    @field_validator("revenue", "years_in_business", mode="before")
    @classmethod
    def _num_opt(cls, v: Any) -> Any:
        return _coerce_optional_number(v)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACTED_FIELDS = [
    "insured_name", "line_of_business", "requested_limit", "requested_deductible",
    "effective_date", "revenue", "years_in_business", "prior_carrier",
    "loss_history_present",
]


class FieldExtraction(BaseModel):
    field: str
    value: Optional[str] = None
    confidence: float = 0.0
    status: str = "source_provided"  # match|mismatch|missing_in_text|extracted_only|source_provided


class DocumentExtraction(BaseModel):
    document_id: str = ""
    doc_type: str = ""
    source: str = "provided"  # provided | llm
    note: str = ""
    fields: list[FieldExtraction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Computed facts (code only — never the model)
# ---------------------------------------------------------------------------

class LossSummary(BaseModel):
    claim_count: int = 0
    total_incurred: float = 0.0
    largest_claim: float = 0.0
    loss_years: list[int] = Field(default_factory=list)
    frequency_per_year: float = 0.0


class SubmissionFacts(BaseModel):
    completeness_score: float = 0.0  # 0..1 over required elements for the line
    missing_elements: list[str] = Field(default_factory=list)
    readiness: str = "not_ready"  # ready | ready_with_gaps | not_ready
    loss_summary: LossSummary = Field(default_factory=LossSummary)
    submission_age_days: int = 0
    discrepancy_count: int = 0
    discrepancies: list[str] = Field(default_factory=list)
    # Where the readiness gaps come from, itemised in code: one entry
    # per missing required element and per extraction cross-check
    # discrepancy. Empty exactly when readiness is "ready". This is
    # what lets the packet/reviewer say "nothing is missing — the
    # gaps are discrepancies" instead of implying missing documents.
    gap_sources: list[str] = Field(default_factory=list)
    document_pair_warning: str = ""


# ---------------------------------------------------------------------------
# Renewal / resubmission diff (computed in code — see diffing.py)
# ---------------------------------------------------------------------------

class DiffFieldChange(BaseModel):
    field: str  # requested_limit | revenue | loss_history | ...
    label: str = ""
    previous: str = ""
    current: str = ""
    note: str = ""  # direction / magnitude, e.g. "increased" or "+29.2%"


class LossClaimRef(BaseModel):
    """One claim as it appeared on one side of a loss-run diff."""

    date: str = ""
    description: str = ""
    amount_incurred: float = 0.0
    status: str = ""


class ClaimDevelopment(BaseModel):
    """One claim matched across two loss runs, and how it moved."""

    date: str = ""
    description: str = ""
    prior_incurred: float = 0.0
    current_incurred: float = 0.0
    delta_incurred: float = 0.0
    prior_status: str = ""
    current_status: str = ""


class LossDevelopment(BaseModel):
    """Per-claim development between the prior loss run and this one.

    The aggregate loss-history change says the totals moved; this
    says *which claims* moved — the ones that appeared on the run,
    the ones that are no longer listed, and the matched claims whose
    incurred (or status) changed while the underwriter wasn't
    looking. Computed in code by ``diffing.match_loss_development``;
    matching is by claim id when the runs carry one, else by
    date + description.
    """

    appeared: list[LossClaimRef] = Field(default_factory=list)
    disappeared: list[LossClaimRef] = Field(default_factory=list)
    developed: list[ClaimDevelopment] = Field(default_factory=list)
    net_incurred_change: float = 0.0


class SubmissionDiff(BaseModel):
    """What changed since this insured's previous stored submission.

    Underwriters price the change, not the snapshot, so the deltas
    are computed in code (``diffing.py``) from the prior submission
    payload and both sides' computed facts, and carried on the
    result, into the diagnosis evidence, and into the packet.
    """

    prior_submission_id: str = ""
    prior_submitted_at: str = ""
    prior_outcome: str = ""
    prior_risk_tier: str = ""
    changes: list[DiffFieldChange] = Field(default_factory=list)
    # Per-claim loss development (appeared / disappeared / developed),
    # present whenever either side's loss run carried claims.
    loss_development: Optional[LossDevelopment] = None


# ---------------------------------------------------------------------------
# Triage + cross-check
# ---------------------------------------------------------------------------

class Triage(BaseModel):
    outcome: str = "incomplete"  # in_appetite | borderline | out_of_appetite | incomplete
    risk_tier: str = "moderate"  # low | moderate | high | elevated
    confidence: float = 0.0
    rationale: str = ""
    signals: list[str] = Field(default_factory=list)
    referral_triggers: list[str] = Field(default_factory=list)


class CrossCheck(BaseModel):
    rule_outcome: str = ""
    llm_outcome: Optional[str] = None
    llm_tier: Optional[str] = None
    llm_confidence: Optional[float] = None
    resolution: str = "not_run"  # agree | rules_authoritative | llm_adopted | rules_only | not_run
    note: str = ""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievedGuideline(BaseModel):
    guideline_id: str
    title: str = ""
    text: str = ""
    score: float = 0.0
    sources: list[str] = Field(default_factory=list)  # retrieval provenance: keyword / semantic

    @property
    def policy_id(self) -> str:  # convenience alias for shared code paths
        return self.guideline_id


# ---------------------------------------------------------------------------
# Diagnosis / options / drafts / checks
# ---------------------------------------------------------------------------

class Diagnosis(BaseModel):
    summary: str = ""
    risk_factors: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source: str = "template"  # llm | template
    note: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class RecoveryOption(BaseModel):
    kind: str  # proceed_to_quote_prep | refer_to_senior_underwriter | request_information | draft_decline_rationale
    title: str = ""
    description: str = ""
    score: float = 0.0
    recommended: bool = False


class Drafts(BaseModel):
    broker_letter: str = ""
    underwriter_packet: str = ""
    subject: str = ""


class VerifyResult(BaseModel):
    grounded: bool = True
    issues: list[str] = Field(default_factory=list)
    source: str = "checklist"  # llm | checklist
    note: str = ""


class ReviewResult(BaseModel):
    verdict: str = "pass"  # pass | concerns | block
    findings: list[str] = Field(default_factory=list)
    source: str = "checklist"  # llm | checklist
    enabled: bool = True


class GuardrailCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Validation(BaseModel):
    passed: bool = True
    checks: list[GuardrailCheck] = Field(default_factory=list)


class LineInformationRequest(BaseModel):
    """One line's slice of a bundle information request: what that
    line — and only that line — is still missing. A bundle's gaps
    live on lines, not on the bundle, and the broker is asked per
    line so nobody chases the property file for a cyber gap."""

    line_of_business: str = ""
    submission_id: str = ""  # the sub-analysis id, e.g. SUB-1014-CYBER
    missing_items: list[str] = Field(default_factory=list)


class InformationRequest(BaseModel):
    needed: bool = False
    missing_items: list[str] = Field(default_factory=list)
    message: str = ""
    # Multi-line bundles: the same request broken down per line
    # (empty for single-line submissions). ``missing_items`` stays
    # the flat union for surfaces that do not render per line.
    line_requests: list[LineInformationRequest] = Field(default_factory=list)


class AutonomyRecommendation(BaseModel):
    eligible_for_auto_triage: bool = False
    reasons: list[str] = Field(default_factory=list)


class DispatchAttempt(BaseModel):
    """One approval-webhook delivery attempt, from the ledger.

    Every dispatch (the approval-time attempt and any operator
    retry) is recorded with when it happened, what the receiver
    answered, and which signature it carried — a failed delivery is
    an operational fact with a retry path, not a status string.
    ``backoff_seconds`` is the delay the retry schedule assigned
    before this attempt (0 for the first; the schedule is bookkeeping
    for the operator-triggered retry, which does not itself sleep).
    """

    attempt: int = 1
    attempted_at: str = ""
    status: str = ""  # sent | failed
    http_status: Optional[int] = None
    signature_id: Optional[str] = None  # truncated X-Trida-Signature value
    error: Optional[str] = None
    backoff_seconds: float = 0.0


class RunTelemetry(BaseModel):
    backend: str = "mock"
    model: str = ""
    model_calls: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    latency_seconds: float = 0.0
    budget: Optional[dict[str, Any]] = None


class TraceStep(BaseModel):
    name: str
    title: str = ""
    details: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    submission_id: str
    status: str = "awaiting_underwriter"  # awaiting_underwriter | approved | rejected
    insured_name: str = ""
    line_of_business: str = ""
    # The guideline corpus version this run triaged under (see
    # guidelines_data.GUIDELINES_VERSION); the change-impact report
    # groups awaiting submissions by it. Empty on results stored
    # before corpus versioning.
    guidelines_version: str = ""
    # Renewal diff vs this insured's previous stored submission
    # (None on a first submission — absence is not a change).
    submission_diff: Optional[SubmissionDiff] = None
    extractions: list[DocumentExtraction] = Field(default_factory=list)
    facts: SubmissionFacts = Field(default_factory=SubmissionFacts)
    injection_flags: list[str] = Field(default_factory=list)
    triage: Triage = Field(default_factory=Triage)
    cross_check: Optional[CrossCheck] = None
    guidelines: list[RetrievedGuideline] = Field(default_factory=list)
    vector_store: str = ""
    diagnosis: Diagnosis = Field(default_factory=Diagnosis)
    options: list[RecoveryOption] = Field(default_factory=list)
    drafts: Optional[Drafts] = None
    verification: Optional[VerifyResult] = None
    review: Optional[ReviewResult] = None
    reviewer_blocked: bool = False
    validation: Optional[Validation] = None
    repair_attempted: bool = False
    repaired: bool = False
    repair_attempts: int = 0
    original_validation: Optional[Validation] = None
    # Borderline broker-letter phrasings found on the drafted
    # letter (see letter_repair.py): forms the blocking guardrails
    # deliberately do not catch but that still read as commitments
    # ("will now move through review", "you will hear from us in
    # the coming days"). Advisory only — they never fail a
    # validation and never reach the gate; the validate node gives
    # the drafter one steered redraft against them, and
    # ``advisory_repaired`` records whether the shipped letter
    # cleared them. The advisories stay on the result either way:
    # what the drafter produced is part of the run's audit.
    letter_advisories: list[dict] = Field(default_factory=list)
    advisory_repaired: bool = False
    information_request: Optional[InformationRequest] = None
    needs_information: bool = False
    autonomy: AutonomyRecommendation = Field(default_factory=AutonomyRecommendation)
    telemetry: Optional[RunTelemetry] = None
    trace: list[TraceStep] = Field(default_factory=list)
    decision: Optional[str] = None  # approved | rejected
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None
    # When the decision was recorded (UTC ISO). The audit trail has
    # always carried who decided and why; it now carries when, in
    # the same field, on every surface that reports the decision.
    decided_at: Optional[str] = None
    # The underwriter's FINAL outcome on the submission, when one
    # has been recorded (see service.record_final_decision): the
    # gate decision above approves or rejects the drafted packet;
    # this is what the desk ultimately did with the risk —
    # bound | declined | not_taken_up. The full append-only history
    # (including superseded entries) lives on the store record;
    # these fields project the current entry, the way decided_at
    # projects the gate decision. None until one is recorded.
    final_decision: Optional[str] = None
    final_decided_by: Optional[str] = None
    final_decided_on: Optional[str] = None  # ISO date the desk decided
    final_reason_code: Optional[str] = None
    dispatch_status: Optional[str] = None  # sent | failed | None
    dispatch_attempts: list[DispatchAttempt] = Field(default_factory=list)
    external_action_taken: bool = False
    error: Optional[str] = None
    # Multi-line bundles: the per-line sub-analyses this result rolls
    # up (empty for a single-line submission). The bundle's own
    # triage/facts/drafts are the roll-up — worst line drives the
    # outcome — and there is exactly one gate, on the bundle.
    multi_line: bool = False
    line_analyses: list["LineAnalysis"] = Field(default_factory=list)
    # Multi-line bundles: the lines whose own facts are missing
    # required elements — the lines holding the bundle back from
    # ready. The gate record names them, so the underwriter sees
    # which line blocks readiness without opening every section.
    # Empty for single-line submissions and complete bundles.
    blocking_lines: list[str] = Field(default_factory=list)


class LineAnalysis(BaseModel):
    """One line's sub-analysis inside a multi-line bundle result.

    The full per-line result rides along — its own triage, facts
    (readiness), renewal diff (keyed to this line's prior), drafts,
    and checks — so the underwriter reads the bundle as one packet
    with per-line sections, and a later renewal can diff against
    this line specifically.
    """

    line_of_business: str = ""
    submission_id: str = ""  # the sub-analysis id, e.g. SUB-1014-PROPERTY
    result: AgentResult


AgentResult.model_rebuild()


class AnalyzeRequest(BaseModel):
    submission: SubmissionInput


class DecisionRequest(BaseModel):
    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    reason: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class SnapshotPruneRequest(BaseModel):
    """An operator's keep-last-N prune of a kept-snapshot
    document (digest snapshots or calibration snapshots — one
    retention discipline, see ``service._prune_kept_snapshots``):
    who is pruning, and how many of the newest snapshots survive.
    The prune is recorded on the document it prunes; ``keep``
    below 1 is refused, because emptying the document would erase
    the baseline the trends read against."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    keep: int

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class SettingsRestoreRequest(BaseModel):
    """A human's point-in-history restore of a settings record
    (see ``proposals.py``): who is restoring, the index of the
    history entry whose state the record returns to, and an
    optional decision note — the "why" kept on the restore entry
    the way a proposal decision keeps its note. On a preview the
    note is only drafted: echoed back, stored nowhere."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    history_index: int
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class BreachAcknowledgementRequest(BaseModel):
    """An operator's acknowledgement of an exhausted breach
    notice (see ``service.acknowledge_exhausted_breach_notice``):
    who picked the case up, and an optional note — how the broker
    was reached, what was said. The note is kept on the record
    with the acknowledgement; it attests to a human action
    outside the system, it does not record a delivery."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class FollowThroughWorkedRequest(BaseModel):
    """An underwriter's worked touch on an escalated
    follow-through note (see
    ``service.mark_escalated_note_worked``): who acted on the
    note, and an optional note — what they chased, what the
    broker said. The touch is kept on the note itself; it
    attests to work done outside the system, it does not answer
    the note's outstanding items."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class ProviderImpactRequest(BaseModel):
    """The opt-in provider pass over the change-impact moved set
    (see ``service.guideline_change_impact_provider``): an optional
    per-call cap on the moved lines cross-checked, overriding
    ``PROVIDER_IMPACT_MAX_LINES`` for this call."""

    cap: Optional[int] = None


class CalibrationSnapshotRequest(BaseModel):
    """An operator's keep of the SLA calibration report (see
    ``service.snapshot_sla_calibration``): who is keeping it,
    and an optional label — the handle a budget debate cites
    ("the post-budget-change report") beyond the snapshot id.
    The label is the taker's own words, kept verbatim on the
    snapshot, its list, and its diffs."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    label: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class CalibrationLabelRenameRequest(BaseModel):
    """An operator's rename of a kept calibration report's
    label (see ``service.rename_sla_calibration_snapshot_label``):
    who is renaming, the label the report carries now, and the
    label it should carry. A label is an address, so the rename
    moves the address on the kept snapshot itself — recorded in
    its label history — instead of re-keeping the report and
    forking the citation. The new label answers to every keep's
    rules: not ``current`` (reserved), not a label another kept
    report already carries."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    from_label: Optional[str] = None
    to_label: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class SlaCalibrationDraftRequest(BaseModel):
    """An operator's draft-from-a-calibration-row request (see
    ``proposals.draft_sla_proposal_from_row``): who is drafting,
    the risk tier, and the candidate budget value of the report
    row to draft from. The row must be proposal-ready and the
    drafting discipline is the proposal scan's — the request
    names a row, it does not lower the evidence bar."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    tier: str
    value: int

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class ProposalDecisionRequest(BaseModel):
    """A human's decision on a calibration proposal (see
    ``proposals.py``): who decided, whether they accepted, and an
    optional note kept with the decision."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    accept: bool = False
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class FinalDecisionRequest(BaseModel):
    """An underwriter's final outcome on a submission (see
    ``service.record_final_decision``): the decision itself
    (``bound`` | ``declined`` | ``not_taken_up``), who decided,
    the date it was decided (ISO date; defaults to today), and an
    optional reason code (``price``, ``terms``, ``capacity``,
    ``broker_withdrew`` … — recorded as given, not enumerated, so
    a desk's own codes survive)."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    decision: str
    decided_on: Optional[str] = None
    reason_code: Optional[str] = None
    note: Optional[str] = None

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer


class DuplicateResolveRequest(BaseModel):
    """A human's resolution of a possible-duplicate finding (see
    ``service.resolve_duplicate``): who resolved it, the verdict
    (``confirmed`` | ``dismissed``), and the reason — required,
    because a dismissal is the memory that keeps the pair out of
    the queue, and an unexplained memory cannot be audited."""

    actor: Optional[str] = None
    approver: Optional[str] = None
    reviewer: Optional[str] = None
    pair_key: str
    decision: str
    reason: str

    def actor_name(self) -> Optional[str]:
        return self.actor or self.approver or self.reviewer
