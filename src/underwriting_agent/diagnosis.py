"""Diagnose node logic: a risk summary over the computed evidence.

The diagnosis reasons over computed evidence only — the completeness
score and readiness flag, the loss-run summary, the extraction
cross-check discrepancies, the triage's own signals and referral
triggers, the retrieved guidelines, and (when the store has history)
the insured/broker memory and underwriter feedback lines. The evidence
list is assembled deterministically in BOTH modes, so the citations an
underwriter sees are always the real inputs. In provider mode the LLM
composes the risk prose from those facts — agenticly, through a
bounded tool loop when the graph supplies a toolbox
(``tools_agent.py``); in the default mode a deterministic template
composes it from the same evidence structure. An unusable LLM reply
falls back to the template, with the reason recorded in ``note``.
"""

from __future__ import annotations

from .extractor import extraction_discrepancies
from .facts import readiness_explanation
from .model_backends import DraftContext
from .schemas import Diagnosis, DocumentExtraction, SubmissionInput
from .screening import sanitized_notes
from .store import format_outcome_counts
from .tools_agent import TOOL_SPECS, DiagnosisToolBox, diagnosis_max_tool_calls


def format_loss_summary(loss: dict) -> str:
    """One computed line for the loss history (used across surfaces)."""
    if not loss or not loss.get("claim_count"):
        return "no claims in the loss history"
    return (
        f"{loss['claim_count']} claim(s) over {len(loss.get('loss_years') or [])} "
        f"year(s); total incurred {loss.get('total_incurred', 0)}; largest claim "
        f"{loss.get('largest_claim', 0)}; frequency "
        f"{loss.get('frequency_per_year', 0)} per year"
    )


def memory_evidence_lines(history: dict | None) -> list[str]:
    """History as evidence lines, in the diagnosis' citation style.

    Only real history produces lines — absence is not evidence.
    Feedback lines carry the human decisions (with their reasons) on
    earlier cases for this insured or broker, so the pattern of
    oversight itself becomes evidence.
    """
    if not history:
        return []
    lines: list[str] = []
    if history.get("insured_count"):
        recent = ", ".join(history.get("insured_recent_outcomes", [])) or "none recorded"
        lines.append(
            f"memory: {history['insured_count']} prior submission(s) for this "
            f"insured in the stored history (insured: {history.get('insured', '')}; "
            f"most recent outcomes: {recent})"
        )
    if history.get("broker_count"):
        breakdown = format_outcome_counts(history.get("broker_outcome_counts", {}))
        lines.append(
            f"broker history: {history['broker_count']} prior submission(s) from "
            f"this broker ({history.get('broker', '')}) in the stored history "
            f"(outcomes: {breakdown})"
        )
    for index, feedback in enumerate(history.get("feedback", [])):
        scope = (
            "for this insured"
            if feedback.get("match") == "insured"
            else "from this broker"
        )
        when = "last" if index == 0 else "an earlier"
        lines.append(
            f"underwriter feedback: {when} decision {scope} was "
            f"{feedback.get('decision', '')} — reason: '{feedback.get('reason', '')}'"
        )
    return lines


def build_evidence(
    *,
    triage: dict,
    facts: dict,
    extractions: list[DocumentExtraction],
    guidelines: list[dict],
    history: dict | None = None,
    discrepancies: list[str] | None = None,
    history_lines: list[str] | None = None,
) -> list[str]:
    """The deterministic evidence list every diagnosis cites.

    ``discrepancies`` / ``history_lines`` are the precomputed
    extraction cross-check and memory/feedback evidence (the graph's
    evidence fan-out computes them in parallel branches); when they
    are None they are computed here, exactly as before — the lines are
    identical either way, only *when* they are computed differs.
    """
    evidence: list[str] = []
    evidence.append(
        f"computed completeness={facts.get('completeness_score')} "
        f"readiness={facts.get('readiness')} (tool: compute_submission_facts)"
    )
    evidence.append(
        "readiness gaps (computed): "
        + readiness_explanation(
            readiness=facts.get("readiness") or "not_ready",
            completeness=facts.get("completeness_score"),
            missing=facts.get("missing_elements") or [],
            discrepancies=facts.get("discrepancies") or [],
        )
    )
    evidence.append(f"computed loss history: {format_loss_summary(facts.get('loss_summary') or {})}")
    if facts.get("missing_elements"):
        evidence.append(
            "missing elements: " + ", ".join(facts["missing_elements"])
        )
    if discrepancies is None:
        discrepancies = extraction_discrepancies(extractions)
    for line in discrepancies:
        evidence.append(f"extraction cross-check: {line}")
    if triage.get("signals"):
        evidence.append("triage signals: " + "; ".join(triage["signals"]))
    if triage.get("referral_triggers"):
        evidence.append("referral triggers: " + "; ".join(triage["referral_triggers"]))
    for g in guidelines:
        evidence.append(f"guideline {g['guideline_id']}: {g['title']}")
    evidence.extend(
        history_lines if history_lines is not None else memory_evidence_lines(history)
    )
    return evidence


def _template_risk_factors(
    *, triage: dict, facts: dict, discrepancies: list[str]
) -> list[str]:
    factors: list[str] = list(triage.get("referral_triggers") or [])
    loss = facts.get("loss_summary") or {}
    if loss.get("claim_count"):
        factors.append(
            f"loss history: {loss['claim_count']} claim(s), total incurred "
            f"{loss.get('total_incurred', 0)}, largest {loss.get('largest_claim', 0)}"
        )
    for element in facts.get("missing_elements") or []:
        factors.append(f"missing element: {element}")
    factors.extend(f"extraction discrepancy: {d}" for d in discrepancies)
    if facts.get("document_pair_warning"):
        factors.append(
            "no loss runs and no structured loss history — absence of losses "
            "is unverified"
        )
    return factors


def build_diagnosis(
    *,
    submission: SubmissionInput,
    triage: dict,
    facts: dict,
    extractions: list[DocumentExtraction],
    guidelines: list[dict],
    backend,
    history: dict | None = None,
    toolbox: DiagnosisToolBox | None = None,
    discrepancies: list[str] | None = None,
    history_lines: list[str] | None = None,
) -> Diagnosis:
    """Compose the diagnosis: LLM prose in provider mode, template otherwise.

    In provider mode with a ``toolbox`` (the graph passes one), the LLM
    diagnosis is agentic: a bounded tool loop (``tools_agent.py``) lets
    the model pull guideline search results, history, and computed
    facts before composing. Without a toolbox the single-call diagnosis
    runs. Either path degrades to the template, with the reason
    recorded in ``note``.
    """
    # The diagnosis feeds the broker's text to the model (provider
    # path); it sees the screened version — sentences the injection
    # screen flagged never reach the prompt.
    broker_text = sanitized_notes(submission)
    if discrepancies is None:
        discrepancies = extraction_discrepancies(extractions)
    evidence = build_evidence(
        triage=triage,
        facts=facts,
        extractions=extractions,
        guidelines=guidelines,
        history=history,
        discrepancies=discrepancies,
        history_lines=history_lines,
    )
    risk_factors = _template_risk_factors(
        triage=triage, facts=facts, discrepancies=discrepancies
    )
    template = Diagnosis(
        summary=(
            f"{triage['outcome']} ({triage['risk_tier']} risk) for "
            f"{submission.insured_name} — {submission.line_of_business} / "
            f"{submission.class_of_business}; completeness "
            f"{facts.get('completeness_score')}, readiness "
            f"{facts.get('readiness')}; {format_loss_summary(facts.get('loss_summary') or {})}; "
            f"{len(guidelines)} governing guideline(s) retrieved."
            + (
                " Readiness gaps: "
                + readiness_explanation(
                    readiness=facts.get("readiness") or "not_ready",
                    completeness=facts.get("completeness_score"),
                    missing=facts.get("missing_elements") or [],
                    discrepancies=facts.get("discrepancies") or [],
                )
                + "."
                if facts.get("readiness") != "ready"
                else ""
            )
        ),
        risk_factors=risk_factors,
        evidence=evidence,
        source="template",
    )
    diagnose_fn = getattr(backend, "diagnose", None)
    if diagnose_fn is None:
        return template
    context = DraftContext(
        submission_id=submission.submission_id,
        insured_name=submission.insured_name,
        broker_name=submission.broker_name,
        line_of_business=submission.line_of_business,
        class_of_business=submission.class_of_business,
        outcome=triage["outcome"],
        risk_tier=triage["risk_tier"],
        rationale=triage["rationale"],
        referral_triggers=triage.get("referral_triggers") or [],
        completeness=facts.get("completeness_score"),
        readiness=facts.get("readiness"),
        missing=facts.get("missing_elements") or [],
        loss_summary=format_loss_summary(facts.get("loss_summary") or {}),
        discrepancies=discrepancies,
        broker_text=broker_text,
        guideline_details=guidelines,
    )
    tools_fn = getattr(backend, "diagnose_with_tools", None)
    if tools_fn is not None and toolbox is not None:
        try:
            llm = tools_fn(
                context, TOOL_SPECS, toolbox.dispatch, diagnosis_max_tool_calls()
            )
        except Exception as exc:  # the diagnosis never fails the run
            llm = None
            template.note = (
                f"LLM agentic diagnosis failed ({exc}) — template diagnosis used"
            )
        if llm:
            return Diagnosis(
                summary=llm["summary"],
                risk_factors=llm["risk_factors"],
                evidence=evidence,
                        source="llm",
                tool_calls=llm.get("tool_calls", []),
            )
        if not template.note:
            template.note = (
                "LLM agentic diagnosis reply was unusable — template diagnosis used"
            )
        return template
    try:
        llm = diagnose_fn(context)
    except Exception as exc:  # the diagnosis never fails the run
        llm = None
        template.note = f"LLM diagnosis failed ({exc}) — template diagnosis used"
    if not llm:
        if not template.note:
            template.note = "LLM diagnosis reply was unusable — template diagnosis used"
        return template
    return Diagnosis(
        summary=llm["summary"],
        risk_factors=llm["risk_factors"],
        evidence=evidence,
        source="llm",
    )
