"""One-command demo: ONE submission, end to end, with a readable trace.

    underwriting-agent-demo            (console script)
    python -m underwriting_agent demo  (module form)
    make demo

Default: offline, deterministic mock backend, synthetic sample data. Set
MODEL_BACKEND=openai|anthropic (plus the API key, e.g. in .env) and the
same trace runs against the real provider — the header line says which
backend produced the drafts. The trace shows every stage a reviewer needs
to judge the blueprint: input → extraction + computed facts → triage with
cross-check → retrieved guidelines → diagnosis → scored options → drafts →
self-verification → independent review → guardrail checks → the
underwriter gate it stops at.
"""

from __future__ import annotations

import argparse
import sys

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .graph import run_submission
from .ports import ModelBackend, Retriever
from .samples import sample_submission_models
from .wiring import build_backend, build_retriever

_LINE = "=" * 74


def print_trace(
    index: int = 0,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
) -> None:
    backend = backend or build_backend()
    retriever = retriever or build_retriever()
    submissions = sample_submission_models()
    submission = submissions[index]
    result = run_submission(submission, backend=backend, retriever=retriever)
    t = result.triage
    backend_desc = (
        "offline mock backend"
        if backend.name == "mock"
        else f"{backend.name} backend via MODEL_BACKEND (real provider calls)"
    )

    print(_LINE)
    print("TRIDA AI BLUEPRINT DEMO — Insurance Underwriting Submission Intake Agent")
    print(f"Reference prototype · synthetic data · {backend_desc} · no external action")
    print(_LINE)

    print("\n[1] INPUT SUBMISSION")
    print(f"    ID          : {submission.submission_id}")
    print(f"    Insured     : {submission.insured_name} · Broker: {submission.broker_name}")
    print(f"    Line / class: {submission.line_of_business} / {submission.class_of_business}")
    print(f"    Requested   : limit {submission.requested_limit or '—'} · deductible {submission.requested_deductible or '—'} · effective {submission.effective_date_requested or '—'}")
    print(f"    Documents   : {[d.document_id or d.doc_type for d in submission.documents] or 'none'}")
    print(f"    Losses      : {len(submission.losses)} structured loss record(s)")
    for flag in result.injection_flags:
        print(f"    ⚠ Injection screen: FLAG — {flag}")
    if result.injection_flags:
        print("      (flagged sentences are kept out of the drafts and the prompts; the facts still triage the case)")
    f = result.facts
    print(f"    Computed    : completeness {f.completeness_score} · readiness {f.readiness} · submission age {f.submission_age_days} day(s)")
    print(f"    Computed    : loss history — {f.loss_summary.claim_count} claim(s), total incurred {f.loss_summary.total_incurred}, largest {f.loss_summary.largest_claim}")
    if f.missing_elements:
        print(f"    Missing     : {', '.join(f.missing_elements)}")
    if f.document_pair_warning:
        print(f"    Warning     : {f.document_pair_warning}")
    for ext in result.extractions:
        notable = [fl for fl in ext.fields if fl.status not in ("match", "source_provided")]
        if notable:
            for fl in notable:
                print(f"    Extraction  : {ext.document_id} field '{fl.field}' — {fl.status} (confidence {fl.confidence})")

    print("\n[2] TRIAGE")
    print(f"    Outcome   : {t.outcome}")
    print(f"    Risk tier : {t.risk_tier}   Confidence: {t.confidence}")
    print(f"    Rationale : {t.rationale}")
    print("    Evidence (signals):")
    for signal in t.signals or ["(none)"]:
        print(f"      - {signal}")
    for trigger in t.referral_triggers:
        print(f"      - referral trigger: {trigger}")
    if result.cross_check is not None:
        cc = result.cross_check
        print(f"    Rules       : {cc.rule_outcome}")
        if cc.llm_outcome is not None:
            print(f"    LLM check   : {cc.llm_outcome} · tier {cc.llm_tier} · confidence {cc.llm_confidence}")
        else:
            print("    LLM check   : no usable LLM triage — rules only")
        print(f"    Cross-check : {cc.resolution} — {cc.note}")

    print("\n[3] RETRIEVED GUIDELINE CONTEXT")
    for g in result.guidelines:
        snippet = g.text if len(g.text) <= 160 else g.text[:157] + "..."
        print(f"    [{g.guideline_id}] {g.title} (score {g.score}, via {'+'.join(g.sources) or 'keyword'})")
        print(f"        {snippet}")
    if not result.guidelines:
        print("    (no guidelines retrieved)")

    print("\n[4] DIAGNOSIS")
    print(f"    Summary   : {result.diagnosis.summary} (composed by: {result.diagnosis.source})")
    for factor in result.diagnosis.risk_factors:
        print(f"      - risk factor: {factor}")
    for call in result.diagnosis.tool_calls:
        print(f"      - tool call: {call.get('name')} — {call.get('summary')}")
    if result.diagnosis.note:
        print(f"    Note      : {result.diagnosis.note}")
    for line in result.diagnosis.evidence:
        print(f"      - evidence: {line}")

    print("\n[5] NEXT-STEP OPTIONS (scored by code — the model never does this arithmetic)")
    for option in result.options:
        mark = " <- recommended" if option.recommended else ""
        print(f"    [{option.kind}] {option.title} — score {option.score}{mark}")

    print("\n[6] DRAFT — BROKER LETTER")
    if result.drafts is not None:
        print(f"    Subject: {result.drafts.subject}")
        for line in result.drafts.broker_letter.splitlines():
            print(f"    {line}")

    print("\n[7] DRAFT — UNDERWRITER PACKET (internal, for the underwriter's decision)")
    if result.drafts is not None:
        for line in result.drafts.underwriter_packet.splitlines():
            print(f"    {line}")

    print("\n[8] SELF-VERIFICATION — the agent critiques its own drafts")
    if result.verification is not None:
        verdict = "GROUNDED" if result.verification.grounded else "NOT GROUNDED"
        source = "LLM critique" if result.verification.source == "llm" else "deterministic checklist"
        print(f"    verdict: {verdict} ({source})")
        for issue in result.verification.issues:
            print(f"    issue: {issue}")
        if result.verification.note:
            print(f"    note: {result.verification.note}")

    print("\n[9] INDEPENDENT REVIEW — a second pair of eyes (not the drafter)")
    if result.review is not None:
        r = result.review
        source = "LLM reviewer" if r.source == "llm" else "deterministic checklist"
        print(f"    verdict: {r.verdict.upper()} ({source})")
        for finding in r.findings:
            print(f"    finding: {finding}")
        if result.reviewer_blocked:
            print("    ⛔ REVIEWER BLOCKED this draft — flagged for the underwriter; the human still decides.")
    else:
        print("    (disabled — REVIEWER=off)")

    print("\n[10] GUARDRAIL CHECKS")
    if result.validation is not None:
        for check in result.validation.checks:
            mark = "PASS" if check.passed else "FAIL"
            print(f"    [{mark}] {check.name} — {check.detail}")
        print(f"    Overall: {'PASSED' if result.validation.passed else 'FAILED'}")
    if result.repair_attempted:
        outcome = "redraft passed" if result.repaired else "redraft still failed"
        print(f"    Repair: bounded repair attempted ({result.repair_attempts} attempt(s)) — {outcome}")
        if result.original_validation is not None:
            for failed in [c for c in result.original_validation.checks if not c.passed]:
                print(f"    Original failure: {failed.name} — {failed.detail}")

    print("\n[11] FINAL STATE")
    print("    AWAITING_UNDERWRITER" if result.status == "awaiting_underwriter" else f"    {result.status}")
    print(f"    status = {result.status} · external_action_taken = {result.external_action_taken}")
    label = "eligible for auto-triage" if result.autonomy.eligible_for_auto_triage else "human decision required"
    print(f"    Autonomy recommendation: {label} (recommendation only — never acted on)")
    for reason in result.autonomy.reasons:
        print(f"      - {reason}")
    if result.needs_information and result.information_request is not None:
        print("    Information : NEEDED — a request to the broker was composed (not sent):")
        for item in result.information_request.missing_items:
            print(f"      - missing: {item}")
    if result.telemetry is not None:
        tel = result.telemetry
        budget_text = ""
        if tel.budget is not None:
            budget_text = f" · token budget {tel.budget['used']}/{tel.budget['limit']}" + (
                " — EXCEEDED (later provider steps degraded to their template paths)"
                if tel.budget["exceeded"]
                else ""
            )
        if tel.input_tokens is not None:
            cost = (
                f" · est. ${tel.estimated_cost_usd:.4f}"
                if tel.estimated_cost_usd is not None
                else " · cost n/a (model not in the price table)"
            )
            print(
                f"    Telemetry : {tel.backend} · {tel.model} · {tel.model_calls} model call(s) · "
                f"{tel.input_tokens} in / {tel.output_tokens} out tokens{cost} · {tel.latency_seconds}s{budget_text}"
            )
        else:
            print(
                f"    Telemetry : {tel.backend} (offline fallback — no model ran) · "
                f"{tel.model_calls} call(s) · tokens n/a · {tel.latency_seconds}s{budget_text}"
            )
    print("    Nothing was quoted, bound, sent, or filed. An underwriter decides via the API/UI.")
    print(_LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwriting Submission Intake Agent — traced demo (offline)")
    parser.add_argument("--index", type=int, default=0, help="Which bundled sample submission to run (0-based)")
    args = parser.parse_args()
    try:
        print_trace(args.index)
    except (RuntimeError, ValueError) as exc:
        # e.g. MODEL_BACKEND=openai with no key — fail loudly and cleanly.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
