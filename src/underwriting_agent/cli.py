"""CLI — batch-style runner over sample submissions (for the traced
single-submission walkthrough, use the demo: ``python -m underwriting_agent demo``).

Usage:
    underwriting-agent                       # first bundled sample submission
    underwriting-agent --all                 # every bundled sample submission
    underwriting-agent --index 2             # a specific bundled sample
    underwriting-agent --file data/sample/sample_submissions.json --all
    underwriting-agent --all --concurrency 4 # the batch, four at a time

The default data source is the sample set bundled inside the package, so
this works identically from a source checkout and a pip install.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import silence_langchain_deprecation_warnings

# Must precede the graph import: the warning fires while langgraph loads.
silence_langchain_deprecation_warnings()

from .events import PrintingSink
from .graph import run_submission
from .ports import ModelBackend, Retriever
from .samples import load_sample_submissions
from .schemas import AgentResult, SubmissionInput
from .service import BatchItem
from .store import InMemoryStore
from .wiring import build_backend, build_retriever, build_service_from_env


def _load_submissions(path: Path | None) -> list[SubmissionInput]:
    if path is None:
        return [SubmissionInput.model_validate(item) for item in load_sample_submissions()]
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SubmissionInput.model_validate(item) for item in raw]


def _render_result(submission: SubmissionInput, result: AgentResult) -> None:
    t = result.triage
    print("=" * 72)
    print(f"Submission {result.submission_id}: {submission.insured_name} ({submission.line_of_business} / {submission.class_of_business})")
    print(f"Triage    : {t.outcome} (risk tier {t.risk_tier}, confidence {t.confidence})")
    print(f"Rationale : {t.rationale}")
    if result.cross_check is not None:
        cc = result.cross_check
        llm_part = (
            f"llm={cc.llm_outcome} ({cc.llm_confidence})"
            if cc.llm_outcome is not None
            else "llm=unavailable"
        )
        print(f"Cross-check: rules={cc.rule_outcome} vs {llm_part} -> {cc.resolution}")
    f = result.facts
    print(f"Facts     : completeness {f.completeness_score} · readiness {f.readiness} · losses {f.loss_summary.claim_count} claim(s), total incurred {f.loss_summary.total_incurred}")
    if result.injection_flags:
        print(f"Screening : {len(result.injection_flags)} injection flag(s) — flagged text kept out of prompts and drafts")
    print(f"Guidelines: {[g.guideline_id for g in result.guidelines]}")
    print(f"Diagnosis : {result.diagnosis.summary} (composed by: {result.diagnosis.source})")
    recommended = next((o for o in result.options if o.recommended), None)
    if recommended is not None:
        print(f"Next step : recommended [{recommended.kind}] {recommended.title} (score {recommended.score}) of {len(result.options)} scored option(s)")
    if result.verification is not None:
        verdict = "grounded" if result.verification.grounded else "NOT GROUNDED"
        issues = f" issues={result.verification.issues}" if result.verification.issues else ""
        print(f"Self-check: {verdict} ({result.verification.source}){issues}")
    if result.review is not None:
        findings = f" findings={result.review.findings}" if result.review.findings else ""
        blocked = " — REVIEWER BLOCKED (flagged for the underwriter)" if result.reviewer_blocked else ""
        print(f"Review    : {result.review.verdict.upper()} ({result.review.source}){findings}{blocked}")
    if result.validation is not None:
        failed = [c.name for c in result.validation.checks if not c.passed]
        print(f"Guardrails: passed={result.validation.passed}" + (f" failed={failed}" if failed else ""))
    if result.repair_attempted:
        print(f"Repair    : attempted ({result.repair_attempts}) — {'repaired' if result.repaired else 'still failing'}")
    print(f"Gate      : {result.status} | external action taken: {result.external_action_taken}")
    label = "eligible for auto-triage" if result.autonomy.eligible_for_auto_triage else "human decision required"
    print(f"Autonomy  : {label} (recommendation only) — {'; '.join(result.autonomy.reasons)}")
    if result.needs_information and result.information_request is not None:
        print(f"Info needed: information request composed (not sent) — missing: {'; '.join(result.information_request.missing_items)}")
    if result.telemetry is not None:
        tel = result.telemetry
        budget_text = ""
        if tel.budget is not None:
            budget_text = f" · token budget {tel.budget['used']}/{tel.budget['limit']}" + (
                " EXCEEDED (later steps degraded)" if tel.budget["exceeded"] else ""
            )
        if tel.input_tokens is not None:
            cost = f" · est. ${tel.estimated_cost_usd:.4f}" if tel.estimated_cost_usd is not None else " · cost n/a"
            print(f"Telemetry : {tel.backend} · {tel.model} · {tel.model_calls} model call(s) · {tel.input_tokens} in / {tel.output_tokens} out tokens{cost} · {tel.latency_seconds}s{budget_text}")
        else:
            print(f"Telemetry : {tel.backend} (offline fallback — no model ran) · {tel.model_calls} call(s) · tokens n/a · {tel.latency_seconds}s{budget_text}")
    print("-" * 72)
    if result.drafts is not None:
        print(f"Subject: {result.drafts.subject}")
        print(result.drafts.broker_letter)
    print()


def _run_batch(submissions: list[SubmissionInput], concurrency: int) -> list[BatchItem]:
    """The concurrent batch path: the service layer with an in-memory
    store (the CLI persists nothing), each item rendered as it lands.
    Configuration was already resolved by main() before this runs, so a
    misconfigured backend fails loudly there, not per item here."""
    service = build_service_from_env(store=InMemoryStore(), checkpointer=False)
    items = service.analyze_batch(submissions, concurrency=concurrency)
    for submission, item in zip(submissions, items):
        if item.error:
            print("=" * 72)
            print(f"Submission {item.submission_id}: ERROR — {item.error}")
            print()
        elif item.result is not None:
            _render_result(submission, item.result)
    return items


def _print_batch_summary(items: list[BatchItem], concurrency: int, wall_clock: float) -> None:
    print("=" * 72)
    print(
        f"Batch summary: {len(items)} submission(s) · concurrency {concurrency} "
        f"· wall-clock {wall_clock:.2f}s"
    )
    for item in items:
        if item.error:
            print(f"  {item.submission_id}: ERROR — {item.error}")
        elif item.result is not None:
            result = item.result
            print(
                f"  {item.submission_id}: {result.triage.outcome} "
                f"({result.triage.risk_tier}) — {result.status}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwriting Submission Intake Agent CLI (offline, synthetic data)")
    parser.add_argument("--file", type=Path, default=None, help="Path to a submissions JSON file (default: bundled samples)")
    parser.add_argument("--index", type=int, default=0, help="Which sample submission to run")
    parser.add_argument("--all", action="store_true", help="Run every sample submission")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="With --all: analyse N submissions concurrently (default 1 = sequential, unchanged)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print the run's structured events live as it executes (sequential path; the rendered result follows as usual)",
    )
    args = parser.parse_args()

    # Backend + retriever come from the environment (.env loaded by the
    # factories). A misconfiguration (e.g. MODEL_BACKEND=openai with no
    # key) fails loudly with the actionable message, not a traceback —
    # and so does a provider failure mid-run (e.g. Ollama not running):
    # provider errors arrive pre-translated via errors.ProviderError.
    try:
        backend = build_backend()
        retriever = build_retriever()
        submissions = _load_submissions(args.file)
        selected = submissions if args.all else [submissions[args.index]]
        started = time.perf_counter()
        if args.all and args.concurrency > 1:
            items = _run_batch(selected, args.concurrency)
        else:
            items = []
            sink = PrintingSink() if args.stream else None
            for submission in selected:
                result = run_submission(
                    submission, backend=backend, retriever=retriever, event_sink=sink
                )
                _render_result(submission, result)
                items.append(
                    BatchItem(submission_id=submission.submission_id, result=result)
                )
        if args.all:
            _print_batch_summary(items, args.concurrency, time.perf_counter() - started)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
