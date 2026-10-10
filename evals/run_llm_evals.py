"""LLM-mode evaluation pack — opt-in, needs a real provider key.

This is NOT the default gate (``run_evals.py`` is, and it stays
deterministic and offline). This pack answers the provider-mode
questions the golden gate cannot:

1. **Cross-check agreement** — how often do the deterministic rules
   and the LLM triage the golden cases the same way, and how do the
   disagreements resolve (``agree`` / ``rules_authoritative`` /
   ``llm_adopted``)?
2. **Draft safety and groundedness** — an LLM judge scores each
   produced broker letter against the computed facts: any claim not
   in the facts, any invented terms, and above all any prohibited
   promise — a quoted premium, a statement that coverage is bound,
   a promised outcome. In underwriting those are the failures that
   matter, and the guardrail layer must also have caught them.

Per-case token usage and latency are printed, with a summary at the
end; estimated cost uses the backend module's price table (models not
in the table report cost as n/a; a local Ollama run is $0).

**Regression gate** — the summary also diffs the run's flagged
verdicts against ``judge_checklist.json`` (see ``judge_gate.py``):
the families earlier runs found and the draft side fixed. A flagged
verdict landing in a pinned family is printed as a named REGRESSION
with its cases; one matching no family is a NEW FINDING. The gate
names things — it never changes the exit code, and it only reads
the judge's output (the drafts get policed; the judge never gets
tuned to pass). Each run also appends its gate summary to
``judge_history.jsonl`` and prints the run-over-run trend by
family, so improvement is a series, not an impression.

Run from the repo root, with a provider configured (see .env.example):

    MODEL_BACKEND=anthropic RETRIEVER=keyword python evals/run_llm_evals.py
    python evals/run_llm_evals.py --limit 8 --outcomes borderline,incomplete
    python evals/run_llm_evals.py --judge-model gpt-4o-mini

Exit codes: 2 = no real provider configured (fails loudly, by design);
1 = at least one draft judged ungrounded / invented-terms /
prohibited-promise, or a judge error; 0 = all judged drafts clean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_gate import (  # noqa: E402
    append_history,
    load_history,
    load_checklist,
    regression_gate,
    run_summary,
    trend_lines,
)

from underwriting_agent.facts import readiness_explanation  # noqa: E402
from underwriting_agent.graph import run_submission  # noqa: E402
from underwriting_agent.model_backends import (  # noqa: E402
    estimate_cost_usd,
    get_backend,
    parse_json_object,
)
from underwriting_agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE  # noqa: E402
from underwriting_agent.retriever import get_retriever  # noqa: E402
from underwriting_agent.schemas import SubmissionInput  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS = Path(__file__).resolve().parent / "llm_results.json"


def judge_draft(backend, result, submission, judge_model: str | None) -> dict:
    """One LLM-judge verdict for a produced draft pair. Never raises: a
    failed judge call is a ``judge_error`` verdict, which fails the pack.

    The judge sees the FULL verified facts the drafter saw — the
    submission's own figures and the cited guideline texts included —
    so a figure the drafter legitimately used is not misread as an
    invention (an earlier pack run flagged exactly those artifacts).
    """
    recommended = next((o for o in result.options if o.recommended), None)
    user = JUDGE_USER_TEMPLATE.format(
        submission_id=result.submission_id,
        insured_name=result.insured_name,
        broker_name=submission.broker_name,
        line_of_business=result.line_of_business,
        class_of_business=submission.class_of_business,
        requested_limit=submission.requested_limit or "not stated",
        requested_deductible=submission.requested_deductible or "not stated",
        revenue=submission.revenue if submission.revenue is not None else "not stated",
        years_in_business=(
            submission.years_in_business
            if submission.years_in_business is not None
            else "not stated"
        ),
        outcome=result.triage.outcome,
        risk_tier=result.triage.risk_tier,
        completeness=result.facts.completeness_score,
        readiness=result.facts.readiness,
        gap_sources=readiness_explanation(
            readiness=result.facts.readiness,
            completeness=result.facts.completeness_score,
            missing=result.facts.missing_elements,
            discrepancies=result.facts.discrepancies,
        ),
        missing=", ".join(result.facts.missing_elements) or "none",
        discrepancies="; ".join(result.facts.discrepancies) or "none",
        guidelines=(
            "\n".join(
                f"[{g.guideline_id}] {g.title}: {g.text}" for g in result.guidelines
            )
            or "none"
        ),
        loss_summary=(
            f"{result.facts.loss_summary.claim_count} claim(s), total incurred "
            f"{result.facts.loss_summary.total_incurred}, largest "
            f"{result.facts.loss_summary.largest_claim}"
        ),
        citations=", ".join(g.guideline_id for g in result.guidelines) or "none",
        recommended_option=(
            f"{recommended.title} [{recommended.kind}]" if recommended else "none"
        ),
        subject=result.drafts.subject if result.drafts else "",
        broker_letter=result.drafts.broker_letter if result.drafts else "",
        underwriter_packet=result.drafts.underwriter_packet if result.drafts else "",
    )
    try:
        text, _usage = backend.complete_with_usage(
            JUDGE_SYSTEM_PROMPT, user, max_tokens=300, model=judge_model
        )
    except Exception as exc:  # a broken judge must not look like a pass
        return {"judge_error": f"{type(exc).__name__}: {exc}"}
    data = parse_json_object(text)
    if data is None:
        return {"judge_error": "judge reply was not a JSON object"}
    try:
        return {
            "grounded": bool(data["grounded"]),
            "invented_terms": bool(data["invented_terms"]),
            "prohibited_promise": bool(data["prohibited_promise"]),
            "score": float(data["score"]),
            "rationale": str(data.get("rationale", ""))[:300],
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"judge_error": f"judge reply missing/invalid fields: {exc}"}


def _load_cases(limit: int | None, outcomes: str | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if outcomes:
        wanted = {o.strip() for o in outcomes.split(",") if o.strip()}
        cases = [c for c in cases if c["expected_outcome"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-mode eval pack (opt-in)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases")
    parser.add_argument("--outcomes", type=str, default=None, help="Comma-separated expected outcomes to include")
    parser.add_argument("--judge-model", type=str, default=None, help="Model override for the judge calls")
    args = parser.parse_args()

    backend = get_backend()
    if backend.name == "mock":
        print(
            "error: MODEL_BACKEND is not set to a real provider — this pack "
            "evaluates the LLM paths and refuses to run against the offline "
            "fallback. Configure a provider first (see .env.example), e.g. "
            "MODEL_BACKEND=anthropic ANTHROPIC_API_KEY=... ",
            file=sys.stderr,
        )
        return 2
    retriever = get_retriever()
    model = getattr(backend, "_model", "") or backend.name
    cases = _load_cases(args.limit, args.outcomes)
    print(f"LLM eval pack: {len(cases)} case(s) · backend {backend.name} · model {model}")

    resolutions: Counter = Counter()
    outcome_hits = 0
    clean = 0
    judged = 0
    total_input = total_output = 0
    rows: list[dict] = []
    failures: list[str] = []

    for case in cases:
        submission = SubmissionInput.model_validate(case["submission"])
        started = time.perf_counter()
        result = run_submission(submission, backend=backend, retriever=retriever)
        latency = round(time.perf_counter() - started, 2)
        usage = backend.usage_totals()
        total_input += int(usage.get("input_tokens", 0))
        total_output += int(usage.get("output_tokens", 0))
        if result.cross_check is not None:
            resolutions[result.cross_check.resolution] += 1
        outcome_ok = result.triage.outcome == case["expected_outcome"]
        outcome_hits += outcome_ok
        verdict = judge_draft(backend, result, submission, args.judge_model)
        judged += 1
        is_clean = (
            "judge_error" not in verdict
            and verdict["grounded"]
            and not verdict["invented_terms"]
            and not verdict["prohibited_promise"]
        )
        clean += is_clean
        if not is_clean:
            failures.append(f"{case['case_id']}: {verdict}")
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_outcome": case["expected_outcome"],
                "final_outcome": result.triage.outcome,
                "outcome_ok": outcome_ok,
                "resolution": result.cross_check.resolution if result.cross_check else None,
                "validation_passed": bool(result.validation and result.validation.passed),
                "judge": verdict,
                "latency_seconds": latency,
            }
        )
        print(
            f"  {case['case_id']}: final={result.triage.outcome} "
            f"(expected {case['expected_outcome']}) · "
            f"{rows[-1]['resolution'] or 'no cross-check'} · "
            f"judge={'clean' if is_clean else 'FLAGGED'} · {latency}s"
        )

    total = len(cases)
    cost = estimate_cost_usd(model, total_input, total_output)
    gate = regression_gate(rows, load_checklist())
    summary = {
        "backend": backend.name,
        "model": model,
        "judge_model": args.judge_model or model,
        "cases": total,
        "final_outcome_accuracy": round(outcome_hits / total, 4) if total else 0.0,
        "cross_check_resolutions": dict(resolutions),
        "drafts_judged": judged,
        "drafts_clean": clean,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": cost,
        "rows": rows,
        "failures": failures,
        "regression_gate": gate,
        "framing": (
            "Provider-mode eval over the synthetic golden set: cross-check "
            "behaviour and LLM-judged draft safety. Not a real-world "
            "accuracy claim."
        ),
    }
    RESULTS.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Final outcome accuracy : {outcome_hits}/{total}")
    print(f"Cross-check resolutions: {dict(resolutions)}")
    print(f"Drafts judged clean    : {clean}/{judged}")
    print(
        f"Tokens                 : {total_input} in / {total_output} out · "
        + (f"est. ${cost:.4f}" if cost is not None else "cost n/a (model not in the price table)")
    )
    if failures:
        print("Flagged drafts:")
        for failure in failures:
            print(f"  - {failure}")
    print(
        f"Judge regression gate  : {len(gate['regressions'])} regression(s), "
        f"{len(gate['new_findings'])} new finding(s) "
        f"({len(gate['holding'])} pinned families holding)"
    )
    for entry in gate["regressions"]:
        print(f"  REGRESSION {entry['family']} — cases: {', '.join(entry['cases'])}")
    for entry in gate["new_findings"]:
        print(f"  NEW FINDING {entry['family']} — cases: {', '.join(entry['cases'])}")
    # Remember this run, then print the trend — the history is
    # append-only, so the series can only ever gain a line.
    append_history(
        run_summary(
            rows,
            gate,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            backend=backend.name,
            model=model,
        )
    )
    for line in trend_lines(load_history()):
        print(line)
    passed = judged > 0 and clean == judged
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
