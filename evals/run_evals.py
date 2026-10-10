"""Golden-dataset evaluation for the Underwriting Submission Intake Agent.

Runs the full pipeline (deterministic rules path, offline fallback
backend) over ``golden.jsonl`` and reports triage outcome accuracy and
risk-tier accuracy. All cases are synthetic. Run from the repo root:

    python evals/run_evals.py

Exit code is 0 when outcome accuracy >= 95%, 1 otherwise (the gate).

Multi-line bundle cases (a submission carrying ``lines``) run through
the service layer — the bundle split lives there — and may pin
per-line expectations via ``expected_lines``; a bundle case's
``prior_submission`` is analysed into the same throwaway store first,
so per-line renewal diffs are scored too.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from underwriting_agent.graph import run_submission  # noqa: E402
from underwriting_agent.schemas import SubmissionInput  # noqa: E402
from underwriting_agent.service import SubmissionService  # noqa: E402
from underwriting_agent.store import InMemoryStore  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
THRESHOLD = 0.95


def _write_results(payload: dict) -> None:
    """Emit results.json next to this script (and into the package, when
    the source tree is present) so the demo console can display it."""
    targets = [Path(__file__).resolve().parent / "results.json"]
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "underwriting_agent"
        / "data"
        / "eval_results.json"
    )
    if packaged.parent.is_dir():
        targets.append(packaged)
    for target in targets:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prior_bundle(case: dict) -> dict | None:
    """The service-style prior bundle for a renewal case.

    A case may carry ``prior_submission`` — the insured's previous
    submission. It is run through the pipeline first, and the
    bundle (payload + computed facts + triage) is built exactly the
    way the service builds it from a stored record, so renewal-diff
    behaviour — including diff-driven referral triggers — is scored
    by the same gate as everything else.
    """
    raw = case.get("prior_submission")
    if not raw:
        return None
    prior = SubmissionInput.model_validate(raw)
    prior_result = run_submission(prior)
    return {
        "submission_id": prior.submission_id,
        "submitted_at": prior.submitted_at,
        "outcome": prior_result.triage.outcome,
        "risk_tier": prior_result.triage.risk_tier,
        "submission": prior.model_dump(mode="json"),
        "facts": prior_result.facts.model_dump(mode="json"),
    }


def evaluate(cases: list[dict]) -> dict:
    """Run the pipeline over the cases and score outcome + tier.

    A case may also carry ``expected_checks`` — a mapping of
    guardrail check name to the pass/fail the run's validation must
    show (e.g. a broker pressuring for forward motion must end with
    ``no_forward_motion_in_letter: false``). Those are scored too
    and reported alongside the triage accuracy. Bundle cases may
    carry ``expected_lines`` — per-line outcome/tier expectations,
    scored through the same checks accounting.
    """
    outcome_correct = 0
    tier_correct = 0
    checks_correct = 0
    checks_total = 0
    per_outcome_total: dict[str, int] = defaultdict(int)
    per_outcome_correct: dict[str, int] = defaultdict(int)
    failures: list[str] = []

    for case in cases:
        if (case["submission"] or {}).get("lines"):
            # Bundle case: the split + roll-up live in the service.
            service = SubmissionService(store=InMemoryStore(), checkpointer=False)
            if case.get("prior_submission"):
                service.analyze(case["prior_submission"])
            result = service.analyze(case["submission"])
        else:
            submission = SubmissionInput.model_validate(case["submission"])
            result = run_submission(submission, prior_submission=_prior_bundle(case))
        expected_outcome = case["expected_outcome"]
        per_outcome_total[expected_outcome] += 1
        if result.triage.outcome == expected_outcome:
            outcome_correct += 1
            per_outcome_correct[expected_outcome] += 1
        else:
            failures.append(
                f"{case['case_id']}: expected outcome {expected_outcome}, "
                f"got {result.triage.outcome}"
            )
        if result.triage.risk_tier == case["expected_tier"]:
            tier_correct += 1
        elif result.triage.outcome == expected_outcome:
            failures.append(
                f"{case['case_id']}: outcome ok, expected tier "
                f"{case['expected_tier']}, got {result.triage.risk_tier}"
            )
        expected_checks = case.get("expected_checks") or {}
        if expected_checks:
            actual = {
                c.name: c.passed for c in result.validation.checks
            } if result.validation else {}
            for name, expected in expected_checks.items():
                checks_total += 1
                if actual.get(name) == expected:
                    checks_correct += 1
                else:
                    failures.append(
                        f"{case['case_id']}: guardrail check {name} expected "
                        f"{'pass' if expected else 'fail'}, got "
                        f"{actual.get(name)}"
                    )
        expected_lines = case.get("expected_lines") or {}
        if expected_lines:
            actual_lines = {
                a.line_of_business: a.result for a in result.line_analyses
            }
            for line, expectation in expected_lines.items():
                line_result = actual_lines.get(line)
                for key, field in (
                    ("outcome", "outcome"),
                    ("tier", "risk_tier"),
                ):
                    want = expectation.get(key)
                    if want is None:
                        continue
                    checks_total += 1
                    got = (
                        getattr(line_result.triage, field)
                        if line_result is not None
                        else None
                    )
                    if got == want:
                        checks_correct += 1
                    else:
                        failures.append(
                            f"{case['case_id']}: line {line} expected "
                            f"{key} {want}, got {got}"
                        )

    total = len(cases)
    return {
        "total": total,
        "outcome_correct": outcome_correct,
        "outcome_accuracy": round(outcome_correct / total, 4) if total else 0.0,
        "tier_correct": tier_correct,
        "tier_accuracy": round(tier_correct / total, 4) if total else 0.0,
        "checks_correct": checks_correct,
        "checks_total": checks_total,
        "per_outcome": {
            outcome: {
                "correct": per_outcome_correct[outcome],
                "total": per_outcome_total[outcome],
                "accuracy": round(per_outcome_correct[outcome] / per_outcome_total[outcome], 4),
            }
            for outcome in sorted(per_outcome_total)
        },
        "failures": failures,
    }


def main() -> int:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = evaluate(cases)
    total = scored["total"]

    _write_results(
        {
            "dataset": "golden.jsonl (synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": total,
            "outcome_match": f"{scored['outcome_correct']}/{total}",
            "tier_match": f"{scored['tier_correct']}/{total}",
            "checks_match": f"{scored['checks_correct']}/{scored['checks_total']}",
            "outcome_correct": scored["outcome_correct"],
            "outcome_accuracy": scored["outcome_accuracy"],
            "tier_correct": scored["tier_correct"],
            "tier_accuracy": scored["tier_accuracy"],
            "per_outcome": scored["per_outcome"],
            "threshold": THRESHOLD,
            "passed": (
                scored["outcome_accuracy"] >= THRESHOLD
                and scored["checks_correct"] == scored["checks_total"]
            ),
            "framing": (
                "Synthetic golden set used as a regression gate. "
                "It is not a real-world accuracy claim."
            ),
        }
    )

    print(f"Golden dataset: {total} synthetic cases")
    print(
        f"Triage outcome accuracy : {scored['outcome_correct']}/{total} "
        f"= {scored['outcome_accuracy']:.1%}"
    )
    print(
        f"Risk-tier accuracy      : {scored['tier_correct']}/{total} "
        f"= {scored['tier_accuracy']:.1%}"
    )
    if scored["checks_total"]:
        print(
            f"Guardrail expectations  : {scored['checks_correct']}/"
            f"{scored['checks_total']} met"
        )
    for outcome, row in scored["per_outcome"].items():
        print(f"  {outcome:<18} {row['correct']}/{row['total']}")
    if scored["failures"]:
        print("Failures:")
        for failure in scored["failures"]:
            print(f"  - {failure}")
    passed = (
        scored["outcome_accuracy"] >= THRESHOLD
        and scored["checks_correct"] == scored["checks_total"]
    )
    print("PASS" if passed else f"FAIL (threshold {THRESHOLD:.0%})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
