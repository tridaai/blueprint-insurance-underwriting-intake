"""Golden-dataset evaluation for the Underwriting Submission Intake Agent.

Runs the full pipeline (deterministic rules path, offline fallback
backend) over ``golden.jsonl`` and reports triage outcome accuracy and
risk-tier accuracy. All cases are synthetic. Run from the repo root:

    python evals/run_evals.py

Exit code is 0 when outcome accuracy >= 95%, 1 otherwise (the gate).
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


def evaluate(cases: list[dict]) -> dict:
    """Run the pipeline over the cases and score outcome + tier."""
    outcome_correct = 0
    tier_correct = 0
    per_outcome_total: dict[str, int] = defaultdict(int)
    per_outcome_correct: dict[str, int] = defaultdict(int)
    failures: list[str] = []

    for case in cases:
        submission = SubmissionInput.model_validate(case["submission"])
        result = run_submission(submission)
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

    total = len(cases)
    return {
        "total": total,
        "outcome_correct": outcome_correct,
        "outcome_accuracy": round(outcome_correct / total, 4) if total else 0.0,
        "tier_correct": tier_correct,
        "tier_accuracy": round(tier_correct / total, 4) if total else 0.0,
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
            "outcome_correct": scored["outcome_correct"],
            "outcome_accuracy": scored["outcome_accuracy"],
            "tier_correct": scored["tier_correct"],
            "tier_accuracy": scored["tier_accuracy"],
            "per_outcome": scored["per_outcome"],
            "threshold": THRESHOLD,
            "passed": scored["outcome_accuracy"] >= THRESHOLD,
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
    for outcome, row in scored["per_outcome"].items():
        print(f"  {outcome:<18} {row['correct']}/{row['total']}")
    if scored["failures"]:
        print("Failures:")
        for failure in scored["failures"]:
            print(f"  - {failure}")
    passed = scored["outcome_accuracy"] >= THRESHOLD
    print("PASS" if passed else f"FAIL (threshold {THRESHOLD:.0%})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
