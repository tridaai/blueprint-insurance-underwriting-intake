"""Eval runners: the deterministic gate passes; the LLM pack is opt-in."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_shape():
    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cases) == 37
    outcomes = {c["expected_outcome"] for c in cases}
    assert outcomes == {"in_appetite", "borderline", "out_of_appetite", "incomplete"}
    tiers = {c["expected_tier"] for c in cases}
    assert tiers <= {"low", "moderate", "high", "elevated"}


def test_deterministic_runner_scores_full_marks():
    sys.path.insert(0, str(ROOT / "evals"))
    import run_evals

    cases = [
        json.loads(line)
        for line in (ROOT / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = run_evals.evaluate(cases)
    assert scored["outcome_correct"] == scored["total"] == 37
    assert scored["tier_correct"] == 37
    assert scored["checks_correct"] == scored["checks_total"] == 5
    assert scored["failures"] == []


def test_llm_pack_refuses_to_run_without_a_provider():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals" / "run_llm_evals.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": str(ROOT)},
    )
    assert proc.returncode == 2
    assert "refuses to run" in proc.stderr
