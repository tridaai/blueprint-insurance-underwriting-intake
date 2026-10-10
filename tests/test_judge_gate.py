"""Judge-pack regression gate: the diffing logic is deterministic.

The pack itself is opt-in and needs a provider; the gate over its
output is not — pinned families, classification, and the regression
diff are plain functions, tested here with verdicts shaped like the
ones the two real pack runs produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import judge_gate  # noqa: E402

CHECKLIST = judge_gate.load_checklist()
KNOWN_IDS = {family["id"] for family in CHECKLIST}


def _row(case_id: str, **verdict) -> dict:
    base = {
        "grounded": True,
        "invented_terms": False,
        "prohibited_promise": False,
        "score": 1.0,
        "rationale": "The drafts accurately reflect the verified facts.",
    }
    base.update(verdict)
    return {"case_id": case_id, "judge": base}


def test_checklist_pins_the_known_families():
    assert KNOWN_IDS == {
        "forward_motion",
        "review_commitment",
        "soft_time_commitment",
        "figure_leakage",
        "invented_detail",
    }
    for family in CHECKLIST:
        assert family["signals"], family["id"]
        assert family["origin"], family["id"]
        assert family["expectation"] == "must_not_recur"


def test_forward_motion_verdict_is_a_named_regression():
    rows = [
        _row(
            "C9",
            grounded=False,
            prohibited_promise=True,
            score=0.5,
            rationale=(
                "The broker letter promises the submission will proceed "
                "to quote preparation at standard authority."
            ),
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    families = {entry["family"] for entry in gate["regressions"]}
    assert "forward_motion" in families
    assert gate["new_findings"] == []


def test_review_commitment_verdict_is_a_named_regression():
    rows = [
        _row(
            "E1",
            grounded=False,
            prohibited_promise=True,
            score=0.65,
            rationale=(
                "The letter states the file is now under senior "
                "underwriter review, a commitment about the case."
            ),
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert {entry["family"] for entry in gate["regressions"]} >= {"review_commitment"}
    assert gate["regressions"][0]["cases"] == ["E1"] or any(
        entry["cases"] == ["E1"] for entry in gate["regressions"]
    )


def test_soft_time_commitment_is_a_named_regression():
    rows = [
        _row(
            "M4",
            grounded=False,
            prohibited_promise=True,
            score=0.7,
            rationale="The letter says 'we will be in touch shortly', a soft time commitment.",
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert "soft_time_commitment" in {e["family"] for e in gate["regressions"]}


def test_invented_figures_are_figure_leakage():
    rows = [
        _row(
            "C1",
            grounded=False,
            invented_terms=True,
            prohibited_promise=True,
            score=0.45,
            rationale=(
                "The drafts invent specific revenue figures ($4M), "
                "coverage limits ($5M), and authority thresholds ($100M)."
            ),
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert "figure_leakage" in {e["family"] for e in gate["regressions"]}


def test_groundedness_slip_falls_to_invented_detail():
    rows = [
        _row(
            "E3",
            grounded=False,
            score=0.65,
            rationale="The letter claims source documentation was not located, contradicting the record.",
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert {e["family"] for e in gate["regressions"]} == {"invented_detail"}


def test_unmatched_flag_is_a_new_finding_not_a_regression():
    rows = [
        _row(
            "X1",
            grounded=True,
            prohibited_promise=True,
            score=0.8,
            rationale="The letter thanks the broker in unusually effusive terms.",
        )
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert gate["regressions"] == []
    assert gate["new_findings"] == [{"family": "unclassified", "cases": ["X1"]}]


def test_clean_run_holds_every_family():
    rows = [_row("C1"), _row("C2")]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert gate["regressions"] == []
    assert gate["new_findings"] == []
    assert set(gate["holding"]) == KNOWN_IDS


def test_judge_errors_never_classify():
    rows = [{"case_id": "C1", "judge": {"judge_error": "provider timeout"}}]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    assert gate["regressions"] == []
    assert gate["new_findings"] == []
    assert set(gate["holding"]) == KNOWN_IDS


# ---------------------------------------------------------------------------
# History + trend: the gate, remembered across runs
# ---------------------------------------------------------------------------


def _summary(at: str, flagged: int, cases: int, **by_family: int) -> dict:
    return {
        "generated_at": at,
        "backend": "anthropic",
        "model": "haiku",
        "cases": cases,
        "flagged": flagged,
        "by_family": by_family,
        "regressions": [],
        "new_findings": [],
        "holding": [],
    }


def test_run_summary_counts_flagged_cases_once_and_families_fully():
    rows = [
        _row(
            "E1",
            grounded=False,
            prohibited_promise=True,
            rationale=(
                "The letter states we will move this submission forward "
                "to quote file preparation."
            ),
        ),
        _row("E2"),
        {"case_id": "E3", "judge": {"judge_error": "provider timeout"}},
    ]
    gate = judge_gate.regression_gate(rows, CHECKLIST)
    summary = judge_gate.run_summary(
        rows, gate, generated_at="2026-10-10T13:00:00+00:00",
        backend="anthropic", model="haiku",
    )
    assert summary["cases"] == 3
    assert summary["flagged"] == 1  # the judge error is not a finding
    assert summary["by_family"] == {"forward_motion": 1}
    assert summary["regressions"] == ["forward_motion"]
    assert "invented_detail" in summary["holding"]


def test_history_is_append_only_and_round_trips(tmp_path):
    path = tmp_path / "judge_history.jsonl"
    assert judge_gate.load_history(path) == []  # missing file = empty
    first = _summary("2026-10-01T00:00:00+00:00", 9, 28, forward_motion=6)
    second = _summary("2026-10-08T00:00:00+00:00", 5, 28, forward_motion=3)
    judge_gate.append_history(first, path)
    judge_gate.append_history(second, path)
    loaded = judge_gate.load_history(path)
    assert loaded == [first, second]  # oldest first, nothing rewritten
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one JSON object per line


def test_load_history_skips_a_malformed_line(tmp_path):
    path = tmp_path / "judge_history.jsonl"
    good = _summary("2026-10-01T00:00:00+00:00", 1, 4, invented_detail=1)
    judge_gate.append_history(good, path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
    assert judge_gate.load_history(path) == [good]


def test_trend_lines_show_each_family_across_runs():
    history = [
        _summary("2026-09-24T00:00:00+00:00", 9, 28,
                 forward_motion=6, invented_detail=3),
        _summary("2026-10-01T00:00:00+00:00", 5, 28,
                 forward_motion=3, invented_detail=2),
        _summary("2026-10-08T00:00:00+00:00", 2, 28, invented_detail=2),
    ]
    lines = judge_gate.trend_lines(history)
    assert lines[0] == "Judge trend over 3 recorded run(s), oldest first:"
    assert lines[1] == "  flagged drafts: 9/28 -> 5/28 -> 2/28"
    # A family absent from a run counts 0 there — the burn-down shows.
    assert "  forward_motion: 6 -> 3 -> 0" in lines
    assert "  invented_detail: 3 -> 2 -> 2" in lines


def test_trend_lines_for_an_empty_history():
    assert judge_gate.trend_lines([]) == ["Judge trend: no recorded runs yet."]
