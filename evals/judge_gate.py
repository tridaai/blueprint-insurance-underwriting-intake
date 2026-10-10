"""Judge-pack regression gate — deterministic, offline, no model.

The LLM-judge pack (``run_llm_evals.py``) finds real problems, but
a finding that is fixed and then quietly reintroduced by a later
prompt change is worse than a new one: it means the loop is not
holding. This module pins the known finding families
(``judge_checklist.json``) and diffs each pack run against them:

- a flagged verdict classified into a **checklist family** is a
  **REGRESSION** — named, with its cases, in the pack output;
- a flagged verdict matching **no family** is a **NEW FINDING** —
  it belongs in the checklist once it is understood and fixed;
- a checklist family **not detected** in the run is **holding**.

Classification is deliberately simple and deterministic: lowercase
substring signals matched against the judge's rationale, families
in checklist order (most specific first), plus one catch-all — a
groundedness failure matching nothing else is an invented detail.
A verdict may carry several families; the gate reports them all.

The gate never changes the pack's verdict: any flagged draft still
fails the pack. And it only ever reads the judge's output — the
drafts get policed, the judge never gets tuned to pass.

**History + trend** — each pack run appends one summary line to
``judge_history.jsonl`` (append-only: a run can add its line, it
can never rewrite an earlier one). The pack prints the trend —
flagged-case counts by family across runs — so "the loop is
improving" is a series of numbers with dates on it, not an
impression from the latest run.
"""

from __future__ import annotations

import json
from pathlib import Path

CHECKLIST_PATH = Path(__file__).resolve().parent / "judge_checklist.json"
HISTORY_PATH = Path(__file__).resolve().parent / "judge_history.jsonl"
UNCLASSIFIED = "unclassified"


def load_checklist(path: Path | None = None) -> list[dict]:
    """The pinned families, in file order (most specific first)."""
    data = json.loads((path or CHECKLIST_PATH).read_text(encoding="utf-8"))
    return data["families"]


def verdict_is_flagged(verdict: dict) -> bool:
    """A judged draft the pack counts as not clean (judge errors are
    pack failures, but they are not findings and never classify)."""
    if not verdict or "judge_error" in verdict:
        return False
    return (
        not verdict.get("grounded", False)
        or bool(verdict.get("invented_terms"))
        or bool(verdict.get("prohibited_promise"))
    )


def classify_verdict(verdict: dict, checklist: list[dict]) -> set[str]:
    """The checklist family ids one flagged verdict belongs to.

    Empty for clean verdicts and judge errors. A flagged verdict
    matching no signal falls to the ``invented_detail`` catch-all
    when it is a groundedness failure, else to ``unclassified``.
    """
    if not verdict_is_flagged(verdict):
        return set()
    rationale = str(verdict.get("rationale", "")).lower()
    matched = {
        family["id"]
        for family in checklist
        if any(signal.lower() in rationale for signal in family.get("signals", []))
    }
    if not matched and not verdict.get("grounded", True):
        known = {family["id"] for family in checklist}
        matched = {"invented_detail"} if "invented_detail" in known else {UNCLASSIFIED}
    return matched or {UNCLASSIFIED}


def regression_gate(rows: list[dict], checklist: list[dict]) -> dict:
    """Diff one pack run's rows (``{case_id, judge}``) against the
    checklist. Regressions and new findings name their cases."""
    known = {family["id"] for family in checklist}
    detected: dict[str, list[str]] = {}
    for row in rows:
        for family_id in classify_verdict(row.get("judge") or {}, checklist):
            detected.setdefault(family_id, []).append(row["case_id"])
    return {
        "regressions": [
            {"family": family_id, "cases": sorted(cases)}
            for family_id, cases in sorted(detected.items())
            if family_id in known
        ],
        "new_findings": [
            {"family": family_id, "cases": sorted(cases)}
            for family_id, cases in sorted(detected.items())
            if family_id not in known
        ],
        "holding": sorted(known - set(detected)),
        "detected": {family_id: sorted(cases) for family_id, cases in sorted(detected.items())},
    }


# ---------------------------------------------------------------------------
# Run history + trend — the gate, remembered
# ---------------------------------------------------------------------------


def run_summary(
    rows: list[dict],
    gate: dict,
    *,
    generated_at: str,
    backend: str = "",
    model: str = "",
) -> dict:
    """One pack run as a history line: when, on what backend, how
    many drafts were flagged, and the flagged-case count per
    family. ``flagged`` counts cases once; ``by_family`` counts a
    case under every family it matched, like the gate itself."""
    return {
        "generated_at": generated_at,
        "backend": backend,
        "model": model,
        "cases": len(rows),
        "flagged": sum(
            1 for row in rows if verdict_is_flagged(row.get("judge") or {})
        ),
        "by_family": {
            family_id: len(cases)
            for family_id, cases in sorted(gate.get("detected", {}).items())
        },
        "regressions": [entry["family"] for entry in gate.get("regressions", [])],
        "new_findings": [entry["family"] for entry in gate.get("new_findings", [])],
        "holding": list(gate.get("holding", [])),
    }


def append_history(summary: dict, path: Path | None = None) -> Path:
    """Append one run summary to the history file. Append-only by
    construction — the file is the trend's memory, and a run that
    could rewrite it could also flatter it."""
    target = path or HISTORY_PATH
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    return target


def load_history(path: Path | None = None) -> list[dict]:
    """Every recorded run summary, oldest first. A missing file is
    an empty history; a malformed line is skipped, never fatal —
    the trend is a report, not a gate."""
    target = path or HISTORY_PATH
    if not target.is_file():
        return []
    runs: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def trend_lines(history: list[dict]) -> list[str]:
    """The trend as printable lines: flagged totals, then one line
    per family with its flagged-case count in each run, oldest
    first. Deterministic — same history, same lines."""
    if not history:
        return ["Judge trend: no recorded runs yet."]
    families: list[str] = []
    for run in history:
        for family_id in run.get("by_family", {}):
            if family_id not in families:
                families.append(family_id)
    lines = [
        f"Judge trend over {len(history)} recorded run(s), oldest first:",
        "  flagged drafts: "
        + " -> ".join(
            f"{run.get('flagged', 0)}/{run.get('cases', 0)}" for run in history
        ),
    ]
    for family_id in families:
        counts = " -> ".join(
            str(run.get("by_family", {}).get(family_id, 0)) for run in history
        )
        lines.append(f"  {family_id}: {counts}")
    return lines
