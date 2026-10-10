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
"""

from __future__ import annotations

import json
from pathlib import Path

CHECKLIST_PATH = Path(__file__).resolve().parent / "judge_checklist.json"
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
