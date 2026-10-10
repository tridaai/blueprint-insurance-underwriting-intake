"""Self-verification node logic: the agent critiques its own drafts.

Two paths, one contract (``VerifyResult``):

- **LLM critique** (provider mode): the drafting model is asked to
  check its own drafts against the verified facts — citations it was
  not given, figures that are not in the computed facts, bind or
  premium language. Self-critique catches composition errors the
  deterministic checklist cannot see.
- **Deterministic checklist** (default mode, and the fallback when an
  LLM critique is unusable): citation IDs in the drafts must be among
  the retrieved guidelines; no dollar figure may appear in draft prose
  (premiums are never quoted, so a figure is always suspect); bind
  language is flagged; a foreign submission id is flagged.

Verification never blocks a run by itself — it is evidence for the
underwriter and an input to the autonomy recommendation. The hard
stop is the guardrail layer.
"""

from __future__ import annotations

import re

from .guardrails import _BIND_PATTERNS, _affirmative
from .model_backends import DraftContext
from .schemas import Drafts, VerifyResult

_CITATION = re.compile(r"\[(UW-[A-Z]+-\d+)\]")
_MONEY = re.compile(r"\$\s?[\d,]+")
_ID_TOKEN = re.compile(r"\bSUB-\d{3,}\b")


def checklist_grounding(
    *, drafts: Drafts, retrieved_ids: list[str], submission_id: str
) -> list[str]:
    """Deterministic grounding issues for a draft (empty = grounded)."""
    combined = f"{drafts.subject}\n{drafts.broker_letter}\n{drafts.underwriter_packet}"
    issues: list[str] = []
    for citation in sorted(set(_CITATION.findall(combined))):
        if citation not in retrieved_ids:
            issues.append(
                f"draft cites guideline {citation}, which was not retrieved for this submission"
            )
    for amount in sorted(set(_MONEY.findall(combined))):
        issues.append(
            f"draft states a monetary figure ({amount}) — figures belong in the "
            "computed facts, not in draft prose, and a premium is never quoted"
        )
    hit = next(
        (h for h in (_affirmative(p, combined) for p in _BIND_PATTERNS) if h), None
    )
    if hit:
        issues.append(f"draft contains bind language: '{hit}'")
    for token in sorted(set(_ID_TOKEN.findall(combined))):
        if token != submission_id:
            issues.append(f"draft references a different submission id ({token})")
    return issues


def verify_drafts(
    *,
    submission,
    triage: dict,
    facts: dict,
    drafts: Drafts,
    retrieved_ids: list[str],
    backend,
) -> VerifyResult:
    """Verify a draft pair: LLM critique in provider mode, checklist otherwise."""
    issues = checklist_grounding(
        drafts=drafts, retrieved_ids=retrieved_ids, submission_id=submission.submission_id
    )
    verify_fn = getattr(backend, "verify_draft", None)
    if verify_fn is None:
        return VerifyResult(grounded=not issues, issues=issues, source="checklist")
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
        loss_summary="",
        subject=drafts.subject,
        broker_letter=drafts.broker_letter,
        underwriter_packet=drafts.underwriter_packet,
        guideline_details=[],
    )
    try:
        critique = verify_fn(context)
    except Exception as exc:  # verification never fails the run
        return VerifyResult(
            grounded=not issues,
            issues=issues,
            source="checklist",
            note=f"LLM self-verification failed ({exc}) — deterministic checklist used",
        )
    if critique is None:
        return VerifyResult(
            grounded=not issues,
            issues=issues,
            source="checklist",
            note="LLM self-verification reply was unusable — deterministic checklist used",
        )
    merged = list(dict.fromkeys([*issues, *critique["issues"]]))
    return VerifyResult(
        grounded=critique["grounded"] and not issues,
        issues=merged,
        source="llm",
        note=critique.get("summary", ""),
    )
