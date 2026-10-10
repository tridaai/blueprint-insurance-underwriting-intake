"""Computed facts — in code, never by the model.

Completeness, the data-readiness flag, and the loss summary are the
numbers the whole pipeline reasons over. They are computed here from
the submission and the extraction cross-check, deterministically, so
the same submission always produces the same facts.
"""

from __future__ import annotations

from datetime import date, datetime

from .guidelines_data import GUIDELINES  # noqa: F401  (re-export convenience)
from .schemas import DocumentExtraction, LossSummary, SubmissionFacts, SubmissionInput

# Required elements per line of business (keys of the completeness score).
REQUIRED_ELEMENTS: dict[str, list[str]] = {
    "cyber": ["application", "revenue", "loss_runs_or_no_loss", "mfa_attestation"],
    "property": ["application", "values_schedule", "loss_runs_or_no_loss", "construction_details"],
    "directors_and_officers": ["application", "financials", "loss_runs_or_no_loss"],
    "professional_liability": ["application", "revenue", "loss_runs_or_no_loss", "years_in_business"],
    "marine": ["application", "equipment_schedule", "loss_runs_or_no_loss"],
    "general": ["application", "loss_runs_or_no_loss"],
}

LINE_ALIASES = {
    "d&o": "directors_and_officers",
    "d and o": "directors_and_officers",
    "e&o": "professional_liability",
    "errors_and_omissions": "professional_liability",
    "inland_marine": "marine",
    "cyber_liability": "cyber",
    "commercial_property": "property",
}


def normalize_line(line: str) -> str:
    key = " ".join(str(line or "").strip().lower().split())
    return LINE_ALIASES.get(key) or LINE_ALIASES.get(key.replace(" ", "_")) or key.replace(" ", "_")


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(s)[:10])
        except ValueError:
            return None


def _doc_types(sub: SubmissionInput) -> set[str]:
    return {d.doc_type for d in sub.documents}


def _all_text(sub: SubmissionInput) -> str:
    parts = [sub.notes, sub.broker_email_text]
    parts += [d.raw_text for d in sub.documents]
    return "\n".join(parts).lower()


def element_present(sub: SubmissionInput, element: str) -> bool:
    types = _doc_types(sub)
    text = _all_text(sub)
    if element == "application":
        return "acord_form" in types
    if element == "financials":
        return "financials" in types
    if element == "values_schedule":
        return "sov" in types
    if element == "equipment_schedule":
        return "sov" in types or "equipment schedule" in text
    if element == "loss_runs_or_no_loss":
        if "loss_run" in types or sub.losses:
            return True
        return any(p in text for p in ("no losses", "no loss", "no claims", "loss-free", "no prior claims"))
    if element == "revenue":
        if sub.revenue is not None:
            return True
        return any(d.fields.get("revenue") for d in sub.documents)
    if element == "years_in_business":
        if sub.years_in_business is not None:
            return True
        return any(d.fields.get("years_in_business") for d in sub.documents)
    if element == "mfa_attestation":
        return any(p in text for p in ("multi-factor", "multifactor", "mfa", "two-factor"))
    if element == "construction_details":
        return any(p in text for p in ("construction", "built in", "sprinkler", "frame", "masonry"))
    return False


def compute_facts(
    sub: SubmissionInput,
    extractions: list[DocumentExtraction] | None = None,
    today: date | None = None,
) -> SubmissionFacts:
    line = normalize_line(sub.line_of_business)
    required = REQUIRED_ELEMENTS.get(line, REQUIRED_ELEMENTS["general"])
    missing = [e for e in required if not element_present(sub, e)]
    completeness = 1.0 - (len(missing) / len(required)) if required else 1.0

    discrepancies: list[str] = []
    for ext in extractions or []:
        for f in ext.fields:
            if f.status == "mismatch":
                discrepancies.append(
                    f"{ext.document_id or ext.doc_type}: field '{f.field}' differs from the document text"
                )

    if completeness >= 0.999 and not discrepancies:
        readiness = "ready"
    elif completeness >= 0.5:
        readiness = "ready_with_gaps"
    else:
        readiness = "not_ready"

    losses = sub.losses
    years = sorted({int(str(l.date)[:4]) for l in losses if str(l.date)[:4].isdigit()})
    span = max(1, (max(years) - min(years) + 1)) if years else 1
    loss_summary = LossSummary(
        claim_count=len(losses),
        total_incurred=round(sum(l.amount_incurred for l in losses), 2),
        largest_claim=max((l.amount_incurred for l in losses), default=0.0),
        loss_years=years,
        frequency_per_year=round(len(losses) / span, 2) if losses else 0.0,
    )

    submitted = _parse_date(sub.submitted_at)
    age = ((today or date.today()) - submitted).days if submitted else 0

    warning = ""
    types = _doc_types(sub)
    if "loss_run" not in types and not sub.losses:
        warning = (
            "no loss runs provided and no structured loss history — loss history "
            "assessment is based on the submission text only"
        )

    return SubmissionFacts(
        completeness_score=round(completeness, 3),
        missing_elements=missing,
        readiness=readiness,
        loss_summary=loss_summary,
        submission_age_days=max(0, age),
        discrepancy_count=len(discrepancies),
        discrepancies=discrepancies,
        document_pair_warning=warning,
    )
