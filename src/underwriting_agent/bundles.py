"""Multi-line submissions: one arrival, several lines, one gate.

Real broker submissions bundle lines — property + cyber on the same
account, sharing one insured, one broker, one email. This module is
the bundle mechanics, all computed in code:

- **Split** (``split_bundle``): the bundle becomes one ordinary
  submission per line — shared insured context, the line's own
  documents/limits/loss run — so each line runs the *same* pipeline
  (extraction, facts, triage, diffing, diagnosis) with its own
  outcome and readiness. Nothing about the per-line engine changes.
- **Roll up** (``rollup_triage`` / ``rollup_facts``): the bundle's
  headline triage follows its most severe line — a bundle is only
  as clean as its worst line — while every line keeps its own full
  result on ``AgentResult.line_analyses``.
- **Combine** (``compose_bundle_drafts``): one broker letter and
  one underwriter packet for the bundle, the packet carrying a
  per-line section with each line's own packet text. The combined
  drafts go through the same guardrails, verification, and review
  as any draft, and there is exactly **one underwriter gate** — on
  the bundle. Decisions are recorded per bundle; the sub-analyses
  are sections of the packet, not separately decidable records.
"""

from __future__ import annotations

from .facts import normalize_line
from .model_backends import _NEXT_STEP, _TRIAGE_SENTENCE
from .schemas import (
    Drafts,
    LineAnalysis,
    LossSummary,
    SubmissionFacts,
    SubmissionInput,
    Triage,
)
from .screening import sanitized_notes

OUTCOME_SEVERITY = {
    "in_appetite": 0,
    "borderline": 1,
    "incomplete": 2,
    "out_of_appetite": 3,
}
TIER_SEVERITY = {"low": 0, "moderate": 1, "high": 2, "elevated": 3}
READINESS_SEVERITY = {"ready": 0, "ready_with_gaps": 1, "not_ready": 2}


def is_bundle(model: SubmissionInput) -> bool:
    return bool(model.lines)


def line_slug(line: str) -> str:
    return normalize_line(line).replace("_", "-").upper() or "LINE"


def split_bundle(model: SubmissionInput) -> list[SubmissionInput]:
    """Split a bundle into one ordinary submission per line.

    Sub-analysis ids derive from the bundle id (``SUB-1014`` →
    ``SUB-1014-PROPERTY``). Bundle-level documents are shared —
    every line sees the account's application — and documents on
    the line entry belong to that line alone; likewise the loss
    run is per line. Line fields left empty fall back to the
    bundle-level values.
    """
    subs: list[SubmissionInput] = []
    used: set[str] = set()
    for entry in model.lines:
        slug = line_slug(entry.line_of_business)
        if slug in used:  # the same line twice: keep ids unique
            suffix = 2
            while f"{slug}-{suffix}" in used:
                suffix += 1
            slug = f"{slug}-{suffix}"
        used.add(slug)
        subs.append(
            SubmissionInput(
                submission_id=f"{model.submission_id}-{slug}",
                insured_name=model.insured_name,
                broker_name=model.broker_name,
                broker_email_text=model.broker_email_text,
                notes=model.notes,
                line_of_business=entry.line_of_business,
                class_of_business=entry.class_of_business or model.class_of_business,
                submitted_at=model.submitted_at,
                effective_date_requested=(
                    entry.effective_date_requested or model.effective_date_requested
                ),
                revenue=entry.revenue if entry.revenue is not None else model.revenue,
                years_in_business=(
                    entry.years_in_business
                    if entry.years_in_business is not None
                    else model.years_in_business
                ),
                prior_carrier=entry.prior_carrier or model.prior_carrier,
                requested_limit=entry.requested_limit or model.requested_limit,
                requested_deductible=(
                    entry.requested_deductible or model.requested_deductible
                ),
                documents=[*model.documents, *entry.documents],
                losses=list(entry.losses),
            )
        )
    return subs


def driving_line_index(analyses: list[LineAnalysis]) -> int:
    """The line that drives the bundle's gate: the most severe
    outcome, ties broken by risk tier, then by line order."""
    return max(
        range(len(analyses)),
        key=lambda i: (
            OUTCOME_SEVERITY.get(analyses[i].result.triage.outcome, 0),
            TIER_SEVERITY.get(analyses[i].result.triage.risk_tier, 0),
            -i,
        ),
    )


def rollup_triage(analyses: list[LineAnalysis]) -> Triage:
    """The bundle's headline triage: the driving (worst) line's
    outcome and tier, every line's referral triggers kept and
    attributed to their line."""
    driving = analyses[driving_line_index(analyses)].result.triage
    per_line = "; ".join(
        f"{a.line_of_business} — {a.result.triage.outcome} / {a.result.triage.risk_tier}"
        for a in analyses
    )
    triggers: list[str] = []
    for a in analyses:
        triggers.extend(
            f"[{a.line_of_business}] {t}" for t in a.result.triage.referral_triggers
        )
    return Triage(
        outcome=driving.outcome,
        risk_tier=driving.risk_tier,
        confidence=driving.confidence,
        rationale=(
            f"Multi-line bundle ({len(analyses)} lines): {per_line}. "
            "The bundle outcome follows the most severe line — one gate "
            "decides the bundle, and no line is quoted or bound on its own."
        ),
        signals=[f"{a.line_of_business}: {a.result.triage.outcome} / {a.result.triage.risk_tier}" for a in analyses]
        + [f"bundle outcome follows the most severe line ({analyses[driving_line_index(analyses)].line_of_business})"],
        referral_triggers=triggers,
    )


def rollup_facts(analyses: list[LineAnalysis]) -> SubmissionFacts:
    """The bundle's headline facts: mean completeness, the worst
    readiness, the lines' losses combined, and the gap sources kept
    per line (so the packet can still say *where* a gap lives)."""
    facts = [a.result.facts for a in analyses]
    completeness = round(sum(f.completeness_score for f in facts) / len(facts), 3)
    readiness = max(
        (f.readiness for f in facts),
        key=lambda r: READINESS_SEVERITY.get(r, 0),
    )
    missing: list[str] = []
    for f in facts:
        for element in f.missing_elements:
            if element not in missing:
                missing.append(element)
    discrepancies: list[str] = []
    gap_sources: list[str] = []
    for a in analyses:
        discrepancies.extend(
            f"{a.line_of_business}: {d}" for d in a.result.facts.discrepancies
        )
        gap_sources.extend(
            f"{a.line_of_business}: {g}" for g in a.result.facts.gap_sources
        )
    claim_count = sum(f.loss_summary.claim_count for f in facts)
    years = sorted({y for f in facts for y in f.loss_summary.loss_years})
    span = max(1, (max(years) - min(years) + 1)) if years else 1
    loss_summary = LossSummary(
        claim_count=claim_count,
        total_incurred=round(sum(f.loss_summary.total_incurred for f in facts), 2),
        largest_claim=max((f.loss_summary.largest_claim for f in facts), default=0.0),
        loss_years=years,
        frequency_per_year=round(claim_count / span, 2) if claim_count else 0.0,
    )
    warnings = [
        f"{a.line_of_business}: {a.result.facts.document_pair_warning}"
        for a in analyses
        if a.result.facts.document_pair_warning
    ]
    return SubmissionFacts(
        completeness_score=completeness,
        missing_elements=missing,
        readiness=readiness,
        loss_summary=loss_summary,
        submission_age_days=max((f.submission_age_days for f in facts), default=0),
        discrepancy_count=len(discrepancies),
        discrepancies=discrepancies,
        gap_sources=gap_sources,
        document_pair_warning="; ".join(warnings),
    )


def bundle_citations(analyses: list[LineAnalysis]) -> list[str]:
    """Every guideline any line retrieved, in line order, deduped —
    the citation set the combined drafts are validated against."""
    citations: list[str] = []
    for a in analyses:
        for guideline in a.result.guidelines:
            if guideline.guideline_id not in citations:
                citations.append(guideline.guideline_id)
    return citations


def compose_bundle_drafts(
    model: SubmissionInput, analyses: list[LineAnalysis]
) -> Drafts:
    """One letter + one packet for the bundle, in the template
    drafts' discipline: status only in the letter, facts in the
    packet, per-line sections verbatim from the lines' own packets.
    The combined drafts are guardrail-validated as a whole by the
    caller, exactly like a single-line draft pair."""
    triage = rollup_triage(analyses)
    driving = analyses[driving_line_index(analyses)]
    lines_text = " + ".join(a.line_of_business for a in analyses)
    subject = (
        f"Submission {model.submission_id} — {model.insured_name} "
        f"(multi-line bundle: {lines_text}): received and triaged"
    )

    recommended = next(
        (o for o in driving.result.options if o.recommended), None
    )
    next_step = _NEXT_STEP.get(
        recommended.kind if recommended else "",
        "we will update you when the underwriting review reaches its next step.",
    )
    citations = bundle_citations(analyses)
    citation_text = " ".join(f"[{c}]" for c in citations)
    paragraphs = [
        f"Dear {model.broker_name} team,",
        (
            f"Thank you for submission {model.submission_id} for "
            f"{model.insured_name}, a multi-line bundle covering "
            f"{lines_text}. Each line was triaged on its own facts:"
        ),
        "\n".join(
            f"- {a.line_of_business}: "
            f"{_TRIAGE_SENTENCE.get(a.result.triage.outcome, '')}"
            for a in analyses
        ),
    ]
    broker_text = sanitized_notes(model)
    if broker_text:
        paragraphs.append(f"For reference, your note: \"{broker_text}\"")
    paragraphs += [
        f"Next step: {next_step}",
        f"Reference guidelines: {citation_text}",
        (
            "This letter acknowledges receipt and records the triage for "
            "underwriter review. It is not a quotation, and nothing in it "
            "binds or places coverage.\nUnderwriting Intake Team"
        ),
    ]
    letter = "\n\n".join(paragraphs)

    packet_lines = [
        f"UNDERWRITER PACKET — submission {model.submission_id} "
        "(multi-line bundle, draft, for underwriter decision)",
        f"Insured: {model.insured_name} · Broker: {model.broker_name}",
        f"Lines: {lines_text}",
        f"Bundle triage: {triage.outcome} · Risk tier: {triage.risk_tier} "
        f"· Confidence: {triage.confidence}",
        f"Rationale: {triage.rationale}",
        "Per-line outcomes:",
    ]
    packet_lines += [
        f"- {a.line_of_business} ({a.submission_id}): "
        f"{a.result.triage.outcome} / {a.result.triage.risk_tier} · "
        f"readiness {a.result.facts.readiness} · "
        f"completeness {a.result.facts.completeness_score}"
        for a in analyses
    ]
    for a in analyses:
        packet_lines += [
            "",
            f"===== Line: {a.line_of_business} — {a.result.triage.outcome} / "
            f"{a.result.triage.risk_tier} (sub-analysis {a.submission_id}) =====",
            a.result.drafts.underwriter_packet if a.result.drafts else "(no packet drafted)",
        ]
    packet_lines.append(
        "Status: draft — awaiting underwriter decision on the bundle. "
        "No premium is quoted and no coverage is bound by this packet."
    )
    return Drafts(
        subject=subject,
        broker_letter=letter,
        underwriter_packet="\n".join(packet_lines),
    )
