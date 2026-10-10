"""Information-request flow: when a submission cannot be assessed.

A submission triaged ``incomplete`` is under-determined: the computed
facts say required elements are absent, so the only honest next step
is asking the broker for them. This module composes that request — the
missing-items list is computed by code (``facts.py``), never invented
by the model; in provider mode the model only words the message.

The request is composed, never sent — like every output here it stops
at the underwriter gate, and leaves only if a human approves it and
routes it through their own system.
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import (
    InformationRequest,
    LineAnalysis,
    LineInformationRequest,
    SubmissionFacts,
    SubmissionInput,
)

# Broker-facing wording for each computed missing element.
_ELEMENT_PHRASES = {
    "application": "a completed application (ACORD) form",
    "revenue": "the insured's annual revenue",
    "loss_runs_or_no_loss": "loss runs for the last three years, or a signed no-loss letter",
    "mfa_attestation": "confirmation that multi-factor authentication is enforced on email and remote access",
    "values_schedule": "a statement of values covering every location",
    "financials": "the last two years of financial statements",
    "construction_details": "construction details for each location (construction type, year built, roof/HVAC updates)",
    "years_in_business": "the number of years the firm has been in business",
    "equipment_schedule": "a schedule of the vessels/equipment to be covered",
}


def humanize_element(element: str) -> str:
    return _ELEMENT_PHRASES.get(element, element.replace("_", " "))


def missing_items(facts: SubmissionFacts, extractions: list | None = None) -> list[str]:
    """The broker-facing missing-items list, computed from the facts
    (plus critical fields the extraction cross-check could not find in
    the document text)."""
    items = [humanize_element(e) for e in facts.missing_elements]
    for extraction in extractions or []:
        for field in extraction.fields:
            if field.status == "missing_in_text" and field.field in (
                "insured_name",
                "line_of_business",
            ):
                phrase = f"the {field.field.replace('_', ' ')} stated in the documents"
                if phrase not in items:
                    items.append(phrase)
    return items


def needs_information(triage_outcome: str, items: list[str]) -> bool:
    """The trigger: an incomplete triage with something concrete to ask for."""
    return triage_outcome == "incomplete" and bool(items)


def template_message(submission: SubmissionInput, items: list[str]) -> str:
    lines = [
        f"Re: submission {submission.submission_id} — {submission.insured_name} "
        f"({submission.line_of_business})",
        "",
        "Thank you for the submission. Before it can be assessed, we still need:",
        *[f"- {item}" for item in items],
        "",
        "Once we have these, the submission will be triaged in full. This "
        "message requests information only — it is not a quotation and "
        "nothing in it binds or places coverage.",
    ]
    return "\n".join(lines)


def build_information_request(
    *,
    submission: SubmissionInput,
    triage_outcome: str,
    facts: SubmissionFacts,
    extractions: list | None,
    backend,
) -> InformationRequest | None:
    """Compose the request when the trigger fires; None otherwise.

    In provider mode the LLM words the message over the computed
    items; the template is the fallback and the default-mode path.
    """
    items = missing_items(facts, extractions)
    if not needs_information(triage_outcome, items):
        return None
    message = template_message(submission, items)
    compose = getattr(backend, "compose_information_request", None)
    if compose is not None:
        context = DraftContext(
            submission_id=submission.submission_id,
            insured_name=submission.insured_name,
            broker_name=submission.broker_name,
            line_of_business=submission.line_of_business,
            missing_items=items,
        )
        try:
            llm_message = compose(context)
        except Exception:  # the request never fails the run
            llm_message = None
        if llm_message:
            message = llm_message
    return InformationRequest(needed=True, missing_items=items, message=message)


def bundle_template_message(
    submission: SubmissionInput, line_requests: list[LineInformationRequest]
) -> str:
    """The bundle request, sectioned per line: each gapped line
    gets its own ask, lines with nothing missing are not listed —
    the broker is never sent chasing one line's file for another
    line's gap."""
    lines = [
        f"Re: submission {submission.submission_id} — {submission.insured_name} "
        "(multi-line bundle)",
        "",
        "Thank you for the submission. Before it can be assessed, we "
        "still need the following, by line:",
    ]
    for line_request in line_requests:
        lines.append("")
        lines.append(f"For the {line_request.line_of_business} line:")
        lines += [f"- {item}" for item in line_request.missing_items]
    lines += [
        "",
        "Lines not listed above have everything they need. Once we "
        "have these items, the affected lines will be triaged in "
        "full. This message requests information only — it is not a "
        "quotation and nothing in it binds or places coverage.",
    ]
    return "\n".join(lines)


def build_bundle_information_request(
    *,
    submission: SubmissionInput,
    triage_outcome: str,
    analyses: list[LineAnalysis],
) -> InformationRequest | None:
    """Compose a bundle's information request, per line.

    Each line's items are computed from that line's own facts —
    the request asks only the lines that are actually missing
    something, each for exactly its own gaps. The message is
    composed in code (not model-worded): line attribution is
    structural, and a model paraphrase is exactly how a cyber gap
    ends up described as a property one. Composed, never sent —
    the same discipline as every request here.
    """
    line_requests: list[LineInformationRequest] = []
    flat: list[str] = []
    for analysis in analyses:
        items = missing_items(analysis.result.facts, analysis.result.extractions)
        if not items:
            continue
        line_requests.append(
            LineInformationRequest(
                line_of_business=analysis.line_of_business,
                submission_id=analysis.submission_id,
                missing_items=items,
            )
        )
        for item in items:
            if item not in flat:
                flat.append(item)
    if not needs_information(triage_outcome, flat):
        return None
    return InformationRequest(
        needed=True,
        missing_items=flat,
        message=bundle_template_message(submission, line_requests),
        line_requests=line_requests,
    )
