"""Synthetic underwriting guidelines corpus.

These guidelines are INVENTED for this blueprint. They are generic
specialty-lines underwriting language written for a reference
prototype — they are not any real insurer's guidelines, appetite, or
wording, and they describe no client engagement.

To adapt the blueprint, replace this corpus with your own guidelines
(and keep the mirror in ``data/sample/guidelines.json`` in sync).
"""

from __future__ import annotations

GUIDELINES: list[dict] = [
    {
        "id": "UW-CYB-01",
        "title": "Cyber appetite — professional and technology services",
        "text": (
            "Cyber liability is in appetite for professional services and technology firms with "
            "annual revenue up to 250 million, multi-factor authentication enforced, and endpoint "
            "detection in place. Submissions with no ransomware-related loss in the last three "
            "years may proceed to quote preparation at standard authority."
        ),
        "lines": ["cyber"],
    },
    {
        "id": "UW-CYB-02",
        "title": "Cyber referral triggers",
        "text": (
            "Refer cyber submissions to a senior underwriter when revenue exceeds standard "
            "authority, when there has been a ransomware or business-email-compromise loss in the "
            "last three years, or when the insured cannot evidence multi-factor authentication."
        ),
        "lines": ["cyber"],
    },
    {
        "id": "UW-PROP-01",
        "title": "Property appetite and loss history",
        "text": (
            "Commercial property is in appetite for well-maintained risks with modern protection. "
            "Three or more claims in five years, or a single claim above 250,000 incurred, is a "
            "referral trigger: loss history of that weight requires senior underwriter review of "
            "cause, remediation, and current protection before any quote preparation."
        ),
        "lines": ["property"],
    },
    {
        "id": "UW-PROP-02",
        "title": "Statement of values standard",
        "text": (
            "A current statement of values is required for property submissions with more than "
            "three locations. Missing or stale values documents make the submission incomplete; "
            "request an updated schedule before proceeding."
        ),
        "lines": ["property"],
    },
    {
        "id": "UW-DNO-01",
        "title": "D&O documentation standard",
        "text": (
            "Directors and officers submissions require two years of financial statements and a "
            "completed application signed by an officer. Without current financials the submission "
            "is incomplete: do not assess risk or indicate terms until financials are received "
            "and reviewed."
        ),
        "lines": ["directors_and_officers", "d&o"],
    },
    {
        "id": "UW-DNO-02",
        "title": "D&O referral triggers",
        "text": (
            "Refer directors and officers submissions when the insured has pending or threatened "
            "litigation, a going-concern qualification, or revenue growth above 100 percent year "
            "over year, or when requested limits exceed standard authority."
        ),
        "lines": ["directors_and_officers", "d&o"],
    },
    {
        "id": "UW-GEN-01",
        "title": "Out-of-appetite classes",
        "text": (
            "The following classes are outside appetite and should receive a drafted decline "
            "rationale for underwriter review: hazardous waste transport, fireworks manufacturing, "
            "and cryptocurrency mining operations. Declines are always issued by a human "
            "underwriter, never automatically."
        ),
        "lines": ["general"],
    },
    {
        "id": "UW-GEN-02",
        "title": "Documentation and completeness standard",
        "text": (
            "A complete submission contains a completed application form, three years of currently "
            "valued loss runs (or a signed no-loss letter), and the financial or values documents "
            "required for the line. Submissions missing required elements are incomplete: request "
            "the specific missing items from the broker and do not proceed."
        ),
        "lines": ["general"],
    },
    {
        "id": "UW-GEN-03",
        "title": "Authority and referral by revenue and limits",
        "text": (
            "Standard underwriter authority covers insureds with revenue up to 100 million and "
            "requested limits up to 5 million. Above either threshold, refer to a senior "
            "underwriter with a full packet: summary, loss analysis, and guideline citations."
        ),
        "lines": ["general"],
    },
    {
        "id": "UW-GEN-04",
        "title": "Broker communications standard",
        "text": (
            "Broker communications acknowledge receipt, state what has been received, list "
            "exactly what is still needed, and give a next-update commitment the team can staff. "
            "Communications never quote a firm premium, never state or imply that coverage is "
            "bound or in force, and never promise an outcome of underwriting review."
        ),
        "lines": ["general"],
    },
    {
        "id": "UW-EO-01",
        "title": "Professional liability (E&O) appetite",
        "text": (
            "Errors and omissions cover for consultants, engineers, and professional firms is in "
            "appetite where the insured has at least three years in business, no unresolved "
            "professional liability claims, and contracts that limit liability. First-year firms "
            "and firms with open claims are referred."
        ),
        "lines": ["professional_liability", "e&o"],
    },
    {
        "id": "UW-MAR-01",
        "title": "Marine and inland marine notes",
        "text": (
            "Inland marine submissions require an equipment schedule with serial numbers and "
            "storage/security details for values above 500,000 at any one location. Missing "
            "schedules make the submission incomplete for that location's values."
        ),
        "lines": ["marine", "inland_marine"],
    },
]
