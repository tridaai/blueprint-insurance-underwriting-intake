"""Letter repair steering: the drafting side of the letter guardrail.

The guardrail (``guardrails.no_forward_motion_in_letter``) is the
gate: it has never moved, and drafts that fail it stay
unapprovable. What the live judge runs kept showing is a drafting
problem, not a gate problem: a real model, handed the failed
checks' names, restates the rule back and produces the same
near-miss phrasings again — "will proceed to underwriter review",
"we will be in touch shortly" — because a rule restated is not a
rewrite instruction. The bounded repair loop was feeding it
exactly that.

This module is the steering between the gate and the drafter. It
changes no verdict:

- :func:`letter_repair_guidance` turns a failed letter check into
  surgical feedback: the offending sentence, quoted from the
  draft itself; the family it belongs to; and the approved status
  shape to rewrite it into — the same shapes the template letter
  uses, so the fix the model is handed is the house style, not an
  improvisation.
- :func:`letter_advisories` scans for **borderline** phrasing:
  forms the blocking patterns do not catch but a reader — or a
  judge — still hears as a commitment ("will now move through
  review", "you will hear from us in the coming days"). An
  advisory never blocks, never fails a validation, and never
  reaches the gate; it only earns the draft one steered redraft
  in the validate node, and the run records the advisories it
  found, so what the drafter produced stays auditable even when
  the redraft clears it.
- The same steering serves the **status note** (see
  :func:`status_note_repair_guidance` and :func:`steer_status_note`):
  the note is held to the letter's families by
  ``guardrails.validate_status_note``, but its drafter is code,
  not a model, so where the letter gets guidance in a prompt,
  the note gets guidance plus a deterministic sentence rewrite —
  the offending sentence replaced by the family's approved note
  shape — and the service revalidates the rewrite exactly as the
  graph revalidates a redraft. Steering covers the wording
  families only: a note failing anything else (bind language, a
  premium figure, a promise, PII) is refused outright, as it
  always was — refusal is the last resort for wording, and the
  only resort for substance.
- The same steering serves the **broker quality report** (see
  :func:`broker_quality_repair_guidance` and
  :func:`steer_broker_quality_draft`): the third broker-facing
  draft, composed in code from record prose like the note,
  validated by ``guardrails.validate_broker_quality_draft``
  and steered deterministically in the report's own shapes —
  so every broker-facing draft answers to one discipline.

The blocking families themselves are not re-implemented here:
:func:`guardrails.letter_wording_hits` is the one computation,
read by the gate and the steering alike.
"""

from __future__ import annotations

import re

from .guardrails import letter_wording_hits
from .schemas import Drafts, Validation

#: What each letter family forbids, in the guardrail's own terms —
#: the problem half of a repair instruction.
_FAMILY_PROBLEM = {
    "forward_motion": (
        "a forward-motion commitment — the letter reports status; "
        "the underwriter decides the next step, and the letter "
        "never commits the file forward"
    ),
    "review_commitment": (
        "a review commitment — the letter never commits the file "
        "to a review step or presents a routing decision as "
        "already made"
    ),
    "soft_time": (
        "a time commitment — the letter gives no response "
        "timeline, soft or otherwise"
    ),
    "authority": (
        "an internal authority reference — authority limits and "
        "thresholds are internal facts the packet records and "
        "the letter never quotes"
    ),
}

#: The approved rewrite for each family — the template letter's
#: own status shapes, so a repaired sentence lands in the house
#: style. Authority has no letter shape at all: the sentence's
#: reference comes out, and the fact stays where it belongs.
_FAMILY_INSTEAD = {
    "forward_motion": (
        "The submission is under review; an underwriter will "
        "review it and decide the next step."
    ),
    "review_commitment": (
        "This letter acknowledges receipt and records the triage "
        "for underwriter review."
    ),
    "soft_time": (
        "We will update you when the review reaches its next step."
    ),
    "authority": (
        "remove the authority reference from the letter entirely "
        "— the packet keeps that fact"
    ),
}

#: Borderline phrasings — advisory only. Each one skirts a
#: blocking family without tripping it: motion into a process
#: without "we" or "to quote", a routing step stated as a future
#: fact, a response promised with the timeline left vague. A
#: human judge reads every one of them as a commitment, which is
#: exactly why they steer a redraft — and exactly why they are
#: not a gate: the blocking lines were drawn deliberately, and
#: widening them by stealth would move the gate, not the drafts.
_BORDERLINE_PATTERNS = [
    ("forward_motion", r"\bwill (now )?(move|proceed|advance) (through|into)\b"),
    ("forward_motion", r"\b(moves|moved|moving) (into|through) (the )?(underwriter )?review\b"),
    ("review_commitment", r"\bwill be reviewed by\b"),
    ("soft_time", r"\bhear from us\b"),
    ("soft_time", r"\bbe in touch\b"),
    ("soft_time", r"\bin the coming days\b"),
]


def _sentences(text: str) -> list[str]:
    """A letter's sentences, coarsely: paragraph breaks first,
    then sentence punctuation. Good enough to quote the sentence
    a hit lives in — the drafter needs the neighbourhood, not a
    parse tree."""
    pieces: list[str] = []
    for chunk in re.split(r"\n+", text or ""):
        for piece in re.split(r"(?<=[.!?])\s+", chunk.strip()):
            piece = piece.strip()
            if piece:
                pieces.append(piece)
    return pieces


def _sentence_with(sentences: list[str], phrase: str) -> str:
    """The sentence carrying ``phrase`` (case-insensitive), or
    the phrase itself when no sentence boundary holds it."""
    lowered = phrase.lower()
    for sentence in sentences:
        if lowered in sentence.lower():
            return sentence
    return phrase


def letter_advisories(letter: str) -> list[dict]:
    """Borderline phrasings in a broker letter, as advisories.

    Each advisory names its family (the blocking family it
    skirts), the matched phrase, the sentence it sits in, the
    problem in the guardrail's terms, and the approved shape to
    rewrite into. One advisory per pattern — the first hit is
    enough steering. Advisories are a drafting instrument only:
    nothing in the gate, the service, or the approval path reads
    them as a verdict.
    """
    text = letter or ""
    sentences = _sentences(text)
    found: list[dict] = []
    for family, pattern in _BORDERLINE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is None:
            continue
        phrase = match.group(0)
        found.append(
            {
                "family": family,
                "phrase": phrase,
                "sentence": _sentence_with(sentences, phrase),
                "problem": _FAMILY_PROBLEM[family],
                "instead": _FAMILY_INSTEAD[family],
            }
        )
    return found


def _guidance_block(header: str, items: list[tuple[str, str, str]]) -> str:
    lines = [header]
    for sentence, problem, instead in items:
        lines.append(f'- Offending: "{sentence}"')
        lines.append(f"  Problem: {problem}")
        lines.append(f"  Rewrite as: {instead}")
    return "\n".join(lines)


def letter_repair_guidance(validation: Validation, drafts: Drafts) -> str:
    """Surgical rewrite instructions for a failed letter check.

    Built from the guardrail's own hits on the failing draft
    (:func:`guardrails.letter_wording_hits`): each family's
    offending sentence is quoted verbatim and paired with the
    problem and the approved shape. Empty when the letter check
    did not fail — other failures keep the plain failed-check
    feedback they have always had.
    """
    failed = any(
        check.name == "no_forward_motion_in_letter" and not check.passed
        for check in validation.checks
    )
    if not failed:
        return ""
    letter = drafts.broker_letter or ""
    sentences = _sentences(letter)
    items = []
    for family, hit in letter_wording_hits(letter).items():
        if hit is None:
            continue
        items.append(
            (
                _sentence_with(sentences, hit),
                f"{_FAMILY_PROBLEM[family]} (matched: '{hit}')",
                _FAMILY_INSTEAD[family],
            )
        )
    if not items:
        return ""
    return _guidance_block(
        "LETTER WORDING — the broker letter failed the status-only "
        "rule. Rewrite each offending sentence below into the "
        "approved shape; keep every other sentence, and every "
        "fact, exactly as it is:",
        items,
    )


def advisory_repair_guidance(advisories: list[dict]) -> str:
    """The steered redraft's instructions for borderline phrasing:
    the same surgical shape as the blocking repair — offending
    sentence, problem, approved rewrite — under a header that
    says plainly the draft passed and this is about how it
    reads."""
    if not advisories:
        return ""
    items = [
        (adv["sentence"], adv["problem"], adv["instead"])
        for adv in advisories
    ]
    return _guidance_block(
        "BORDERLINE WORDING — the broker letter passes the "
        "guardrails, but the sentences below still read as "
        "commitments a broker could hold us to. Rewrite each one "
        "into the approved shape; change nothing else:",
        items,
    )


# ---------------------------------------------------------------------------
# Status-note steering — the same discipline for the other draft
# ---------------------------------------------------------------------------

#: The approved rewrite for each family in a status note: the
#: letter's shapes restated for an internal reader (the note's
#: reader is the underwriter, not the broker, so there is no
#: "you"). Authority again has no shape at all — the reference
#: comes out, and the fact stays on the record it belongs to.
_NOTE_FAMILY_INSTEAD = {
    "forward_motion": (
        "The case is under review; an underwriter will review "
        "it and decide the next step."
    ),
    "review_commitment": (
        "This note records where the case stands for "
        "underwriter review."
    ),
    "soft_time": (
        "The case will be updated when the review reaches its "
        "next step."
    ),
    "authority": (
        "remove the authority reference from the note entirely "
        "— the record keeps that fact"
    ),
}


def status_note_repair_guidance(validation: Validation, note: str) -> str:
    """Surgical rewrite instructions for a failed status note.

    The note-side twin of :func:`letter_repair_guidance`: built
    from the guardrail's own hits (``letter_wording_hits`` — the
    note validator's wording check runs the same families), each
    family's offending sentence quoted verbatim from the note
    and paired with the problem and the approved note shape.
    Empty when the note's wording check did not fail — the
    steering exists for the wording families only.
    """
    failed = any(
        check.name == "no_forward_motion_in_letter" and not check.passed
        for check in validation.checks
    )
    if not failed:
        return ""
    text = note or ""
    sentences = _sentences(text)
    items = []
    for family, hit in letter_wording_hits(text).items():
        if hit is None:
            continue
        items.append(
            (
                _sentence_with(sentences, hit),
                f"{_FAMILY_PROBLEM[family]} (matched: '{hit}')",
                _NOTE_FAMILY_INSTEAD[family],
            )
        )
    if not items:
        return ""
    return _guidance_block(
        "STATUS NOTE WORDING — the drafted status note failed "
        "the status-only rule. Rewrite each offending sentence "
        "below into the approved shape; keep every other "
        "sentence, and every fact, exactly as it is:",
        items,
    )


def _steer_wording(
    text: str, validation: Validation, shapes: dict[str, str]
) -> str | None:
    """The deterministic sentence rewrite behind the steered
    redrafts: every sentence carrying a wording hit is replaced
    by its family's approved shape from ``shapes`` (an
    authority-carrying sentence comes out — the record keeps
    that fact), and every other sentence is kept byte for byte.
    Returns the rewrite, or ``None`` when steering does not
    apply: the wording check is not the *only* failure (bind
    language, a premium figure, a promise, or PII is refused,
    never steered around), or no wording hit can be located in
    the text. The rewrite is a draft, not a verdict — the
    caller revalidates it and serves it only if it passes.
    """
    failed = {check.name for check in validation.checks if not check.passed}
    if failed != {"no_forward_motion_in_letter"}:
        return None
    text = text or ""
    hits = letter_wording_hits(text)
    families = [family for family, hit in hits.items() if hit]
    if not families:
        return None
    lines: list[str] = []
    for line in text.split("\n"):
        if not any(
            (hits[family] or "").lower() in line.lower() for family in families
        ):
            lines.append(line)
            continue
        pieces = []
        for piece in re.split(r"(?<=[.!?])\s+", line):
            matched: str | None = None
            for family in families:
                hit = hits[family]
                if hit and hit.lower() in piece.lower():
                    matched = family
                    break
            if matched is None:
                pieces.append(piece)
            elif matched != "authority":
                pieces.append(shapes[matched])
            # An authority-carrying sentence is not rewritten —
            # it comes out. The fact it referenced stays on the
            # record, where internal facts belong.
        lines.append(" ".join(pieces))
    steered = "\n".join(lines)
    return steered if steered != text else None


def steer_status_note(note: str, validation: Validation) -> str | None:
    """One bounded steered redraft of a failed status note.

    The note's drafter is code, so where the letter's steering
    is handed to a model as a prompt, the note's is applied
    directly: every sentence carrying a wording hit is replaced
    by its family's approved note shape (an authority-carrying
    sentence comes out — the record keeps that fact), and every
    other sentence is kept byte for byte. Returns the rewritten
    note, or ``None`` when steering does not apply: the
    validation passed, the wording check is not the *only*
    failure (bind language, a premium figure, a promise, or PII
    in a note is refused, never steered around), or no wording
    hit can be located in the text. The rewrite is a draft, not
    a verdict — the caller revalidates it with
    ``guardrails.validate_status_note`` and stores it only if
    it passes, exactly as the graph stores only a passing
    redraft.
    """
    return _steer_wording(note or "", validation, _NOTE_FAMILY_INSTEAD)


# ---------------------------------------------------------------------------
# Broker-quality-report steering — the third drafting surface
# ---------------------------------------------------------------------------

#: The approved rewrite for each family in a broker quality
#: report: the shapes restated for a report whose reader may
#: be the broker (a human shares it; the agent never sends
#: it), written about the broker's record in the third person.
#: Authority again has no shape — the reference comes out.
_REPORT_FAMILY_INSTEAD = {
    "forward_motion": (
        "The figures record the submissions as they stand; "
        "each submission is decided on its own file."
    ),
    "review_commitment": (
        "This report records the broker's submission history "
        "for internal review."
    ),
    "soft_time": (
        "This report is re-read when a human reviews the "
        "broker's record again."
    ),
    "authority": (
        "remove the authority reference from the report "
        "entirely — the record keeps that fact"
    ),
}


def broker_quality_repair_guidance(validation: Validation, text: str) -> str:
    """Surgical rewrite instructions for a failed quality report.

    The report-side twin of :func:`status_note_repair_guidance`:
    built from the guardrail's own hits (``letter_wording_hits``
    — the report validator's wording check runs the same
    families), each family's offending sentence quoted verbatim
    and paired with the problem and the approved report shape.
    Empty when the report's wording check did not fail — the
    steering exists for the wording families only.
    """
    failed = any(
        check.name == "no_forward_motion_in_letter" and not check.passed
        for check in validation.checks
    )
    if not failed:
        return ""
    body = text or ""
    sentences = _sentences(body)
    items = []
    for family, hit in letter_wording_hits(body).items():
        if hit is None:
            continue
        items.append(
            (
                _sentence_with(sentences, hit),
                f"{_FAMILY_PROBLEM[family]} (matched: '{hit}')",
                _REPORT_FAMILY_INSTEAD[family],
            )
        )
    if not items:
        return ""
    return _guidance_block(
        "BROKER QUALITY REPORT WORDING — the drafted report "
        "failed the status-only rule. Rewrite each offending "
        "sentence below into the approved shape; keep every "
        "other sentence, and every figure, exactly as it is:",
        items,
    )


def steer_broker_quality_draft(
    text: str, validation: Validation
) -> str | None:
    """One bounded steered redraft of a failed quality report.

    The report's drafter is code, like the status note's, so
    the steering is applied directly in the report's own
    shapes (see :func:`_steer_wording` for the shared
    discipline): wording families only, an authority-carrying
    sentence comes out, everything else byte for byte.
    Returns the rewrite or ``None`` when steering does not
    apply — the caller revalidates with
    ``guardrails.validate_broker_quality_draft`` and serves
    the draft only if it passes.
    """
    return _steer_wording(text or "", validation, _REPORT_FAMILY_INSTEAD)
