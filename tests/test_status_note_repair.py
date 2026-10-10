"""Round 15: steered repair for the drafted status note.

The broker letter's repair steering (round 14) turned a failed
letter check into surgical feedback — the offending sentence
quoted, its family named, the approved shape given — and the
draft shipped only when the redraft passed the same gate. The
status note was held to the same families by
``validate_status_note`` but had no steering: a note whose
composition interpolated record prose carrying a forbidden
phrasing (a gap source saying "we will proceed with the file")
was refused outright, and the underwriter got no note at all.

Now a wording-only failure earns one steered redraft: the
offending sentence is rewritten into the family's approved
note shape (an authority-carrying sentence comes out), the
rewrite is revalidated, and the note is stored with the repair
on its record — the original validation and the guidance that
steered it. Anything else failing — bind language, a premium
figure, a promise, PII — is still refused outright: steering
is for wording, refusal is for substance.
"""

from __future__ import annotations

import copy

import pytest

from underwriting_agent.guardrails import validate_status_note
from underwriting_agent.letter_repair import (
    status_note_repair_guidance,
    steer_status_note,
)
from underwriting_agent.samples import load_sample_submissions
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s["submission_id"]: s for s in load_sample_submissions()}

CLEAN_NOTE = (
    "INTERNAL DRAFT — underwriter status note for submission SUB-1\n"
    "Status: awaiting underwriter decision. Submitted 2026-08-01; "
    "70 day(s) in the queue.\n"
    "Next step: underwriter review and decision."
)


def _note_with(sentence: str) -> str:
    return (
        "INTERNAL DRAFT — underwriter status note for submission SUB-1\n"
        "Status: awaiting underwriter decision. Submitted 2026-08-01; "
        "70 day(s) in the queue.\n"
        f"Outstanding: readiness gaps — {sentence}\n"
        "Next step: underwriter review and decision."
    )


# ---------------------------------------------------------------------------
# The guidance: the note's twin of the letter's steering
# ---------------------------------------------------------------------------


def test_guidance_is_empty_for_a_passing_note():
    validation = validate_status_note(CLEAN_NOTE, "SUB-1")
    assert validation.passed
    assert status_note_repair_guidance(validation, CLEAN_NOTE) == ""


@pytest.mark.parametrize(
    "sentence,family",
    [
        ("We will proceed with this file now.", "forward-motion"),
        ("The file will proceed to underwriter review.", "review commitment"),
        ("We will be in touch shortly.", "time commitment"),
        ("The case sits within our standard authority.", "authority"),
    ],
)
def test_guidance_quotes_the_sentence_and_names_the_family(sentence, family):
    note = _note_with(sentence)
    validation = validate_status_note(note, "SUB-1")
    assert not validation.passed
    guidance = status_note_repair_guidance(validation, note)
    assert "STATUS NOTE WORDING" in guidance
    assert sentence in guidance  # the offending sentence, quoted
    assert family in guidance  # its family, in the guardrail's terms
    assert "Rewrite as:" in guidance  # and the approved shape


def test_guidance_is_empty_when_only_substance_fails():
    note = _note_with("The premium is $12,000 for this file.")
    validation = validate_status_note(note, "SUB-1")
    assert not validation.passed
    assert status_note_repair_guidance(validation, note) == ""


# ---------------------------------------------------------------------------
# The steer: one bounded, deterministic redraft
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "We will proceed with this file now.",
        "The file will proceed to underwriter review.",
        "We will be in touch shortly.",
    ],
)
def test_steer_rewrites_the_offending_sentence_into_the_approved_shape(sentence):
    note = _note_with(sentence)
    validation = validate_status_note(note, "SUB-1")
    steered = steer_status_note(note, validation)
    assert steered is not None
    assert sentence not in steered
    # Every other line survives byte for byte.
    for line in CLEAN_NOTE.split("\n"):
        assert line in steered
    # And the rewrite passes the same gate that refused the draft.
    assert validate_status_note(steered, "SUB-1").passed


def test_steer_removes_an_authority_sentence_outright():
    note = _note_with("The case sits within our standard authority.")
    validation = validate_status_note(note, "SUB-1")
    steered = steer_status_note(note, validation)
    assert steered is not None
    assert "authority" not in steered
    assert validate_status_note(steered, "SUB-1").passed


def test_steer_refuses_to_launder_substance_failures():
    # A premium figure beside a wording hit: steering is for
    # wording only — the note is refused, not repaired around
    # the figure.
    note = _note_with(
        "We will proceed with the file; the premium is $12,000."
    )
    validation = validate_status_note(note, "SUB-1")
    failed = {c.name for c in validation.checks if not c.passed}
    assert failed == {"no_forward_motion_in_letter", "no_firm_premium"}
    assert steer_status_note(note, validation) is None


def test_steer_does_not_apply_to_a_passing_note():
    validation = validate_status_note(CLEAN_NOTE, "SUB-1")
    assert steer_status_note(CLEAN_NOTE, validation) is None


# ---------------------------------------------------------------------------
# The service: refusal becomes the last resort, not the only one
# ---------------------------------------------------------------------------


def make_service() -> SubmissionService:
    return SubmissionService(store=InMemoryStore(), checkpointer=False)


def old_submission(sid: str = "SUB-9001") -> dict:
    sub = copy.deepcopy(SAMPLES["SUB-1001"])
    sub["submission_id"] = sid
    sub["submitted_at"] = "2026-08-01"
    return sub


def _poison_gaps(service: SubmissionService, gap_sources: list[str]) -> None:
    """Route the note's composition through the gap-source line
    carrying the given prose — the interpolation a real record's
    extracted text can carry into a drafted note."""
    record = service._get_store().get("SUB-9001")
    record.result.facts.missing_elements = []
    record.result.facts.gap_sources = gap_sources
    service._get_store().save(record)


def test_a_wording_failure_is_steered_and_stored_with_its_repair():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service,
        ["Broker advised we will proceed with the file once the audits land"],
    )
    note = service.draft_status_note("SUB-9001")
    assert note["repair_attempted"] is True
    assert note["repaired"] is True
    assert note["validation"]["passed"] is True
    original_failed = {
        c["name"] for c in note["original_validation"]["checks"] if not c["passed"]
    }
    assert original_failed == {"no_forward_motion_in_letter"}
    assert "will proceed" not in note["text"]
    assert "STATUS NOTE WORDING" in note["repair_guidance"]
    # The steered note is the stored note, outstanding snapshot
    # and all — the follow-through machinery reads it as usual.
    assert service.status_note("SUB-9001") == note
    assert note["outstanding"]["gap_sources"] == [
        "Broker advised we will proceed with the file once the audits land"
    ]


def test_a_clean_note_carries_no_repair_record():
    service = make_service()
    service.analyze(old_submission())
    note = service.draft_status_note("SUB-9001")
    assert "repair_attempted" not in note
    assert "original_validation" not in note


def test_a_substance_failure_is_still_refused_outright():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(service, ["Broker quoted the premium at $12,000 for this file"])
    with pytest.raises(ValueError, match="not stored"):
        service.draft_status_note("SUB-9001")
    assert service.status_note("SUB-9001") is None


def test_a_mixed_failure_is_still_refused_outright():
    service = make_service()
    service.analyze(old_submission())
    _poison_gaps(
        service,
        [
            "We will proceed with the file; "
            "broker quoted the premium at $12,000"
        ],
    )
    with pytest.raises(ValueError, match="not stored"):
        service.draft_status_note("SUB-9001")
    assert service.status_note("SUB-9001") is None
