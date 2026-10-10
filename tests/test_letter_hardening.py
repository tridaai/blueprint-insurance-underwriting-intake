"""Round 14: letter-drafting hardening — police the drafts, harder.

The live judge runs kept flagging the same near-miss phrasings in
broker letters: "will proceed to underwriter review", "we will be
in touch shortly". The guardrail that catches them never moved —
what was missing was drafting-side steering. The bounded repair
loop handed the model the failed checks' names (a rule restated),
and the first draft's prompt stated prohibitions without the
approved shapes.

Now the repair feedback is surgical (``letter_repair.py``: the
offending sentence quoted, the family named, the approved rewrite
given), the draft prompt carries the approved status shapes, and
a borderline scan steers one redraft against phrasings the
blocking patterns deliberately do not catch — advisories that
never block, recorded on the run either way. These tests pin the
behaviour with stub backends: a drafter that produces the
borderline phrasing gets repaired into compliant phrasing; a
drafter that never complies keeps its failing verdict.
"""

from __future__ import annotations

from underwriting_agent.graph import run_submission
from underwriting_agent.guardrails import letter_wording_hits, validate_drafts
from underwriting_agent.letter_repair import (
    advisory_repair_guidance,
    letter_advisories,
    letter_repair_guidance,
)
from underwriting_agent.model_backends import MockModelBackend
from underwriting_agent.prompts import DRAFT_SYSTEM_PROMPT
from underwriting_agent.samples import sample_submission_models
from underwriting_agent.schemas import Drafts

SAMPLES = {s.submission_id: s for s in sample_submission_models()}

OFFENDING_SENTENCE = (
    "Your submission will proceed to underwriter review, and we "
    "will be in touch shortly with the outcome."
)
BORDERLINE_SENTENCE = (
    "Your file will now move through review, and you will hear "
    "from us in the coming days."
)


def _letter(context, sentence: str) -> str:
    return (
        f"Dear {context['broker_name']} team,\n\n"
        f"Thank you for submission {context['submission_id']} for "
        f"{context['insured_name']}. {sentence}\n\n"
        "Next step: the file is with the underwriter.\n\n"
        "This letter acknowledges receipt. It is not a quotation, "
        "and nothing in it binds or places coverage.\n"
        "Underwriting Intake Team"
    )


class BorderlineStub:
    """A stub drafter: the first draft carries the phrasing under
    test; a steered redraft returns the house template when
    ``comply`` is set, and parrots the same letter when it is
    not. Every repair feedback it receives is kept for the
    assertions."""

    name = "borderline-stub"
    _model = "stub-1"

    def __init__(self, sentence: str, *, comply: bool = True) -> None:
        self._sentence = sentence
        self._comply = comply
        self.feedback_seen: list[str] = []
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def usage_totals(self):
        return dict(self._usage)

    def reset_usage(self):
        self._usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def draft_documents(self, context):
        self._usage["calls"] += 1
        subject, _mock_letter, packet = MockModelBackend().draft_documents(
            context
        )
        feedback = context.get("repair_feedback") or ""
        if feedback:
            self.feedback_seen.append(feedback)
            if self._comply:
                return MockModelBackend().draft_documents(context)
            return subject, _letter(context, self._sentence), packet
        return subject, _letter(context, self._sentence), packet


# ---------------------------------------------------------------------------
# The blocking family: steered repair lands the compliant rewrite
# ---------------------------------------------------------------------------


def test_blocking_borderline_phrasing_is_repaired_into_compliance():
    stub = BorderlineStub(OFFENDING_SENTENCE)
    result = run_submission(SAMPLES["SUB-1001"], backend=stub)

    # The run ends valid, and the repair is on the record.
    assert result.validation.passed
    assert result.repair_attempted and result.repaired
    assert result.original_validation is not None
    assert not result.original_validation.passed
    failed = {
        c.name for c in result.original_validation.checks if not c.passed
    }
    assert "no_forward_motion_in_letter" in failed

    # The shipped letter is clean under the guardrail's own scan.
    hits = letter_wording_hits(result.drafts.broker_letter)
    assert all(hit is None for hit in hits.values())
    assert OFFENDING_SENTENCE not in result.drafts.broker_letter

    # The steering the stub received was surgical: the offending
    # sentence quoted verbatim, and the approved shapes named —
    # not just the failed checks restated.
    assert stub.feedback_seen, "the repair loop never redrafted"
    feedback = stub.feedback_seen[0]
    assert "LETTER WORDING" in feedback
    assert OFFENDING_SENTENCE in feedback
    assert "records the triage for underwriter review" in feedback
    assert "update you when the review reaches its next step" in feedback


def test_a_drafter_that_never_complies_keeps_its_failing_verdict():
    stub = BorderlineStub(OFFENDING_SENTENCE, comply=False)
    result = run_submission(SAMPLES["SUB-1001"], backend=stub)
    # The gate did not move to meet the draft: still failed, still
    # unapprovable, still awaiting the underwriter.
    assert not result.validation.passed
    assert result.repair_attempted and not result.repaired
    assert result.status == "awaiting_underwriter"


# ---------------------------------------------------------------------------
# The advisory family: borderline phrasing steers a redraft, never a block
# ---------------------------------------------------------------------------


def test_borderline_phrasing_passes_the_gate_but_is_repaired_anyway():
    stub = BorderlineStub(BORDERLINE_SENTENCE)
    result = run_submission(SAMPLES["SUB-1001"], backend=stub)

    # Nothing ever failed: no blocking repair was attempted, and
    # the verdict was a pass from the start.
    assert result.validation.passed
    assert not result.repair_attempted
    assert result.original_validation is None

    # The advisories were found, recorded, and steered against.
    families = {adv["family"] for adv in result.letter_advisories}
    assert families == {"forward_motion", "soft_time"}
    phrases = {adv["phrase"].lower() for adv in result.letter_advisories}
    assert any("move through" in p for p in phrases)
    assert any("hear from us" in p for p in phrases)
    assert any("coming days" in p for p in phrases)
    assert result.advisory_repaired

    # The shipped letter is advisory-clean, and the steering the
    # stub saw named the borderline sentence and its rewrite.
    assert letter_advisories(result.drafts.broker_letter) == []
    assert BORDERLINE_SENTENCE not in result.drafts.broker_letter
    assert stub.feedback_seen
    assert "BORDERLINE WORDING" in stub.feedback_seen[-1]
    assert BORDERLINE_SENTENCE in stub.feedback_seen[-1]


def test_an_unrepaired_advisory_is_recorded_not_blocked():
    stub = BorderlineStub(BORDERLINE_SENTENCE, comply=False)
    result = run_submission(SAMPLES["SUB-1001"], backend=stub)
    # The draft still passes — advisories are not a gate — but the
    # record says the phrasing was seen and not repaired away.
    assert result.validation.passed
    assert result.letter_advisories
    assert not result.advisory_repaired
    # The borderline letter itself is what ships — the advisory
    # steered, the drafter declined, the gate never moved.
    assert BORDERLINE_SENTENCE in result.drafts.broker_letter


def test_advisory_repair_honours_the_repair_switch(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_REPAIR", "off")
    stub = BorderlineStub(BORDERLINE_SENTENCE)
    result = run_submission(SAMPLES["SUB-1001"], backend=stub)
    assert result.validation.passed
    assert result.letter_advisories  # still seen, still recorded
    assert not result.advisory_repaired
    assert stub.feedback_seen == []  # but no redraft was steered


# ---------------------------------------------------------------------------
# The steering instruments themselves
# ---------------------------------------------------------------------------


def test_the_house_template_letter_carries_no_advisories():
    for sample_id in ("SUB-1001", "SUB-1002", "SUB-1013"):
        result = run_submission(SAMPLES[sample_id], backend=MockModelBackend())
        assert result.letter_advisories == []
        assert result.validation.passed


def test_wording_hits_name_every_family():
    letter = (
        "we will proceed with this file within standard authority; "
        "it will proceed to underwriter review and we will be in "
        "touch shortly."
    )
    hits = letter_wording_hits(letter)
    assert hits["forward_motion"] == "we will proceed"
    assert hits["review_commitment"] == "will proceed to underwriter review"
    assert hits["soft_time"] == "in touch shortly"
    assert hits["authority"] == "standard authority"


def test_guidance_is_empty_when_the_letter_check_passed():
    submission = SAMPLES["SUB-1001"]
    drafts = Drafts(
        subject="Submission SUB-1001",
        broker_letter=(
            "Submission SUB-1001 is under review. Next step: an "
            "underwriter will review it and decide."
        ),
        underwriter_packet="packet",
    )
    validation = validate_drafts(submission, drafts, ["UW-CYB-01"])
    assert validation.passed
    assert letter_repair_guidance(validation, drafts) == ""
    assert advisory_repair_guidance([]) == ""


def test_guidance_quotes_the_sentence_and_names_the_shape():
    submission = SAMPLES["SUB-1001"]
    drafts = Drafts(
        subject="Submission SUB-1001",
        broker_letter=(
            "Dear broker team. " + OFFENDING_SENTENCE + " Next step: review."
        ),
        underwriter_packet="packet",
    )
    validation = validate_drafts(submission, drafts, ["UW-CYB-01"])
    assert not validation.passed
    guidance = letter_repair_guidance(validation, drafts)
    assert OFFENDING_SENTENCE in guidance
    assert "will proceed to underwriter review" in guidance  # the matched hit
    assert "records the triage for underwriter review" in guidance


def test_advisories_see_what_the_blocking_patterns_deliberately_miss():
    letter = f"Dear broker team. {BORDERLINE_SENTENCE} Next step: review."
    hits = letter_wording_hits(letter)
    assert all(hit is None for hit in hits.values())  # the gate passes it
    advisories = letter_advisories(letter)
    assert {a["family"] for a in advisories} == {"forward_motion", "soft_time"}
    assert all(a["sentence"] == BORDERLINE_SENTENCE for a in advisories)
    guidance = advisory_repair_guidance(advisories)
    assert BORDERLINE_SENTENCE in guidance
    assert "update you when the review reaches its next step" in guidance


def test_draft_prompt_carries_the_approved_status_shapes():
    prompt = " ".join(DRAFT_SYSTEM_PROMPT.split())
    assert "approved shapes" in prompt
    assert "being referred to a senior underwriter" in prompt
    assert "you will hear from us" in prompt
    assert "in the coming days" in prompt
