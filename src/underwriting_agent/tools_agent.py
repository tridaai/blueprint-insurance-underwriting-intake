"""Agent tools for the diagnosis step (provider mode).

In provider mode the diagnosis is *agentic*: instead of one composed
call over pre-assembled facts, the model runs a bounded tool loop —
it may pull more guideline context, the computed submission facts, or
the insured's and broker's history from prior submissions before it
composes the risk summary. The loop itself lives in the backends
(``model_backends.py``, per-provider tool formats); this module owns
what the tools ARE: their schemas (one neutral definition both
provider formats render from) and their dispatch.

Every tool here is a read over data the pipeline already holds — the
guideline corpus behind the retriever interface, the store's history
entries, and the computed facts. No tool sends, files, mutates, or
decides anything; the graph's human-gate rule is untouched.

The bound: ``DIAGNOSIS_MAX_TOOL_CALLS`` (default 4, hard cap 6) caps
how many tool calls one diagnosis may make. When the budget runs out
the model is told to compose with what it has.
"""

from __future__ import annotations

from .config import env_int, load_dotenv
from .schemas import SubmissionInput
from .store import broker_summary, format_outcome_counts

DEFAULT_MAX_TOOL_CALLS = 4
HARD_MAX_TOOL_CALLS = 6

# Neutral tool definitions: name, description, JSON-schema parameters.
# The Anthropic backend renders these as {name, description,
# input_schema}; the OpenAI backend as {type: "function", function:
# {name, description, parameters}}. One definition, two wire formats.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_guidelines",
        "description": (
            "Search the underwriting guideline corpus with your own query "
            "wording and get the top matching guidelines with their IDs. "
            "Use when the guidelines already provided look incomplete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in underwriting wording."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "submission_facts",
        "description": (
            "The computed facts for THIS submission: triage outcome and "
            "risk tier, completeness score, data-readiness flag, loss-run "
            "summary, referral triggers, and extraction cross-check "
            "discrepancies. Always prefer these over assumptions."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "insured_history",
        "description": (
            "Prior submissions for an insured from the stored history: "
            "count and the outcomes they received (in_appetite, "
            "borderline, out_of_appetite, incomplete)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"insured_name": {"type": "string"}},
            "required": ["insured_name"],
        },
    },
    {
        "name": "broker_history",
        "description": (
            "Prior submissions from a broker from the stored history: "
            "total count and outcome counts — a broker whose submissions "
            "routinely arrive incomplete is itself a risk signal."
        ),
        "parameters": {
            "type": "object",
            "properties": {"broker_name": {"type": "string"}},
            "required": ["broker_name"],
        },
    },
]


def diagnosis_max_tool_calls() -> int:
    """The tool-call budget for one diagnosis (env, read at run time)."""
    load_dotenv()
    return min(max(env_int("DIAGNOSIS_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS), 0), HARD_MAX_TOOL_CALLS)


class DiagnosisToolBox:
    """Dispatches the diagnosis tools over the data one run already holds.

    ``dispatch(name, args)`` returns ``(result_text, summary)``: the
    text fed back to the model, and a one-line summary for the trace.
    Unknown tools and bad arguments raise — the backend loop converts a
    raise into an error tool-result and the model composes without it,
    so a broken tool degrades the diagnosis, never the run.
    """

    def __init__(
        self,
        *,
        submission: SubmissionInput,
        triage: dict,
        facts: dict,
        discrepancies: list[str] | None = None,
        retriever=None,
        priors: list[dict] | None = None,
    ) -> None:
        self._submission = submission
        self._triage = triage
        self._facts = facts
        self._discrepancies = discrepancies or []
        self._retriever = retriever
        self._priors = priors or []

    def dispatch(self, name: str, args: dict) -> tuple[str, str]:
        handler = {
            "search_guidelines": self._search_guidelines,
            "submission_facts": self._submission_facts,
            "insured_history": self._insured_history,
            "broker_history": self._broker_history,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return handler(args or {})

    # -- the tools ------------------------------------------------------

    def _search_guidelines(self, args: dict) -> tuple[str, str]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("search_guidelines needs a non-empty 'query'")
        if self._retriever is None:
            raise ValueError("no retriever is configured for this run")
        results = self._retriever.retrieve(query, top_k=3)
        if not results:
            return "No guidelines matched that query.", f"search_guidelines({query!r}) -> 0 guidelines"
        text = "\n".join(f"[{r.guideline_id}] {r.title}: {r.text}" for r in results)
        ids = ", ".join(r.guideline_id for r in results)
        return text, f"search_guidelines({query!r}) -> {len(results)} guidelines: {ids}"

    def _submission_facts(self, args: dict) -> tuple[str, str]:
        t, f = self._triage, self._facts
        loss = f.get("loss_summary") or {}
        lines = [
            f"triage: {t.get('outcome')} (risk tier {t.get('risk_tier')}, confidence {t.get('confidence')})",
            f"referral triggers: {'; '.join(t.get('referral_triggers') or []) or 'none'}",
            f"completeness: {f.get('completeness_score')} · readiness: {f.get('readiness')}",
            f"missing elements: {', '.join(f.get('missing_elements') or []) or 'none'}",
            (
                f"loss history: {loss.get('claim_count', 0)} claim(s), total incurred "
                f"{loss.get('total_incurred', 0)}, largest claim {loss.get('largest_claim', 0)}"
            ),
            f"submission age: {f.get('submission_age_days')} day(s)",
        ]
        lines += [f"extraction discrepancy: {d}" for d in self._discrepancies]
        summary = (
            f"submission_facts -> {t.get('outcome')}/{t.get('risk_tier')}, "
            f"completeness={f.get('completeness_score')}, readiness={f.get('readiness')}"
        )
        return "\n".join(lines), summary

    def _insured_history(self, args: dict) -> tuple[str, str]:
        insured = str(args.get("insured_name") or self._submission.insured_name)
        hits = [e for e in self._priors if insured and e.get("insured_name") == insured]
        if not hits:
            return (
                f"No prior submissions for insured {insured!r} in the stored history.",
                f"insured_history({insured!r}) -> no prior submissions",
            )
        outcomes = ", ".join(e["outcome"] for e in hits[:5])
        return (
            f"{len(hits)} prior submission(s) for insured {insured!r} in the stored "
            f"history (most recent outcomes: {outcomes}).",
            f"insured_history({insured!r}) -> {len(hits)} prior submission(s): {outcomes}",
        )

    def _broker_history(self, args: dict) -> tuple[str, str]:
        broker = str(args.get("broker_name") or self._submission.broker_name)
        data = broker_summary(self._priors, broker)
        total = data["broker_count"]
        if total == 0:
            return (
                f"No prior submissions from broker {broker!r} in the stored history.",
                f"broker_history({broker!r}) -> no prior submissions",
            )
        breakdown = format_outcome_counts(data["broker_outcome_counts"])
        return (
            f"{total} prior submission(s) from broker {broker!r} in the stored "
            f"history; outcomes: {breakdown}.",
            f"broker_history({broker!r}) -> {total} prior submission(s), outcomes: {breakdown}",
        )
