# Architecture — Insurance Underwriting Submission Intake Agent

This document explains how the blueprint is built and why: the graph,
the split between code and model, the cross-check policy, the
guardrails, the checkpointed gate, and the path to production. It is
written for the engineer who will take this core and adapt it.

> **Provenance.** The structural approach of this blueprint — NIM
> endpoint profiles, an eval harness in the repo, guardrails as a
> first-class component, observability, and decomposable components
> behind interfaces — follows the conventions NVIDIA's blueprint
> programme popularised. The implementation and content are original
> Trida AI work: every line of code, prose, and sample data in this
> repository was written for this blueprint, and nothing is copied
> from NVIDIA (or any other) repositories, regardless of what their
> licences would allow.

---

## 1. Problem framing

Intake is the highest-volume, most document-bound step in specialty
underwriting, and the worst place to spend underwriter judgment:
before an underwriter can decide anything, someone must re-key the
submission, establish what is missing, read the loss runs, check the
class against appetite, apply the authority matrix, and assemble a
packet. The decisions themselves — proceed, refer, ask for more, or
decline — belong to licensed humans and, in this design, stay there.

The agent therefore has a narrow, explicit mandate: **do all of the
assembly, draft both documents, and stop.** Its failure modes are
asymmetric in a way that shapes the whole design: a triage that is
wrong costs a review cycle; a letter that quotes a premium or implies
coverage is bound creates legal exposure. The architecture spends
most of its machinery on the second class of failure (guardrails,
verification, independent review, the gate) and treats triage quality
as an eval-gated, human-supervised concern.

## 2. System overview

```
            ┌────────────────────── LangGraph pipeline ──────────────────────┐
submission →│ extract → ingest → screen → classify → ┌ retrieve ┐            │
            │                                          ├ discrepancies ┤→       │
            │                                          └ history ┘            │
            │ diagnose → options → draft → verify → review → validate       │
            │ → gate → approval_gate (interrupt)                              │
            └──────────────────────────────────────────────────────────────┘
                                        │ decision (service layer)
                                 store (PostgreSQL) ──→ optional webhook
```

Six replaceable collaborators sit behind protocols (`ports.py`):
the **model backend**, the **retriever**, the **store**, the
**checkpointer**, the **object store**, and an **event sink** for
streaming. `wiring.py` is the composition root; everything else
programs against the seams.

### Production topology

The API process is stateless; every durable thing lives in one
database and one bucket. This is what the compose stack runs, and
what a managed-services deployment maps onto directly:

```
                        ┌─────────────────────────────────────────────┐
  underwriter / broker  │  api  (stateless, N replicas)               │
  systems ─── HTTPS ───▶│  FastAPI + LangGraph pipeline               │
                        │  JSON logs · X-Request-ID · /readiness      │
                        └──────┬───────────────┬──────────────┬───────┘
                               │               │              │
                     DATABASE_URL        S3_*          provider env
                               ▼               ▼              ▼
                  ┌──────────────────┐ ┌───────────────┐ ┌────────────┐
                  │ PostgreSQL       │ │ S3-compatible │ │ model      │
                  │ + pgvector       │ │ bucket        │ │ provider / │
                  │ · approval store │ │ (MinIO in     │ │ Ollama /   │
                  │ · checkpoints    │ │  compose)     │ │ NVIDIA NIM │
                  │ · guideline      │ │ submission    │ │            │
                  │   embeddings     │ │ documents     │ │            │
                  └──────────────────┘ └───────────────┘ └────────────┘
```

Schema ships as numbered SQL files in `migrations/` and is applied
at startup by a small runner (`db.py`, recorded in
`schema_migrations`); the checkpointer's tables are created by the
official Postgres saver's own `setup()`. The SQLite and in-memory
store/checkpointer implementations that remain in the codebase are
**test doubles** — the hermetic suite injects them explicitly, and a
run without `DATABASE_URL` uses the in-memory pair and persists
nothing. Local files are not a runtime mode.

The split of responsibilities is the architecture:

- **Code computes.** Completeness, readiness, the loss summary,
  triage signals, option scores, guardrail checks. Deterministic,
  testable, identical every run.
- **The model does language work.** Extraction, an independent
  triage opinion, risk prose, option wording, both drafts,
  self-critique, review critique. Each capability is optional and
  discovered with `getattr` — absence engages a deterministic path.
- **Humans decide.** The graph has no node after the gate.

## 3. Graph and state design

State is a `TypedDict` of plain dicts (Pydantic models dumped at node
boundaries, validated on the way in) so checkpoints serialise
cleanly. Nodes:

| Node | Work | Model? |
| --- | --- | --- |
| `extract` | per-document field extraction + cross-check | provider mode |
| `ingest` | computed facts (`facts.py`) | never |
| `screen` | injection screening of broker text | never |
| `classify` | rules triage + LLM cross-check (`crosscheck.py`) | provider mode |
| `retrieve` | guideline retrieval (fan-out branch) | never |
| `discrepancies` | extraction discrepancy lines (fan-out branch) | never |
| `history` | memory + feedback evidence lines, renewal diff (fan-out branch) | never |
| `diagnose` | risk diagnosis; bounded tool loop in provider mode | provider mode |
| `options` | proposals (model wording) + deterministic scoring | optional |
| `draft` | broker letter + underwriter packet | always (template or LLM) |
| `verify` | self-verification of the drafts | provider mode |
| `review` | independent review of the drafts | provider mode |
| `validate` | guardrails + bounded repair loop | never (repair redrafts) |
| `gate` | autonomy recommendation + information request | never |
| `approval_gate` | `interrupt()` — pause for the human decision | never |

The three evidence branches fan out from `classify` because they are
independent reads over state the earlier nodes produced; `diagnose`
joins them. The joined state is identical to a sequential run — the
fan-out is an execution detail, not a behaviour change.

Every node is wrapped with timing and an event emission; node
durations land in the result's trace (`duration_ms`), and the same
events stream over SSE from the API.

Two cross-cutting wrappers sit between the graph and the backend:

- **Resilience** (`resilience.py`): per-call timeout ceiling
  (`NODE_TIMEOUT_SECONDS`), retries only for the idempotent language
  steps and only on `ProviderError` (max 2 attempts). Drafting and
  the gate never retry.
- **Token budget** (`RUN_TOKEN_BUDGET`): a per-run guard that, once
  cumulative provider usage crosses the cap, makes the optional
  capabilities disappear (the deterministic paths engage) and
  degrades drafting to the template backend. The run completes; it
  never fails for budget reasons; telemetry reports the accounting.

## 4. Key design decisions and trade-offs

### 4.1 Computed facts are code, never model output

Completeness, readiness, and the loss summary decide triage, gate
autonomy, and appear in the packet — so they are computed in
`facts.py` from the submission and the extraction cross-check, with
unit tests pinning the math. The model sees these numbers as *input*
and may cite them; it cannot change them. This is the single most
important property of the system: every number an underwriter reads
has a deterministic provenance.

### 4.2 The triage cross-check, and its one adoption rule

Rules and the LLM triage independently; `crosscheck.py` resolves:

1. **Agree** → the rule result stands.
2. **Disagree, rules strong** → the rule result stands
   (`rules_authoritative`); the disagreement is surfaced.
3. **Disagree, rules weak, LLM confident** → the LLM triage is
   adopted (`llm_adopted`) — the case rules are known to miss: a risk
   described in phrasing no rule covers. The adopted result is
   flagged in its signals and rationale so the underwriter sees
   exactly why the triage is not the rule output.

"Rules weak" = the rules' own confidence below 0.85; "LLM confident"
= ≥ 0.85. Anything else (no provider, provider error, unusable
reply) resolves to the rules alone (`rules_only`). Adoption never
touches computed facts.

### 4.3 Guardrails as code, with a bounded repair loop

The guardrail checks in `guardrails.py` are the product's promise,
implemented as deterministic tests over the draft text:

- `no_bind_language` — affirmative bind phrasing ("coverage is
  bound", "must be in force", "you are covered", …), checked against
  a negation window so the *required disclaimer* ("no coverage is
  bound") passes and the broker's relayed demand does not.
- `no_firm_premium` — a premium word near a figure, or a quoted/firm
  premium phrase. (A requested *limit* figure is the broker's ask and
  is explicitly not flagged.)
- `no_prohibited_promises` — outcome promises ("we guarantee
  approval") and time commitments ("your quote will arrive within 24
  hours"), the latter being the same failure with the promiser
  redacted.
- `no_forward_motion_in_letter` — the one letter-scoped check.
  Forward-motion commitments ("we will proceed", "we will move this
  submission forward", "will proceed to quote preparation") and
  internal authority vocabulary ("standard authority", authority
  thresholds and figures) fail in the broker letter, which reports
  status only; the internal packet is not scanned, because those
  facts belong there. This began as prompt discipline after an
  LLM-judge pack flagged forward-motion phrasing in provider-mode
  letters; it is code now, with golden eval cases pinning it. The
  pack's second run extended the family twice more — review
  commitments ("will proceed to underwriter review", "is now under
  senior underwriter review") and soft time commitments ("we will
  be in touch shortly") — each now a pattern family in the same
  check, with its own golden cases (RC1/ST1/RC2). The discipline is
  one-directional: the drafts get policed; the judge never gets
  tuned to pass. The pack's regression gate
  (`evals/judge_gate.py`) diffs each run against the pinned
  finding families, and every run appends its summary to
  `evals/judge_history.jsonl` — the pack prints the per-family
  trend across runs, so whether the fixes are holding is a dated
  series, not the latest run's impression.
- `no_pii_in_draft` — SSN, Luhn-valid card numbers, passport
  patterns.
- Structural: the drafts must reference the submission id, cite at
  least one retrieved guideline, and state the next step.

When validation fails, the **repair loop** redrafts with the failed
checks fed back (`GUARDRAIL_REPAIR`, default on; attempts capped at
3, default 1) and re-validates. The original validation is preserved
on the result — a repaired run shows its first failure. If repair
cannot fix it (the template backend on SUB-1011 cannot), the draft
stays failed and the service layer **refuses approval** (API 422).
Rejecting is never blocked: the safe direction is always available.

### 4.4 Injection screening is at ingest, and it is subtraction

Broker text is untrusted. `screening.py` pattern-matches
instruction-like sentences (system impersonation, ignore-instructions,
policy override, directed decisions, directed binding, directed
promises), records them as flags the underwriter sees, and *removes
the sentences* from the text prompts are built from; surviving text
is fenced as data. Reported speech is deliberately not flagged — a
broker reporting a client's demand is a fact about the submission,
and catching it is the guardrails' job (see SUB-1011). The two
layers have different jobs on purpose.

### 4.5 The diagnosis is evidence-bound, and agentic only in provider mode

`diagnosis.py` assembles the evidence list deterministically in both
modes: computed-facts lines, discrepancy lines, triage signals,
guideline citations, memory lines, feedback lines. The LLM (single
call, or the bounded tool loop with `search_guidelines` /
`submission_facts` / `insured_history` / `broker_history`) writes
prose *over* that evidence; the evidence list itself is attached
verbatim. An unusable or failed LLM reply degrades to a template
diagnosis built from the same evidence, with the reason in `note`.
Tool calls are capped (`DIAGNOSIS_MAX_TOOL_CALLS`, default 4, hard
cap 6) and logged into the trace.

### 4.6 Options are scored by code; the model only words them

The four option kinds are a closed vocabulary. The model may propose
titles and descriptions; kinds are validated and **every score is
computed** (`options.score_option`: outcome fit + readiness fit −
tier and loss-frequency penalties for proceeding). The recommended
option is the argmax. This keeps the "what does the agent recommend"
question auditable: the score formula is in the repo, with the
weights an adopting team can tune.

### 4.7 Two critics, different checks

Self-verification (`verify.py`) asks whether the drafts are grounded
in the facts. The independent reviewer (`reviewer.py`) asks a
different question — is the *packet* complete (summary, citations, a
named recommendation, next step in the letter) — and runs a disjoint
deterministic checklist in every mode plus an LLM critique (its own
persona; `REVIEWER_MODEL` can point it at a different model) in
provider mode. A block verdict flags the run and disqualifies it
from any auto-triage recommendation, but the decision stays human.

### 4.8 The gate is checkpointed; the store is the record

With checkpoints on (default), the graph *pauses* at
`approval_gate` via LangGraph `interrupt()`, its state persisted
under the submission id. In the production configuration the
checkpointer is the **official LangGraph Postgres saver**
(`langgraph-checkpoint-postgres`) over a psycopg 3 pool, in the same
PostgreSQL as the store — so one database holds decisions, graph
state, and (via pgvector, Section 2) the guideline embeddings.
Approve/reject in the service layer record the decision in the
store, then resume the thread with the decision payload. The store
is the decision of record; the checkpointer holds graph state. A
missing or finished thread never blocks a decision — resume is
best-effort by design. The stdlib-SQLite saver implementation is
retained for the test suite, which exercises the same
pause/resume contract without a server.

### 4.9 Memory and the feedback loop

Analysed submissions accumulate in the store. A new analysis pulls
priors for the same insured and broker into the diagnosis evidence
(counts and recent outcomes — never raw text from other files), and
**decisions with reasons** become feedback lines ("underwriter
feedback: last decision for this insured was rejected — reason:
'…'"). Human oversight thus compounds: the reasons underwriters
write at the gate teach every later run for that insured or broker.

### 4.10 Autonomy is a printed recommendation, never a behaviour

`autonomy.py` evaluates a deliberately narrow eligibility policy
(in-appetite, low/moderate tier, readiness `ready`, guardrails clean
on the first draft, no cross-check disagreement, no reviewer block)
and the result carries the recommendation plus the reasons. Nothing
acts on it. It exists so an adopting team can see, case by case,
what a future autonomy policy would have to be comfortable with.

## 5. Data model

Pydantic throughout (`schemas.py`). The boundaries that matter:

- `SubmissionInput` — the intake contract (Section 5.1).
- `SubmissionFacts` — computed facts; the only numbers in the system.
- `Triage` / `CrossCheck` — the decision and its provenance.
- `Diagnosis`, `RecoveryOption`, `Drafts` — the model's work
  products.
- `Validation` — the guardrail verdicts (every check blocking).
- `AgentResult` — the full run: every intermediate, the trace with
  per-node durations, telemetry (backend, model, calls, tokens,
  estimated cost, budget status), and — after the gate — the
  decision fields.

### 5.1 Intake contract and normalisation

Real intake is messy, so normalisation happens at the schema
boundary: document-type aliases fold to a six-word vocabulary
(`acord_form, loss_run, financials, sov, broker_email, supplemental`;
unknown types are kept and flagged, never dropped), numeric document
fields coerce to strings, and money strings ("$45,000") coerce to
numbers on the submission and its loss records. Line aliases fold too
("D&O" → `directors_and_officers`, "E&O" →
`professional_liability`).

## 6. Failure modes

| Failure | Behaviour |
| --- | --- |
| Provider down / timeout | Translated `ProviderError` (API 502); safe nodes retry once; cross-check / diagnosis / verify / review degrade to deterministic paths |
| Provider reply unparseable | Treated as "no result" — the deterministic path runs; the degradation is noted on the artifact it touched |
| Draft fails guardrails | Bounded repair; if still failing, approval is refused (422) and rejection remains available |
| Injection in broker text | Flagged, sentence removed, triage proceeds on facts; the flag is visible at the gate |
| Token budget exceeded | Remaining provider steps degrade to templates; the run completes |
| Bad submission in a batch | That item carries the error; the batch completes |
| Webhook down after approval | Approval stands; `dispatch_status=failed` records the follow-up |
| Checkpoint thread missing at decision time | Decision records in the store anyway; resume is best-effort |
| Database unreachable | `/readiness` returns 503 so the orchestrator drains the replica; analyses fail loudly rather than falling back to local state |
| Document fetch from object storage fails | The document's text stays empty; completeness scoring surfaces the gap instead of the run crashing |
| Document archive write fails | The analysis stands; that document simply carries no `object_key` on the stored payload |

## 7. Security notes

- **No credentials in the repo.** Keys, `DATABASE_URL`, and bucket
  credentials come from the environment / `.env` (git-ignored);
  construction and provider errors are translated so a key never
  appears in a message. In production these arrive from a secrets
  manager, not a file (Section 9).
- **Untrusted text discipline.** Broker text is screened, fenced,
  and never interpolated into system prompts; the guardrails — not
  the model — police what leaves in a letter.
- **Optional shared-secret API gate** (`API_KEY` → `X-API-Key`). It
  is a demo control, not identity; health and the console stay open
  by design.
- **No external action by construction.** No graph node can send,
  file, quote, or bind. The single opt-in egress is the approval
  webhook, fired only *after* a human decision, and its failure
  never rewrites history. With `ACTION_WEBHOOK_SECRET` set the
  dispatch is HMAC-SHA256-signed (`X-Trida-Signature`) over the
  exact body bytes. Every attempt is written to a delivery ledger
  on the record (timestamp, HTTP status, truncated signature id,
  error, scheduled backoff), and a failed delivery is retryable
  through the service/API — the approval it belongs to is never
  re-opened by a retry.
- **The store is the reporting surface.** Inbox grouping, party
  scorecards, `/metrics`, and the audit export are deterministic
  projections of the same records the gate writes (`insights.py`) —
  no separate analytics pipeline to drift from the record.
- **Renewals diff against the store.** The classify node computes
  the renewal diff (`diffing.py`): the insured's most recent stored
  submission (payload + its computed facts) against the new one,
  field by field. The result carries it (`submission_diff`), the
  diagnosis cites it as evidence, and the packet prints a "Changes
  since last submission" section. A first submission has no diff —
  absence is not a change. The diff is also an input to triage, not
  just a description: new claims clearing a severity floor, a limit
  jump beyond a configured percentage, or a class change become
  referral triggers on the rule triage before the cross-check
  (`diff_referral_triggers` / `apply_diff_triggers`), so the
  cross-check and everything downstream see the referral the change
  earned. The diff also develops the loss run **per claim**
  (`match_loss_development`): claims are matched across the two
  loss runs by claim id, else date + description (one-to-one,
  conservative — an ambiguous match reports as appeared +
  disappeared rather than a wrong pairing), and adverse
  development across matched claims past a floor is its own
  trigger. New-claim detection reads the appeared set, so a claim
  swapped for another is a new claim even at equal counts. The
  prior lookup is keyed **per line**: a renewal diffs against the
  insured's most recent stored submission *of the same line* — for
  a multi-line bundle, that is the matching line of the most recent
  prior bundle (or an earlier single-line submission of that line).
- **Bundles split at the service, not the graph.** A submission
  carrying `lines` is split by `bundles.split_bundle` into one
  ordinary `SubmissionInput` per line (line fields win; bundle
  documents and context are shared) and each runs the unmodified
  graph with no checkpointer. The bundle result is composed in
  code: triage follows the most severe line (outcome, then tier),
  facts take the worst readiness and the per-line gap sources,
  drafts are one packet with per-line sections plus one letter,
  and the combined drafts pass the same verification, review, and
  guardrails as a single-line draft. One record, one gate: the
  store never holds a sub-analysis as a decidable item. The
  bundle's information request is per line as well
  (`clarify.build_bundle_information_request`, composed in code so
  line attribution stays exact): only gapped lines are asked, each
  for its own items, and the result records `blocking_lines` —
  the lines whose missing elements hold the bundle back.
- **Dispatch and trend are projections too.** The dispatch
  operations view (`insights.dispatch_overview`) reads every
  record's delivery ledger as one surface — failures oldest first,
  each with the next backoff from the recorded schedule and
  whether its retry is due — and failures age on it: age from the
  first failed attempt, bucketed like the queue's, with a *stuck*
  flag for failures older than `DISPATCH_STUCK_AGE_HOURS` that
  still have attempts left under `DISPATCH_ATTEMPT_BUDGET`
  (exhausted is a different word, and the view keeps them apart).
  Scorecards carry an `approval_trend` (earlier-half vs
  recent-half approval rate over the group's decision order), so
  a party drifting toward rejections is visible before the flat
  rate moves.
- **The queue has an SLA; the thresholds have a calibration.**
  Queue items are built by one projection (`queue_item_for`) that
  also stamps an SLA state: the budget for the item's risk tier
  (`QUEUE_SLA_DAYS_*`, defaults elevated 2 / high 3 / moderate 5 /
  low 7 days — the budget shrinks as the tier rises, because an
  elevated case needs a senior underwriter soonest and ages
  worst). Breaches are the queue's worklist, worst overrun first.
  A breached case can draft an **underwriter status note**
  (`service.draft_status_note`): composed in code from the record
  and that same queue projection (so note and queue cannot
  disagree), checked by `guardrails.validate_status_note` — the
  broker letter's text families, status-only discipline — and
  stored on the record only if it passes. It is an internal
  draft; the system has no path that sends it. Bundle items
  carry their `blocking_lines` with the wait per line (the
  bundle's own queue age — a line has blocked since intake) and
  the line's missing items, so a one-line strand reads
  differently from a wholly unready submission. The
  diff-trigger thresholds get the same evidence treatment in
  `calibration.py`: stored renewal pairs are rebuilt with the
  service's own per-line prior chain and the trigger computation
  is replayed at candidate values, one knob at a time —
  referrals and family counts per setting against what actually
  fired and what underwriters decided. And the per-pair diffs
  chain into an **insured loss-development timeline**
  (`insights.insured_loss_timeline`): entries per submission
  (per line) plus one event stream per claim — appeared,
  developed, no longer listed. Both surfaces are read-only; the
  timeline reports stored diffs and never recomputes them.
- **Round 8 closes three loops those surfaces opened.** (1) A
  breach is *announced*: `service.notify_sla_breaches` computes
  crossings from the queue projection at call time (invoked, not
  timed) and fires one signed `sla_breach` event per submission
  through the same `_post_webhook` seam approvals use — the
  signing/posting/ledger code exists exactly once, and the
  approval payload now names its event (`"event": "approval"`)
  so a receiver can tell the two apart. The notice lives on the
  record with its own ledger (`sla_breach_attempts`, separate
  from the approval ledger — different event, different retry);
  the recorded notice *is* the dedupe, and a failed firing waits
  for an operator retry that carries the original breach
  snapshot rather than today's age. With no webhook configured
  nothing is recorded, so a later configuration still announces
  standing breaches. (2) The status note is *followed through*:
  drafting snapshots its outstanding items in data
  (`note["outstanding"]`), and a resubmission from the same
  insured runs `insights.status_note_follow_through` against its
  computed facts — the packet gains the section (what was
  outstanding, what arrived, what is still missing) and the
  note's record gains a `follow_through` marking. One function
  feeds both surfaces, so packet and record cannot disagree;
  the note's text is never rewritten. Pre-snapshot notes report
  `not_recorded` instead of pretending to have been answered.
  (3) Calibration *proposes*: `proposals.py` scans the replay
  for one-sided decided evidence (gained referrals all rejected
  / lost referrals all approved; undecided or contradicting
  moved cases draft nothing) and stores proposal documents in
  the new `service_documents` table (migration `0004`; a
  key/value document capability on the Store port). Accepting a
  proposal writes the threshold into a settings document, and
  threshold resolution — environment base, settings override,
  in `proposals.effective_thresholds` — is the single path the
  service passes into every analysis and the calibration replay
  centres on. The trade-off is explicit: thresholds are no
  longer purely deploy-time configuration, so the settings
  record carries its own audit history (proposal, from → to,
  actor, timestamp) and the calibration view names its source.
  The loss arc, finally, also renders *inside* the packet
  (`insights.loss_timeline_packet_lines`): the stored timeline
  with the current diff replayed over its claim states, so the
  cumulative the packet quotes includes this submission.
- **Round 9 extends the proposal machinery to the queue, and
  makes it two-way and time-aware.** (1) SLA budgets calibrate
  like thresholds: `proposals.decided_wait_cases` replays the
  decided records' waits per risk tier, and the one-sided bar is
  the wait's version of the referral bar — a tighter budget
  drafts only when its newly flagged cases were all rejected, a
  looser one only when its un-flagged cases were all approved
  anyway; awaiting cases are excluded by construction (they
  have no wait outcome). A second settings document
  (`queue_sla_budgets`) carries the accepted budgets, and
  `effective_sla_budgets` is the single resolution the queue,
  the breach check, and note drafting read — so the budget a
  case breached under is the budget proposals tune. (2) Drafts
  expire: a proposal's evidence is a store snapshot, and after
  `PROPOSAL_EXPIRY_DAYS` deciding it is refused. Refresh re-runs
  the drafting scan for that one setting, re-based on the values
  in force — renewed in place when the evidence still carries a
  move, withdrawn (with reason) when it does not. The refresh
  audit entry keeps the stale snapshot, so evidence is never
  silently replaced. (3) The follow-through markings leave the
  record: `insights.follow_through_digest` projects every note
  into the inbox, unresolved-and-oldest first; notes no
  resubmission has reached report `awaiting_resubmission` with
  their snapshot still outstanding rather than a blank. (4)
  Settings revert: `_revert_settings_change` restores the latest
  change's prior value as a new history entry naming the undone
  proposal — revert is itself a change, so the audit trail is
  append-only end to end, and reverting a revert is just the
  same rule one level down.
- **Round 10 gives operations one worklist, and settings a full
  way back.** (1) The ops digest (`insights.ops_digest`,
  `GET /ops/digest`): the four rotting-quietly projections —
  `dispatch_overview`'s stuck failures, `queue_aging`'s breaches
  (against the budgets in force), the follow-through digest's
  outstanding entries, and the proposals' expiry state — composed
  into one read with a deterministic headline. It is a pure
  composition of existing projections, so the digest can never
  disagree with the screen each surface came from. (2)
  Follow-through ages: an open loop past
  `FOLLOW_THROUGH_ESCALATION_DAYS` is flagged escalated *in the
  digest itself* (one flag, read by both the inbox and the ops
  digest) — deliberately not an SLA breach: the note and the case
  have different clocks. (3) The refresh-all sweep
  (`proposals.refresh_expired_proposals`) is a loop over the
  per-proposal refresh, never a bulk path around it — each
  draft's audit entry is written by the same code that writes a
  single refresh. (4) Point-in-history restore:
  `_values_as_of` replays the environment base through the
  history to any entry — flat entries and the `changes`-carrying
  entries (restores, and reverts of restores) alike — and the
  restore appends one entry naming its target, so recovery from
  a mis-tuned sequence is one audited decision; `_revert_settings_change`
  learned the `changes` shape, which makes a restore revertible
  by the same rule as everything else.
- **Round 11 makes movement visible, and previews the writes.**
  (1) Digest snapshots: the digest evaporated at every read, so
  `build_digest_snapshot` keeps one on demand — headline, counts,
  and the section membership (entry ids, sorted) as a stored
  document under the taker's name — and `digest_trend` compares
  the kept snapshots: counts as a series, change first→latest and
  previous→latest, and per-section entered/cleared between the
  latest two, so churn and net movement are never conflated.
  Snapshots are operator-taken, never on a timer: a trend reads
  only what someone decided was worth keeping. (2) Escalation
  reaches the worklist: `queue_aging` reads the follow-through
  digest itself and matches escalated notes to queue items by
  insured (the note's record is usually an older, decided one),
  so the item carries the flag and the note's age, and the view
  lists every escalated note with its insured's awaiting ids.
  One escalation definition, computed once, read everywhere.
  (3) Both previews are the same code with the save left out:
  `_restore_plan` computes the restore's target values and
  per-dimension moves for the restore and its preview alike (the
  preview reports a no-op as `would_change: false` instead of
  raising), and `_refresh_scan` runs the refresh's evidence scan
  for the refresh, its single preview, and the sweep preview —
  a previewed outcome is therefore the outcome the write
  produces, pinned by a test that previews, then sweeps, and
  compares the results field for field.
- **Round 12 prices the silence, and gives the budgets a
  calibration of their own.** (1) `sla_budget_calibration`
  replays `decided_wait_cases` the way `trigger_calibration`
  replays renewals: per tier a wait distribution against the
  budget in force — provenance read from the settings history,
  because the document stores the whole budgets map and presence
  proves nothing — and candidate rows carrying the moved set
  against the budget in force, flagged `proposal_ready` exactly
  when `_best_sla_candidate` would draft from it. The report and
  the proposal scan are two reads of one evidence base.
  (2) A breach notice's ledger is a different event from an
  approval ledger, so exhaustion is computed from the notice's
  own attempts (`_exhausted_breach_notices`): budget spent, last
  attempt failed, case still awaiting — aged from the first
  observation, listed apart in the dispatch view and the digest,
  because its remedy is a phone call and merging it into the
  retry list would train operators to click past it.
  (3) A restore entry gains `decision_note`: the audit record
  already answered what/who/when for every other consequential
  act (proposal decisions carry notes), and a restore is the
  most judgement-shaped act of all; the preview echoes the
  draft note because that is where the wording gets decided.
  (4) The digest reads itself against the latest snapshot
  (`latest_snapshot` → a `movement` block + one headline
  sentence): the trend endpoint compares kept snapshots to each
  other, but the operator's first screen is the live digest, so
  that is where since-snapshot movement belongs. Absent a
  snapshot the block is None — the system does not fabricate a
  baseline to look trend-aware.
- **Synthetic data only**, enforced socially (CONTRIBUTING) rather
  than technically — a real deployment must add data governance.

## 8. What to change first (prototyping order)

1. **Your appetite and authority matrix** — `triage.py` /
   `facts.py` required elements, classes, limits, triggers. Add
   golden cases for every rule you change.
2. **Your guidelines** — replace the corpus behind the retriever.
   Bump `GUIDELINES_VERSION` when you do: results are stamped with
   the version they triaged under, and `GET /guidelines/change-impact`
   (service: `guideline_change_impact`, `retriage.py`) re-runs
   rules-only triage over the awaiting queue to report which
   outcomes the change would move. The report mutates nothing —
   whether a moved submission is re-run or left to its underwriter
   with the note is a human call. Every move is attributed
   (`retriage.explain_move`): the guideline document and the
   specific line in it the fresh outcome rests on, computed from
   the same trigger strings and facts the rules produced.
   Multi-line bundles are re-triaged **per line** — each stored
   sub-analysis against its own recomputed facts, the affected
   lines named with their movement and reasons, plus what the
   bundle roll-up would become — never as one flattened pass over
   the bundle payload, which would be fiction.
3. **Your intake adapter** — map your inbox/portal feed to
   `SubmissionInput`.
4. **Your output route** — point `ACTION_WEBHOOK_URL` at your work
   queue or policy admin system.
5. **Your provider** — set `MODEL_BACKEND`; run `make llm-evals` and
   read the cross-check agreement before trusting the LLM paths.

## 9. Production: what is in the blueprint, and what is not

Already the production shape, in this repo:

- PostgreSQL as the system of record for decisions/history, the
  official LangGraph Postgres checkpointer for gate state, and
  pgvector for guideline embeddings — one database, schema owned by
  numbered migrations applied at startup.
- Submission documents in S3-compatible object storage behind a
  port; submissions reference object keys.
- A stateless API (multi-stage image, non-root, healthcheck),
  structured JSON logs with request IDs, `/health` + `/readiness`,
  env-only configuration, graceful shutdown via the lifespan.

What a production deploy still needs beyond the blueprint — the
honest list:

- **Secrets management.** Compose defaults and `.env` are for
  reference runs; production injects `DATABASE_URL`, provider keys,
  and bucket credentials from a secrets manager / workload identity.
- **TLS everywhere.** API ingress, api→Postgres, and api→object
  storage are plaintext in the reference stack.
- **Backups and recovery.** Point-in-time recovery for Postgres and
  versioning/replication for the documents bucket; the blueprint
  ships neither policies nor restore drills.
- **Real identity and authorisation.** SSO plus per-broker/per-team
  access control; the decision endpoints record *who* — production
  must prove it. The optional `API_KEY` is a demo gate.
- **Database HA and connection management at scale.** The app opens
  short-lived connections per store call and one pool for the
  checkpointer; a fleet deployment adds a pooler (PgBouncer) and a
  replicated Postgres.
- **Ingestion hardening.** Email/portal adapters with attachment
  handling, malware scanning, and OCR/document AI behind the intake
  contract.
- **Data governance for real submission PII.** Encryption at rest,
  retention schedules, redaction in logs and traces.
- **A review queue product.** Assignment and bulk actions — SLA
  budgets and breach flags exist on this demo queue (§4), routing
  a case to a named underwriter's queue does not.
- **Audit export** of decisions, traces, and guardrail verdicts.
- **Monitoring and alerting** on the emitted telemetry and JSON logs
  (provider latency, budget degradations, repair rates, reviewer
  blocks, readiness flaps) — the signals exist; the dashboards and
  paging do not.
- **Load and soak testing** of the batch path and the gate, and an
  ANN index (HNSW) once the guideline corpus outgrows exact scans.
- **A real eval program**: golden cases grown from (anonymised,
  consented) historical submissions, reviewed by underwriters.

## 10. Limitations of this prototype

- The appetite matrix, guidelines, and authority limits are a
  synthetic teaching model.
- The offline fallback drafts templates; it validates plumbing and
  guardrails, not language quality.
- Extraction consumes document text only; scanned-document OCR is
  out of scope (adapter concern).
- Docker stacks are reviewed but not build-verified in the
  development environment (no Docker daemon there). The Postgres,
  checkpointer, and pgvector paths *are* verified against a real
  PostgreSQL by the gated integration tests.
