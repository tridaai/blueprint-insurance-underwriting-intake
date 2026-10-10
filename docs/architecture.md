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
| `history` | memory + feedback evidence lines (fan-out branch) | never |
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
  never rewrites history.
- **Synthetic data only**, enforced socially (CONTRIBUTING) rather
  than technically — a real deployment must add data governance.

## 8. What to change first (prototyping order)

1. **Your appetite and authority matrix** — `triage.py` /
   `facts.py` required elements, classes, limits, triggers. Add
   golden cases for every rule you change.
2. **Your guidelines** — replace the corpus behind the retriever.
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
- **A review queue product.** Assignment, SLAs, and bulk actions —
  the console here is a demo surface.
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
