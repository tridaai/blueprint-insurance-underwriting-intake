"""FastAPI surface for the Underwriting Submission Intake Agent prototype.

Run:  uvicorn underwriting_agent.api:app    (or `make serve`)
UI:   http://localhost:8000/                (minimal demo console, no build step)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .config import env_str, load_dotenv, silence_langchain_deprecation_warnings
from .db import database_url, ensure_migrated, ping as db_ping
from .observability import configure_json_logging, new_request_id, request_id_ctx

# Must precede the service import (which loads the graph/langgraph).
silence_langchain_deprecation_warnings()

from .errors import ProviderError
from .events import CallbackSink, RunEvent
from .guidelines_data import GUIDELINES
from .insights import audit_csv
from .samples import load_sample_submissions
from .schemas import (
    AgentResult,
    BreachAcknowledgementRequest,
    DecisionRequest,
    ProposalDecisionRequest,
    SettingsRestoreRequest,
    SnapshotPruneRequest,
    SlaCalibrationDraftRequest,
    SubmissionInput,
)
from .wiring import build_service_from_env

# Load the repo-root .env at startup (real environment variables win), so
# MODEL_BACKEND / RETRIEVER / API keys can live in the file — see .env.example.
load_dotenv()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional API-key gate, enabled by setting ``API_KEY``.

    When ``API_KEY`` is set, every data endpoint (read + mutating)
    requires the matching ``X-API-Key`` header. When it is unset the
    API is open — the local-dev default, stated here and in the docs
    rather than implied. The console page and /health stay open either
    way; the console carries an API-key field for the gated calls.
    """
    expected = env_str("API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key (send the X-API-Key header).",
        )


_AUTH = [Depends(require_api_key)]

logger = logging.getLogger("underwriting_agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: JSON logging, schema migrations. Shutdown: drain.

    With ``DATABASE_URL`` set, migrations run before the first
    request is served — a production process never starts against a
    half-migrated database. Shutdown closes the checkpointer's
    connection pool; uvicorn's own SIGTERM handling (stop accepting,
    finish in-flight requests, then this lifespan exit) provides the
    graceful drain.
    """
    configure_json_logging(env_str("LOG_LEVEL", "INFO") or "INFO")
    if database_url():
        ensure_migrated()
        logger.info("startup: migrations current, database configured")
    else:
        logger.info("startup: no DATABASE_URL — in-memory test doubles")
    yield
    try:
        checkpointer = service._get_checkpointer()
        pool = getattr(checkpointer, "conn", None)
        if pool is not None and hasattr(pool, "close"):
            pool.close()
    except Exception:  # shutdown must not fail on cleanup
        pass
    logger.info("shutdown: drained")


app = FastAPI(
    title="Trida AI Blueprint — Insurance Underwriting Submission Intake Agent",
    description="Reference prototype. Synthetic data only. Drafts stop at an underwriter gate; the agent never quotes a premium, never binds coverage, and never declines on its own.",
    version="0.1.0",
    lifespan=lifespan,
)

service = build_service_from_env()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign/propagate X-Request-ID; one JSON access line per request."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "event": "http_request",
            },
        )
        return response
    finally:
        request_id_ctx.reset(token)


@app.exception_handler(ProviderError)
def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    """A provider failure is a clean 502 with the actionable, translated
    message (backend, endpoint, likely fix) — never a stack dump."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"
_BRAND_DIR = _STATIC_DIR / "brand"

# Branded static assets (logo, favicon) for the demo console.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return FileResponse(_BRAND_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return FileResponse(_BRAND_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/evals/results", dependencies=_AUTH)
def eval_results() -> dict:
    """Latest eval summary, as emitted by ``evals/run_evals.py``.

    Looked up in the packaged copy first (present in installs/Docker),
    then in a repo checkout's ``evals/results.json``. 404 when no eval
    run has been recorded yet — the console shows a 'run evals' state.
    """
    candidates = [
        Path(__file__).parent / "data" / "eval_results.json",
        Path.cwd() / "evals" / "results.json",
        Path(__file__).resolve().parents[2] / "evals" / "results.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise HTTPException(
        status_code=404,
        detail="No eval results yet. Run: python evals/run_evals.py",
    )


def _decision_actor(request: DecisionRequest) -> str:
    actor = request.actor_name()
    if not actor:
        raise HTTPException(
            status_code=422,
            detail="Provide 'actor' — the underwriter's name ('approver'/'reviewer' are accepted legacy aliases).",
        )
    return actor


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "prototype": "trida-blueprint-insurance-underwriting-intake"}


@app.get("/readiness")
def readiness() -> JSONResponse:
    """Readiness = the process can serve real work.

    With ``DATABASE_URL`` configured that means the system of record
    answers a ping — a replica that cannot reach its database is not
    ready and says so (503), so a load balancer drains it. Without a
    database the process serves on in-memory test doubles and is
    ready by definition; the payload names that mode honestly.
    """
    if database_url():
        if db_ping():
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "database": "postgres"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unreachable"},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "database": "in-memory test doubles"},
    )


@app.get("/guidelines", dependencies=_AUTH)
def list_guidelines() -> list[dict]:
    return GUIDELINES


@app.get("/guidelines/change-impact", dependencies=_AUTH)
def guidelines_change_impact() -> dict:
    """Guideline change-impact report: rules-only re-triage over the
    awaiting queue against the current corpus version — which
    submissions' outcome or tier would move. Report only; nothing
    stored is changed by reading it."""
    return service.guideline_change_impact()


@app.get("/guidelines/diff-trigger-calibration", dependencies=_AUTH)
def diff_trigger_calibration() -> dict:
    """Diff-trigger calibration: replay the stored renewals at
    candidate referral-trigger thresholds and show how many
    referrals each setting would have produced, family by family —
    alongside what actually fired and what underwriters decided.
    Read-only; nothing stored is changed by reading it. The replay
    centres on the thresholds in force (``threshold_source`` says
    whether they come from the environment or an accepted
    proposal's settings record)."""
    return service.diff_trigger_calibration()


@app.get("/calibration/settings", dependencies=_AUTH)
def calibration_settings() -> dict:
    """The diff-trigger thresholds in force and their provenance:
    the values, the source (environment, or the settings record an
    accepted proposal wrote), and the record's change history."""
    return service.diff_threshold_settings()


@app.post("/calibration/settings/revert", dependencies=_AUTH)
def revert_threshold_settings(request: DecisionRequest) -> dict:
    """Revert the threshold settings record's latest accepted
    change: the prior value is restored as a new audited history
    entry naming the proposal it undoes. Nothing is deleted —
    the way back is itself on the record."""
    try:
        return service.revert_diff_threshold_settings(_decision_actor(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/queue/sla/settings/revert", dependencies=_AUTH)
def revert_sla_budget_settings(request: DecisionRequest) -> dict:
    """Revert the SLA-budget settings record's latest accepted
    change, the same append-only way."""
    try:
        return service.revert_sla_budget_settings(_decision_actor(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/calibration/settings/restore", dependencies=_AUTH)
def restore_threshold_settings(request: SettingsRestoreRequest) -> dict:
    """Restore the threshold settings record to ANY prior history
    entry's state — not just the latest change's prior value, as
    revert does. The restore is one new audited history entry
    naming the entry it restores to; the entries it passes over
    stay on the record. An optional ``note`` in the request is
    kept on the entry as its decision note."""
    try:
        return service.restore_diff_threshold_settings(
            _decision_actor(request),
            request.history_index,
            note=request.note or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/queue/sla/settings/restore", dependencies=_AUTH)
def restore_sla_budget_settings(request: SettingsRestoreRequest) -> dict:
    """Restore the SLA-budget settings record to a chosen history
    entry's state, the same append-only way, with the same
    optional decision note on the restore entry."""
    try:
        return service.restore_sla_budget_settings(
            _decision_actor(request),
            request.history_index,
            note=request.note or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/calibration/settings/restore/preview", dependencies=_AUTH)
def preview_restore_threshold_settings(
    request: SettingsRestoreRequest,
) -> dict:
    """Preview a threshold-settings restore: the values in force
    now, the values as of the chosen history entry, and exactly
    which dimension values would change — a dry run. Nothing is
    written and nothing is appended to the record; a preview of
    a restore that would move nothing says so instead of
    failing. A ``note`` in the request is the draft decision
    note: echoed back with the moves it would be stored with,
    kept nowhere."""
    try:
        return service.preview_restore_diff_threshold_settings(
            request.history_index, note=request.note or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/queue/sla/settings/restore/preview", dependencies=_AUTH)
def preview_restore_sla_budget_settings(
    request: SettingsRestoreRequest,
) -> dict:
    """Preview an SLA-budget settings restore, the same dry-run
    way: which tier budgets would change, from what to what,
    before the operator commits — and the drafted decision note
    echoed back beside them."""
    try:
        return service.preview_restore_sla_budget_settings(
            request.history_index, note=request.note or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/calibration/proposals", dependencies=_AUTH)
def calibration_proposals() -> dict:
    """Every stored calibration proposal with the thresholds in
    force. A proposal is a drafted threshold change with its
    evidence attached; it changes nothing until a human accepts
    it."""
    return {
        "proposals": service.calibration_proposals(),
        "settings": service.diff_threshold_settings(),
        "sla_settings": service.sla_budget_settings(),
    }


@app.post("/calibration/proposals/draft", dependencies=_AUTH)
def draft_calibration_proposals() -> dict:
    """Scan the stored renewals for proposal-worthy threshold moves
    and draft them (see proposals.py for the evidence bar: the
    moved referrals must be entirely decided and one-sided).
    Drafting writes only proposal documents — no threshold moves."""
    return service.draft_calibration_proposals()


@app.post("/calibration/proposals/refresh-expired", dependencies=_AUTH)
def refresh_expired_calibration_proposals(request: DecisionRequest) -> dict:
    """Refresh every expired proposal in one call: each expired
    draft's evidence scan re-runs individually against today's
    store — renewed in place with a fresh snapshot, or withdrawn
    when the evidence no longer carries the move — and the report
    lists renewals and withdrawals side by side. The sweep decides
    nothing; a renewed draft still waits for its human."""
    return service.refresh_expired_calibration_proposals(
        _decision_actor(request)
    )


@app.get("/calibration/proposals/refresh-expired/preview", dependencies=_AUTH)
def preview_refresh_expired_calibration_proposals() -> dict:
    """Preview the refresh-all sweep: every expired draft's
    evidence scan runs individually — the same scans the sweep
    runs — and the report lists would-renew and would-withdraw
    side by side, without writing anything. The sweep that
    follows on the same store produces exactly these outcomes."""
    return service.preview_refresh_expired_calibration_proposals()


@app.post("/calibration/proposals/{proposal_id}/decide", dependencies=_AUTH)
def decide_calibration_proposal(
    proposal_id: str, request: ProposalDecisionRequest
) -> dict:
    """Decide one calibration proposal. Accepting writes the
    proposed threshold into the settings record the classifier
    reads, with the change in the record's audit history;
    rejecting leaves every threshold untouched. A proposal is
    decided once."""
    try:
        return service.decide_calibration_proposal(
            proposal_id,
            _decision_actor(request),
            accept=request.accept,
            note=request.note or "",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/calibration/proposals/{proposal_id}/refresh", dependencies=_AUTH)
def refresh_calibration_proposal(
    proposal_id: str, request: DecisionRequest
) -> dict:
    """Refresh one expired proposal: re-run its evidence scan
    against the store as it stands now. The proposal is renewed
    in place (fresh snapshot, fresh expiry window) when the
    evidence still carries a one-sided move, or withdrawn when
    it no longer does. A live draft cannot be refreshed, and an
    expired draft cannot be decided until it is."""
    try:
        return service.refresh_calibration_proposal(
            proposal_id, _decision_actor(request)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/calibration/proposals/{proposal_id}/refresh/preview",
    dependencies=_AUTH,
)
def preview_refresh_calibration_proposal(proposal_id: str) -> dict:
    """Preview one expired proposal's refresh: its evidence scan
    runs and the outcome it would produce is reported — would
    renew, with the fresh snapshot it would take, or would
    withdraw, with the reason — while nothing is written. The
    refresh's own validations apply unchanged."""
    try:
        return service.preview_refresh_calibration_proposal(proposal_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/samples", dependencies=_AUTH)
def list_samples() -> list[dict]:
    return load_sample_submissions()


@app.get("/inbox", dependencies=_AUTH)
def inbox() -> dict:
    """The intake queue grouped by insured and by broker — counts,
    and each party's latest outcome, risk tier, and readiness —
    plus the status-note follow-through digest: notes answered,
    partially answered, and still outstanding across insureds,
    oldest outstanding first."""
    return service.inbox()


@app.get("/inbox/insureds", dependencies=_AUTH)
def inbox_insureds() -> list[dict]:
    return service.inbox()["insureds"]


@app.get("/inbox/brokers", dependencies=_AUTH)
def inbox_brokers() -> list[dict]:
    return service.inbox()["brokers"]


@app.get("/scorecards", dependencies=_AUTH)
def scorecards() -> dict:
    """History scorecards for every insured and broker in the store:
    submissions, outcome mix, approval rate, average completeness,
    top referral reasons."""
    return service.scorecards()


@app.get("/queue", dependencies=_AUTH)
def queue_aging_view() -> dict:
    """Queue aging: the submissions still awaiting an underwriter,
    bucketed by days since they were submitted (with the outcome
    mix and the single oldest case) — a stranded queue is visible
    here before a broker complains. When a digest snapshot has
    been kept, the breach list is trend-aware: breached items are
    flagged new or continuing against the latest snapshot, and
    ``breach_movement`` names the new, continuing, and cleared
    breaches since it."""
    return service.queue()


@app.get("/queue/sla/settings", dependencies=_AUTH)
def sla_budget_settings() -> dict:
    """The queue SLA budgets in force and their provenance: the
    values, the source (environment, or the settings record an
    accepted SLA proposal wrote), and the record's change
    history."""
    return service.sla_budget_settings()


@app.get("/queue/sla/calibration", dependencies=_AUTH)
def sla_budget_calibration() -> dict:
    """SLA-budget calibration: per risk tier, the decided cases'
    waits as a distribution (percentiles against the budget in
    force, its source named) and a sweep of candidate budgets —
    breach counts and decision splits at each, with the cases
    each candidate would move and whether that evidence is the
    one-sided kind an SLA proposal is drafted from. Read-only;
    nothing stored is changed by reading it."""
    return service.sla_budget_calibration()


@app.post("/queue/sla/calibration/draft", dependencies=_AUTH)
def draft_sla_proposal_from_row(request: SlaCalibrationDraftRequest) -> dict:
    """Draft an SLA-budget proposal from one chosen calibration
    row: the (tier, value) row is recomputed against the store as
    it stands and — when it is proposal-ready — stored as a draft
    carrying the row's evidence verbatim, stamped with the row
    origin and the drafter's name. The drafting discipline is
    the proposal scan's (one open question per tier, rejected
    directions never re-asked, swept candidates only); a row
    that is not proposal-ready is refused with the reason, and
    drafting moves no budget — the proposal still waits for its
    human decision."""
    try:
        return service.draft_sla_proposal_from_row(
            request.tier, request.value, _decision_actor(request)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/queue/sla/calibration/snapshots", dependencies=_AUTH)
def snapshot_sla_calibration(request: DecisionRequest) -> dict:
    """Keep a snapshot of the SLA calibration report as it stands
    now: the budgets in force, the decided cases weighed, and per
    tier which candidate rows were proposal-ready — stored dated
    and under the taker's name. Operator-kept, never on a timer;
    snapshots are what the calibration diff compares."""
    return service.snapshot_sla_calibration(_decision_actor(request))


@app.get("/queue/sla/calibration/snapshots", dependencies=_AUTH)
def sla_calibration_snapshots() -> dict:
    """Every stored SLA-calibration snapshot, in the order kept,
    with the prunes recorded on the document."""
    return {
        "snapshots": service.sla_calibration_snapshots(),
        "prunes": service.sla_calibration_snapshot_prunes(),
    }


@app.post("/queue/sla/calibration/snapshots/prune", dependencies=_AUTH)
def prune_sla_calibration_snapshots(request: SnapshotPruneRequest) -> dict:
    """Prune the kept calibration snapshots to the newest ``keep``:
    the only way a kept report leaves the document, operator-called
    and recorded on the document itself (who, when, what left, the
    surviving range). ``keep`` below 1 is refused — the document
    always keeps a baseline for the diff."""
    try:
        return service.prune_sla_calibration_snapshots(
            _decision_actor(request), request.keep
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/queue/sla/calibration/snapshots/prune/preview", dependencies=_AUTH
)
def preview_prune_sla_calibration_snapshots(
    request: SnapshotPruneRequest,
) -> dict:
    """Preview the calibration-snapshot prune: which kept reports
    a keep-last-``keep`` would remove and what range would
    survive — a dry run that writes nothing and records nothing,
    so the document a budget debate cites is pruned against the
    same evidence the prune itself will keep. ``keep`` below 1
    is refused, exactly as the prune refuses it."""
    try:
        return service.preview_prune_sla_calibration_snapshots(request.keep)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/queue/sla/calibration/snapshots/diff", dependencies=_AUTH)
def sla_calibration_snapshot_diff() -> dict:
    """The proposal-ready movement between the two latest
    calibration snapshots: per tier, which candidate rows became
    proposal-ready and which stopped being, beside the budget
    and decided-case movement. Fewer than two snapshots kept is
    reported as no diff, not as no movement."""
    return service.sla_calibration_snapshot_diff()


@app.post("/queue/sla-proposals/draft", dependencies=_AUTH)
def draft_sla_budget_proposals() -> dict:
    """Scan the decided cases' waits for proposal-worthy SLA
    budget moves and draft them (see proposals.py for the
    evidence bar: the moved cases must be entirely decided and
    one-sided). Drafting writes only proposal documents — no
    budget moves. The drafted proposals join the calibration
    proposals list and are decided on the same endpoint."""
    return service.draft_sla_budget_proposals()


@app.post("/queue/sla-notifications", dependencies=_AUTH)
def notify_sla_breaches() -> dict:
    """Run the SLA-breach check: the first time the queue projection
    observes a case past its SLA budget, fire one signed
    ``sla_breach`` webhook through the output-routing seam and
    record the notice on the record (the record is the dedupe —
    re-running the check never re-fires). With no ACTION_WEBHOOK_URL
    configured, breaches are reported as observed-but-unannounced
    and nothing is recorded."""
    return service.notify_sla_breaches()


@app.get("/metrics", dependencies=_AUTH)
def metrics() -> dict:
    """Operational aggregates over the store: runs, decisions,
    outcome mix, model calls/tokens/cost, latency, dispatch totals."""
    return service.metrics()


@app.get("/ops/digest", dependencies=_AUTH)
def ops_digest_view() -> dict:
    """The operator digest: one composed read over the surfaces
    that rot quietly — stuck approval-webhook dispatches,
    exhausted SLA-breach notices, queue cases past their SLA
    budget, status-note follow-through still outstanding (aged
    ones escalated), and calibration proposals nearing or past
    their evidence expiry. Computed at read time and never sent;
    reading it changes nothing. When a snapshot has been kept,
    the digest carries a ``movement`` block against the latest
    snapshot and the headline speaks it."""
    return service.ops_digest()


@app.post("/ops/digest/snapshots", dependencies=_AUTH)
def snapshot_ops_digest(request: DecisionRequest) -> dict:
    """Snapshot the ops digest as it stands now: the headline,
    the counts, and the ids behind each section, stored under the
    taker's name. Operator-taken, never on a timer — snapshots are
    what the digest trend compares. Taking one changes nothing
    but the snapshot list."""
    return service.snapshot_ops_digest(_decision_actor(request))


@app.get("/ops/digest/snapshots", dependencies=_AUTH)
def ops_digest_snapshots() -> dict:
    """Every stored ops-digest snapshot, in the order taken, with
    the prunes recorded on the document."""
    return {
        "snapshots": service.ops_digest_snapshots(),
        "prunes": service.ops_digest_snapshot_prunes(),
    }


@app.post("/ops/digest/snapshots/prune", dependencies=_AUTH)
def prune_ops_digest_snapshots(request: SnapshotPruneRequest) -> dict:
    """Prune the kept digest snapshots to the newest ``keep``: the
    only way a kept snapshot leaves the document, operator-called
    and recorded on the document itself (who, when, what left, the
    surviving range). ``keep`` below 1 is refused — the document
    always keeps a baseline for the trend."""
    try:
        return service.prune_ops_digest_snapshots(
            _decision_actor(request), request.keep
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ops/digest/snapshots/prune/preview", dependencies=_AUTH)
def preview_prune_ops_digest_snapshots(
    request: SnapshotPruneRequest,
) -> dict:
    """Preview the digest-snapshot prune: which kept snapshots a
    keep-last-``keep`` would remove and what range would survive
    — a dry run that writes nothing and records nothing, so the
    operator prunes against the same evidence the prune itself
    will keep. ``keep`` below 1 is refused, exactly as the prune
    refuses it."""
    try:
        return service.preview_prune_ops_digest_snapshots(request.keep)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/ops/digest/trend", dependencies=_AUTH)
def ops_digest_trend() -> dict:
    """Movement across the stored digest snapshots: the headline
    counts as a series, their change from first to latest and
    since the previous snapshot, and which ids entered or cleared
    each section between the latest two snapshots."""
    return service.ops_digest_trend()


@app.get("/dispatch", dependencies=_AUTH)
def dispatch_ops() -> dict:
    """Dispatch operations: the approval-webhook ledgers across all
    records — total attempts, records delivered, failures (oldest
    first) with their last error, and which failures are due a
    retry under the recorded backoff schedule — plus the
    SLA-breach notices whose own ledger exhausted its attempts
    undelivered on cases still awaiting a decision."""
    return service.dispatch_ops()


@app.post(
    "/dispatch/exhausted/{submission_id}/acknowledge", dependencies=_AUTH
)
def acknowledge_exhausted_breach_notice(
    submission_id: str, request: BreachAcknowledgementRequest
) -> dict:
    """Acknowledge an exhausted SLA-breach notice as worked: an
    operator attests — under their name, with an optional note —
    that they picked up a case whose breach notice spent its
    delivery attempts and never landed. The acknowledgement is
    kept on the record, the case leaves the exhaustion worklist
    and the digest's count, and the delivery ledger is never
    rewritten: the webhook still never arrived, and the record
    keeps saying so. Only a currently-exhausted notice can be
    acknowledged, and only once."""
    try:
        return service.acknowledge_exhausted_breach_notice(
            submission_id,
            _decision_actor(request),
            note=request.note or "",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/dispatch/retry-failed", dependencies=_AUTH)
def dispatch_retry_failed() -> dict:
    """Retry every failed approval-webhook dispatch. One outcome
    per failed record; approvals themselves are never rewritten."""
    outcomes = service.retry_failed_dispatches()
    return {"count": len(outcomes), "retried": outcomes}


@app.get("/audit/export", dependencies=_AUTH)
def audit_export(format: str = Query(default="json")):
    """The audit trail — one row per stored submission with its
    decision and telemetry — as JSON (default) or CSV."""
    rows = service.audit_rows()
    if format == "csv":
        return PlainTextResponse(
            audit_csv(rows),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit-export.csv"},
        )
    if format != "json":
        raise HTTPException(status_code=422, detail="format must be 'json' or 'csv'.")
    return {"count": len(rows), "records": rows}


@app.get("/scorecards/insureds/{name}", dependencies=_AUTH)
def insured_scorecard(name: str) -> dict:
    card = service.scorecard("insured", name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"No stored submissions for insured {name!r}.")
    return card


@app.get("/insureds/{name}/loss-timeline", dependencies=_AUTH)
def insured_loss_timeline_view(name: str) -> dict:
    """The insured's loss-development timeline: every submission
    (per line for bundles) in order, its stored renewal diff, and
    one event stream per claim — appeared, developed, no longer
    listed — chained across renewals."""
    timeline = service.loss_timeline(name)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"No stored submissions for insured {name!r}.")
    return timeline


@app.get("/scorecards/brokers/{name}", dependencies=_AUTH)
def broker_scorecard(name: str) -> dict:
    card = service.scorecard("broker", name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"No stored submissions from broker {name!r}.")
    return card


@app.get("/reports/brokers/{name}/quality", dependencies=_AUTH)
def broker_quality(name: str) -> dict:
    """Broker submission-quality report: completeness trend,
    extraction discrepancy rate, and top missing elements over the
    broker's stored history — with the drafted report text. A draft
    artifact for internal review; nothing is sent to the broker."""
    report = service.broker_quality_report(name)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No stored submissions from broker {name!r}.")
    return report


@app.post("/submissions/analyze", response_model=AgentResult, dependencies=_AUTH)
def analyze(submission: SubmissionInput) -> AgentResult:
    return service.analyze(submission)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/submissions/analyze/stream", dependencies=_AUTH)
def analyze_stream(submission: SubmissionInput) -> StreamingResponse:
    """The same analysis as ``POST /submissions/analyze``, streamed.

    The run executes on a worker thread; its structured events (see
    ``events.py``) are forwarded as Server-Sent Events as they happen.
    The final event is ``run_completed`` and carries the full result
    JSON under ``result`` — the same payload the non-stream endpoint
    returns. A failed run ends the stream with a ``run_failed`` event
    carrying the clean, translated error message (no stack dump).
    """
    events_queue: queue.Queue = queue.Queue()
    sink = CallbackSink(events_queue.put)
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["result"] = service.analyze(submission, event_sink=sink)
        except Exception as exc:  # surfaced as the run_failed event below
            outcome["error"] = exc
        finally:
            events_queue.put(None)  # sentinel: the run is over

    threading.Thread(target=work, daemon=True).start()

    def generate():
        completed: RunEvent | None = None
        while True:
            event = events_queue.get()
            if event is None:
                break
            if event.type == "run_completed":
                completed = event  # held back: it goes out last, with the result
                continue
            yield _sse(event.to_dict())
        if "error" in outcome:
            yield _sse(
                {
                    "type": "run_failed",
                    "submission_id": submission.submission_id,
                    "node": None,
                    "duration_ms": None,
                    "detail": {"error": str(outcome["error"])},
                }
            )
            return
        final = (
            completed.to_dict()
            if completed is not None
            else {
                "type": "run_completed",
                "submission_id": submission.submission_id,
                "node": None,
                "duration_ms": None,
                "detail": {"status": outcome["result"].status},
            }
        )
        final["result"] = outcome["result"].model_dump(mode="json")
        yield _sse(final)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/submissions/{submission_id}", response_model=AgentResult, dependencies=_AUTH)
def get_result(submission_id: str) -> AgentResult:
    result = service.get(submission_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Submission not analyzed yet.")
    return result


@app.post("/submissions/{submission_id}/approve", response_model=AgentResult, dependencies=_AUTH)
def approve(submission_id: str, request: DecisionRequest) -> AgentResult:
    try:
        return service.approve(
            submission_id, approver=_decision_actor(request), reason=request.reason or ""
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/submissions/{submission_id}/reject", response_model=AgentResult, dependencies=_AUTH)
def reject(submission_id: str, request: DecisionRequest) -> AgentResult:
    try:
        return service.reject(
            submission_id, reviewer=_decision_actor(request), reason=request.reason or ""
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/submissions/{submission_id}/status-note", dependencies=_AUTH)
def draft_status_note(submission_id: str) -> dict:
    """Draft the underwriter status note for an SLA-breached queue
    case: composed from the record, guardrail-checked, stored on
    the record as an internal draft. Nothing is sent. 422 when the
    submission is not awaiting a decision or is within its SLA
    budget — notes are for breaches."""
    try:
        return service.draft_status_note(submission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/submissions/{submission_id}/status-note", dependencies=_AUTH)
def get_status_note(submission_id: str) -> dict:
    """The stored status note for a submission (404 when none has
    been drafted)."""
    note = service.status_note(submission_id)
    if note is None:
        raise HTTPException(status_code=404, detail="No status note drafted for this submission.")
    return note


@app.get("/submissions/{submission_id}/sla-breach", dependencies=_AUTH)
def sla_breach(submission_id: str) -> dict:
    """The SLA-breach notice for one submission and its own delivery
    ledger (separate from the approval ledger: a different event).
    ``notice`` is null until a breach check has fired for it."""
    view = service.sla_breach(submission_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Submission not analyzed yet.")
    return view


@app.post("/submissions/{submission_id}/sla-breach/retry", dependencies=_AUTH)
def retry_sla_breach(submission_id: str) -> dict:
    """Retry a failed SLA-breach dispatch: a new attempt on the
    breach ledger, carrying the original breach snapshot. The queue
    case itself is never changed by output routing."""
    try:
        return service.retry_sla_breach_notice(submission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/submissions/{submission_id}/dispatch", dependencies=_AUTH)
def dispatch_ledger(submission_id: str) -> dict:
    """The approval-webhook delivery ledger for one submission:
    every attempt with its timestamp, HTTP status, signature id,
    and the backoff the retry schedule assigned it."""
    ledger = service.dispatch_ledger(submission_id)
    if ledger is None:
        raise HTTPException(status_code=404, detail="Submission not analyzed yet.")
    return ledger


@app.post("/submissions/{submission_id}/dispatch/retry", response_model=AgentResult, dependencies=_AUTH)
def retry_dispatch(submission_id: str) -> AgentResult:
    """Retry a failed approval-webhook dispatch. The retry is a new
    ledger attempt; the approval itself is never rewritten."""
    try:
        return service.retry_dispatch(submission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
