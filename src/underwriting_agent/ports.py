"""Ports: the seams of the agent, declared in one place.

The pipeline depends on replaceable collaborators, each a
``typing.Protocol`` (structural — an implementation qualifies by
having the methods, not by inheriting):

- :class:`ModelBackend` — the language model. Real providers
  (OpenAI / Anthropic / Ollama) in provider mode; the deterministic
  template renderer as the offline fallback. Implementations live in
  ``model_backends.py``.
- :class:`Retriever` — guideline retrieval (keyword / semantic / hybrid).
  Implementations live in ``retriever.py``.
- :class:`Store` — where analysis records and human decisions live.
  Implementations live in ``store.py``: PostgreSQL (the system of
  record, selected by ``DATABASE_URL``), plus SQLite and in-memory
  test doubles the suite injects explicitly.
- :class:`EventSink` — where structured run events go (streaming,
  tracing). Implementations live in ``events.py``.
- :class:`Checkpointer` — LangGraph graph-state persistence behind
  the human-approval gate: the saver interface (the checkpoint
  library's ``BaseCheckpointSaver`` surface). Production runs use
  the official LangGraph Postgres saver; ``checkpoints.py`` selects
  it and also holds a stdlib-SQLite implementation the tests use.
  The checkpointer holds *graph state*; the :class:`Store` remains
  the record of *decisions*.
- :class:`ObjectStore` — submission documents as objects in an
  S3-compatible bucket. Implementations live in ``object_store.py``.

``wiring.py`` is the composition root that builds the concrete
collaborators from configuration; the rest of the codebase programs
against these protocols. This module is deliberately a leaf: it
imports nothing from the package at runtime, so anything may import
it without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence

if TYPE_CHECKING:  # annotation-only imports (no runtime cycles)
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
    from langchain_core.runnables import RunnableConfig

    from .events import RunEvent
    from .model_backends import DraftContext
    from .schemas import RetrievedGuideline
    from .store import ApprovalRecord


class ModelBackend(Protocol):
    """The language-model seam. Only drafting is mandatory: every other
    capability (extraction, triage cross-check, diagnosis, options,
    verification, review) is optional and discovered with ``getattr`` —
    a backend that lacks one simply engages that node's deterministic
    path. The offline fallback implements drafting only."""

    name: str

    def draft_documents(self, context: "DraftContext") -> tuple[str, str, str]:
        """Return (subject, broker letter, underwriter packet) drafts."""
        ...


class Retriever(Protocol):
    """The guideline-retrieval seam."""

    def retrieve(self, query: str, top_k: int = 3) -> list["RetrievedGuideline"]: ...


class Store(Protocol):
    """The record seam: analyses in, decisions and history out."""

    def save(self, record: "ApprovalRecord") -> None: ...

    def get(self, submission_id: str) -> "ApprovalRecord | None": ...

    def all_records(self) -> "list[ApprovalRecord]": ...

    def prior_submissions(self, exclude_submission_id: str | None = None) -> list[dict]: ...

    def decision_feedback(self, exclude_submission_id: str | None = None) -> list[dict]: ...

    def get_document(self, key: str) -> "dict | None":
        """One service document by key (calibration proposals, the
        accepted threshold settings — see ``proposals.py``), or None."""
        ...

    def save_document(self, key: str, payload: dict) -> None:
        """Write one service document, replacing any prior value."""
        ...


class EventSink(Protocol):
    """The run-events seam: one ``emit`` per structured event."""

    def emit(self, event: "RunEvent") -> None: ...


class ObjectStore(Protocol):
    """The document-storage seam: submission documents as objects.

    Keys are opaque strings the intake side chooses (the service's
    archive path uses ``submissions/<id>/documents/<doc>.txt``).
    Implementations live in ``object_store.py``: S3-compatible
    (boto3) for production, in-memory for tests.
    """

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class Checkpointer(Protocol):
    """The graph-state seam behind the approval gate.

    Mirrors the sync surface of LangGraph's ``BaseCheckpointSaver``;
    the shipped implementation (``checkpoints.py``) subclasses that
    base over stdlib SQLite, so it satisfies this protocol and
    LangGraph's own expectations at once.
    """

    def get_tuple(self, config: "RunnableConfig") -> "CheckpointTuple | None": ...

    def list(
        self,
        config: "RunnableConfig | None",
        *,
        filter: dict[str, Any] | None = None,
        before: "RunnableConfig | None" = None,
        limit: int | None = None,
    ) -> Iterator["CheckpointTuple"]: ...

    def put(
        self,
        config: "RunnableConfig",
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: dict[str, Any],
    ) -> "RunnableConfig": ...

    def put_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None: ...

    def delete_thread(self, thread_id: str) -> None: ...
