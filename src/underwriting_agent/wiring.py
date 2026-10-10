"""Composition root: build the runtime collaborators from the environment.

Every surface (API, CLI, demo, evals) constructs its service through
here, so configuration means the same thing everywhere: the factories
in ``model_backends`` / ``retriever`` / ``store`` / ``checkpoints``
read the environment (with the repo-root ``.env`` loaded), and tests
inject doubles by passing them explicitly.
"""

from __future__ import annotations

from .checkpoints import get_checkpointer
from .model_backends import get_backend
from .ports import Checkpointer, ModelBackend, Retriever, Store
from .retriever import get_retriever
from .service import SubmissionService
from .store import default_store


def build_backend() -> ModelBackend:
    return get_backend()


def build_retriever() -> Retriever:
    return get_retriever()


def build_store() -> Store:
    return default_store()


def build_checkpointer() -> Checkpointer | None:
    return get_checkpointer()


def build_service_from_env(
    *,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    store: Store | None = None,
    checkpointer: Checkpointer | bool | None = None,
) -> SubmissionService:
    """A service wired from the environment, with any collaborator
    overridable (tests and the CLI batch path inject doubles here)."""
    return SubmissionService(
        backend=backend,
        retriever=retriever,
        store=store,
        checkpointer=checkpointer,
    )
