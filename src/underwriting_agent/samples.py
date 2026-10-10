"""Bundled synthetic sample submissions.

The samples ship inside the package (``data/sample_submissions.json``)
so the CLI, demo, API, and console work identically from a source
checkout and a pip install. Every submission is synthetic — invented
insureds, brokers, and loss histories, written for this blueprint.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import SubmissionInput

_DATA = Path(__file__).parent / "data" / "sample_submissions.json"


def load_sample_submissions() -> list[dict]:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def sample_submission_models() -> list[SubmissionInput]:
    return [SubmissionInput.model_validate(item) for item in load_sample_submissions()]
