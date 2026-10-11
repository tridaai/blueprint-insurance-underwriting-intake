"""Guideline corpus snapshots — the text in force, kept per version.

The corpus is versioned (``guidelines_data.GUIDELINES_VERSION``) and
every result is stamped with the version it triaged under. Until now
that stamp was all that survived: the change-impact report
(``retriage.py``) attributes a move by quoting the guideline line the
fresh outcome rests on — from the corpus as it reads *today*. Edit
the corpus and yesterday's attribution silently quotes text nobody
triaged under.

A snapshot closes that hole. When the corpus is registered under a
version, its full text is kept — immutably — in the store's document
slot, beside the proposals and settings documents:

- Registering a version that is not yet kept stores the snapshot.
- Registering the same version with the same text is an idempotent
  no-op (the same snapshot comes back, ``created`` false).
- Registering the same version with *different* text raises
  :class:`CorpusVersionConflict`. A version names one text forever;
  edited text is a new version (bump ``GUIDELINES_VERSION``), never
  a silent overwrite of the version results already cite.

With snapshots kept, three reads become possible: the guideline text
**as it read when a submission triaged** (the snapshot for the
version stamped on its result), a **text diff across two versions**
(which guideline changed, and exactly which lines left and entered),
and change-impact attribution that can quote both the line as it
reads now and the line as it read then.

Snapshots are evidence, not configuration: nothing in the pipeline
triages *from* a snapshot — triage always reads the live corpus.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from datetime import datetime, timezone

from .guidelines_data import GUIDELINES, GUIDELINES_VERSION
from .store import ApprovalRecord

# The store document holding every kept snapshot, keyed by version —
# the same one-document idiom as the proposals and settings records.
SNAPSHOTS_DOCUMENT = "guidelines_corpus_snapshots"


class CorpusVersionConflict(ValueError):
    """A version was re-registered with different text than it holds.

    A corpus version names one text forever. The fix is operational,
    and the message says it: bump ``GUIDELINES_VERSION`` for the
    edited corpus instead of re-using the version results cite.
    """


def corpus_content_hash(guidelines: list[dict]) -> str:
    """Content hash of a corpus — the retriever's hash idiom, widened
    to the whole corpus: id, title, text and lines, in corpus order."""
    return hashlib.sha256(
        "\n".join(
            f"{g['id']}|{g['title']}|{g['text']}|{','.join(g.get('lines', []))}"
            for g in guidelines
        ).encode("utf-8")
    ).hexdigest()


def _corpus_copy(guidelines: list[dict]) -> list[dict]:
    return [
        {
            "id": g["id"],
            "title": g["title"],
            "text": g["text"],
            "lines": list(g.get("lines", [])),
        }
        for g in guidelines
    ]


def build_snapshot(
    guidelines: list[dict], version: str, *, registered_at: str | None = None
) -> dict:
    """The stored shape of one version's corpus text."""
    return {
        "version": version,
        "content_hash": corpus_content_hash(guidelines),
        "registered_at": registered_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guideline_count": len(guidelines),
        "guidelines": _corpus_copy(guidelines),
    }


def load_snapshots(store) -> dict[str, dict]:
    """Every kept snapshot, keyed by corpus version ({} when none)."""
    document = store.get_document(SNAPSHOTS_DOCUMENT) or {}
    return dict(document.get("versions") or {})


def get_snapshot(store, version: str) -> dict | None:
    """The kept snapshot for one version, or None when none is kept."""
    return load_snapshots(store).get(version)


def register_snapshot(
    store,
    guidelines: list[dict] | None = None,
    version: str | None = None,
    *,
    registered_at: str | None = None,
) -> tuple[dict, bool]:
    """Keep the corpus text under its version — immutably.

    Returns ``(snapshot, created)``. Re-registering the same version
    with identical text returns the kept snapshot with ``created``
    false; different text under a kept version raises
    :class:`CorpusVersionConflict` and stores nothing.
    """
    guidelines = GUIDELINES if guidelines is None else guidelines
    version = GUIDELINES_VERSION if version is None else version
    versions = load_snapshots(store)
    existing = versions.get(version)
    if existing is not None:
        if existing.get("content_hash") == corpus_content_hash(guidelines):
            return existing, False
        raise CorpusVersionConflict(
            f"guidelines corpus version {version!r} is already kept with "
            "different text (content hash "
            f"{(existing.get('content_hash') or '')[:12]}…); a version "
            "names one text forever — bump GUIDELINES_VERSION for the "
            "edited corpus instead of re-registering this version"
        )
    snapshot = build_snapshot(guidelines, version, registered_at=registered_at)
    versions[version] = snapshot
    store.save_document(SNAPSHOTS_DOCUMENT, {"versions": versions})
    return snapshot, True


def snapshot_metadata(snapshot: dict) -> dict:
    """A snapshot without its text — the list view."""
    return {
        "version": snapshot.get("version", ""),
        "content_hash": snapshot.get("content_hash", ""),
        "registered_at": snapshot.get("registered_at", ""),
        "guideline_count": snapshot.get("guideline_count", 0),
    }


def _text_lines(text: str) -> list[str]:
    """A guideline text as its lines — the sentences an auditor reads
    (the same split the change-impact attribution quotes by)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def diff_snapshots(old: dict, new: dict) -> dict:
    """Line-level text diff between two kept snapshots, in code.

    Guidelines are matched by id: ids only in the new snapshot are
    ``added``, ids only in the old are ``removed``, and an id in both
    whose title or text moved is ``changed`` with the exact lines
    that left and entered (sentence-level, order preserved).
    """
    old_by_id = {g["id"]: g for g in old.get("guidelines", [])}
    new_by_id = {g["id"]: g for g in new.get("guidelines", [])}
    added = [gid for gid in new_by_id if gid not in old_by_id]
    removed = [gid for gid in old_by_id if gid not in new_by_id]
    changed: list[dict] = []
    unchanged = 0
    for guideline_id, new_guideline in new_by_id.items():
        old_guideline = old_by_id.get(guideline_id)
        if old_guideline is None:
            continue
        if (
            old_guideline.get("text") == new_guideline.get("text")
            and old_guideline.get("title") == new_guideline.get("title")
        ):
            unchanged += 1
            continue
        old_lines = _text_lines(old_guideline.get("text", ""))
        new_lines = _text_lines(new_guideline.get("text", ""))
        removed_lines: list[str] = []
        added_lines: list[str] = []
        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "delete"):
                removed_lines.extend(old_lines[i1:i2])
            if tag in ("replace", "insert"):
                added_lines.extend(new_lines[j1:j2])
        changed.append(
            {
                "guideline_id": guideline_id,
                "title": new_guideline.get("title", ""),
                "title_changed": old_guideline.get("title")
                != new_guideline.get("title"),
                "removed_lines": removed_lines,
                "added_lines": added_lines,
            }
        )
    return {
        "from_version": old.get("version", ""),
        "to_version": new.get("version", ""),
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": unchanged,
        "note": (
            "Text diff computed in code between the two kept corpus "
            "snapshots; lines are the guideline texts' sentences, "
            "quoted verbatim from the snapshots."
        ),
    }


def guideline_texts_as_triaged(
    store, record: ApprovalRecord, guideline_id: str | None = None
) -> dict:
    """The cited guidelines' text as it read when this record triaged.

    Each cited guideline (``result.guidelines``) is answered from the
    kept snapshot for the version stamped on the result — ``source:
    "snapshot"``. When no snapshot is kept for that version (it
    predates snapshotting, or was never registered while in force),
    the entry falls back to the text stored on the triage record
    itself and says so — the record's own copy, never today's corpus
    dressed as history.
    """
    result = record.result
    version = result.guidelines_version or ""
    snapshot = get_snapshot(store, version) if version else None
    by_id = (
        {g["id"]: g for g in snapshot.get("guidelines", [])} if snapshot else {}
    )
    entries: list[dict] = []
    for cited in result.guidelines:
        if guideline_id and cited.guideline_id != guideline_id:
            continue
        kept = by_id.get(cited.guideline_id)
        if kept is not None:
            entries.append(
                {
                    "guideline_id": cited.guideline_id,
                    "title": kept.get("title", ""),
                    "text": kept.get("text", ""),
                    "source": "snapshot",
                }
            )
        else:
            entries.append(
                {
                    "guideline_id": cited.guideline_id,
                    "title": cited.title,
                    "text": cited.text,
                    "source": "record",
                }
            )
    return {
        "submission_id": result.submission_id,
        "triaged_under": version or "unversioned",
        "snapshot_kept": snapshot is not None,
        "guidelines": entries,
        "note": (
            "Texts are quoted from the corpus snapshot kept for the "
            "version this submission triaged under. An entry sourced "
            "from the record means no snapshot is kept for that "
            "version; the text stored on the triage record itself is "
            "shown instead."
            if snapshot is None
            else "Texts are quoted verbatim from the corpus snapshot "
            "kept for the version this submission triaged under."
        ),
    }
