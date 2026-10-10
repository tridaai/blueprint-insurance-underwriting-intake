"""Object storage: the port, the S3 implementation over a stub
client, and the service's fetch/archive behaviour."""

from __future__ import annotations

import io

import pytest

from underwriting_agent.object_store import InMemoryObjectStore, S3ObjectStore
from underwriting_agent.samples import sample_submission_models
from underwriting_agent.schemas import DocumentInput
from underwriting_agent.service import SubmissionService
from underwriting_agent.store import InMemoryStore

SAMPLES = {s.submission_id: s for s in sample_submission_models()}


def test_inmemory_object_store_round_trip():
    store = InMemoryObjectStore()
    assert store.exists("k") is False
    assert store.put("k", b"abc") == "k"
    assert store.exists("k") is True
    assert store.get("k") == b"abc"
    with pytest.raises(KeyError):
        store.get("missing")


class _StubS3Client:
    """Records the three boto3 calls the S3 store makes."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.put_calls.append(
            {"Bucket": Bucket, "Key": Key, "Body": Body, "ContentType": ContentType}
        )
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}


def test_s3_object_store_over_stub_client():
    client = _StubS3Client()
    store = S3ObjectStore("submissions-bucket", client=client)
    assert store.exists("a.txt") is False
    store.put("a.txt", b"hello", "text/plain")
    assert client.put_calls[0]["Bucket"] == "submissions-bucket"
    assert store.exists("a.txt") is True
    assert store.get("a.txt") == b"hello"


def test_service_fetches_key_only_documents():
    objects = InMemoryObjectStore()
    objects.put("inbox/sub-9001-acord.txt", b"ACORD form text fetched from the bucket")
    submission = SAMPLES["SUB-1001"].model_copy(
        update={
            "submission_id": "SUB-9001",
            "documents": [
                DocumentInput(
                    doc_type="acord_form",
                    document_id="ACORD-9",
                    object_key="inbox/sub-9001-acord.txt",
                )
            ],
        }
    )
    records = InMemoryStore()
    service = SubmissionService(
        store=records, checkpointer=False, object_store=objects
    )
    service.analyze(submission)
    saved = records.get("SUB-9001")
    assert saved is not None
    assert (
        saved.submission["documents"][0]["raw_text"]
        == "ACORD form text fetched from the bucket"
    )


def test_service_archives_inline_documents_and_records_keys():
    objects = InMemoryObjectStore()
    records = InMemoryStore()
    service = SubmissionService(
        store=records, checkpointer=False, object_store=objects
    )
    service.analyze(SAMPLES["SUB-1001"])
    saved = records.get("SUB-1001")
    assert saved is not None
    archived = [
        doc for doc in saved.submission["documents"] if doc.get("object_key")
    ]
    assert archived, "expected at least one document to be archived"
    for doc in archived:
        assert objects.exists(doc["object_key"])
        assert objects.get(doc["object_key"]).decode("utf-8") == doc["raw_text"]


def test_service_without_object_store_is_unchanged():
    records = InMemoryStore()
    service = SubmissionService(
        store=records, checkpointer=False, object_store=False
    )
    service.analyze(SAMPLES["SUB-1001"])
    saved = records.get("SUB-1001")
    assert saved is not None
    assert all(not doc.get("object_key") for doc in saved.submission["documents"])
