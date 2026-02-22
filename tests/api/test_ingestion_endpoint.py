import pytest
from fastapi.testclient import TestClient

pytest.importorskip("multipart")

from rag_lab.main import app
from rag_lab.schemas.ingestion import IngestionFileResult


def test_upload_endpoint_happy_path(monkeypatch):
    async def _fake_ingest_upload(upload_file):
        return IngestionFileResult(
            status="processed",
            original_filename=upload_file.filename or "",
            detail="Indexed successfully",
            doc_id="doc-1",
            file_hash="hash",
            stored_path="data/uploads/doc-1.txt",
            content_type=upload_file.content_type,
            size_bytes=5,
            chunks_count=1,
        )

    monkeypatch.setattr(
        "rag_lab.api.v1.endpoints.ingestion.ingest_upload",
        _fake_ingest_upload,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/ingestion/upload",
        files=[("files", ("note.txt", b"hello", "text/plain"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"][0]["status"] == "processed"
    assert payload["files"][0]["chunks_count"] == 1
