from typing import Literal

from pydantic import BaseModel


class IngestionFileResult(BaseModel):
    status: Literal["processed", "failed"]
    original_filename: str
    detail: str | None = None
    doc_id: str | None = None
    file_hash: str | None = None
    stored_path: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    chunks_count: int = 0


class IngestionUploadResponse(BaseModel):
    files: list[IngestionFileResult]
