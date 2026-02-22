from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from rag_lab.core.config import settings
from rag_lab.schemas.ingestion import IngestionFileResult
from rag_lab.services.file_storage_service import FileStorageError, save_upload
from rag_lab.services.vector_store_service import get_vector_store_service

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise IngestionError(500, "PDF support dependency is not installed") from exc

    try:
        with path.open("rb") as file_handle:
            reader = PdfReader(BytesIO(file_handle.read()))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(400, "Unable to parse PDF content") from exc


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    if suffix == ".pdf":
        return _extract_text_from_pdf(path)
    raise IngestionError(400, f"Unsupported file type '{suffix}'")


def chunk_text(text: str) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ModuleNotFoundError as exc:
        raise IngestionError(500, "LangChain text splitter dependency is not installed") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]


async def ingest_upload(upload_file: UploadFile) -> IngestionFileResult:
    try:
        stored_file = await save_upload(upload_file)
    except FileStorageError as exc:
        raise IngestionError(exc.status_code, exc.detail) from exc

    text = extract_text(stored_file.stored_path)
    if not text:
        raise IngestionError(400, "No extractable text found in the uploaded file")

    chunks = chunk_text(text)
    if not chunks:
        raise IngestionError(400, "Chunking produced zero chunks")

    vector_store = get_vector_store_service()
    chunks_count = vector_store.upsert_document_chunks(
        doc_id=stored_file.doc_id,
        file_name=stored_file.original_filename,
        stored_path=stored_file.stored_path,
        chunks=chunks,
    )
    logger.info(
        "Indexed file '%s' (doc_id=%s, chunks=%s, duplicate=%s)",
        stored_file.original_filename,
        stored_file.doc_id,
        chunks_count,
        stored_file.is_duplicate,
    )

    return IngestionFileResult(
        status="processed",
        original_filename=stored_file.original_filename,
        detail="Indexed successfully",
        doc_id=stored_file.doc_id,
        file_hash=stored_file.file_hash,
        stored_path=str(stored_file.stored_path),
        content_type=stored_file.content_type,
        size_bytes=stored_file.size_bytes,
        chunks_count=chunks_count,
    )
