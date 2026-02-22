from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from rag_lab.core.config import ensure_runtime_directories, settings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MANIFEST_PATH = "manifest.json"


class FileStorageError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class StoredFile:
    doc_id: str
    file_hash: str
    original_filename: str
    stored_path: Path
    content_type: str
    size_bytes: int
    uploaded_at: str
    is_duplicate: bool


def _manifest_file_path() -> Path:
    return settings.uploads_dir / MANIFEST_PATH


def _read_manifest() -> dict[str, dict[str, str | int]]:
    path = _manifest_file_path()
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(data, dict):
        return data

    return {}


def _write_manifest(manifest: dict[str, dict[str, str | int]]) -> None:
    path = _manifest_file_path()
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


async def save_upload(upload_file: UploadFile) -> StoredFile:
    ensure_runtime_directories()

    original_filename = upload_file.filename or ""
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise FileStorageError(
            400,
            f"Unsupported file extension '{suffix or '<none>'}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await upload_file.read()
    if not content:
        raise FileStorageError(400, "Uploaded file is empty")
    if len(content) > settings.max_upload_size_bytes:
        raise FileStorageError(
            413,
            f"Uploaded file exceeds size limit ({settings.max_upload_size_bytes} bytes)",
        )

    file_hash = hashlib.sha256(content).hexdigest()
    doc_id = file_hash
    manifest = _read_manifest()
    existing = manifest.get(doc_id)
    if existing:
        existing_path = Path(str(existing["stored_path"]))
        if existing_path.exists():
            return StoredFile(
                doc_id=doc_id,
                file_hash=file_hash,
                original_filename=str(existing["original_filename"]),
                stored_path=existing_path,
                content_type=str(existing["content_type"]),
                size_bytes=int(existing["size_bytes"]),
                uploaded_at=str(existing["uploaded_at"]),
                is_duplicate=True,
            )

    stored_name = f"{doc_id[:12]}-{uuid4().hex}{suffix}"
    stored_path = settings.uploads_dir / stored_name
    stored_path.write_bytes(content)

    uploaded_at = datetime.now(UTC).isoformat()
    content_type = upload_file.content_type or "application/octet-stream"

    metadata = {
        "doc_id": doc_id,
        "file_hash": file_hash,
        "original_filename": original_filename,
        "stored_path": str(stored_path),
        "content_type": content_type,
        "size_bytes": len(content),
        "uploaded_at": uploaded_at,
    }
    manifest[doc_id] = metadata
    _write_manifest(manifest)
    stored_path.with_suffix(stored_path.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return StoredFile(
        doc_id=doc_id,
        file_hash=file_hash,
        original_filename=original_filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=len(content),
        uploaded_at=uploaded_at,
        is_duplicate=False,
    )
