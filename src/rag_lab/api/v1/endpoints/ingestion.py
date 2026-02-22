import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from rag_lab.schemas.ingestion import IngestionFileResult, IngestionUploadResponse
from rag_lab.services.ingestion_service import IngestionError, ingest_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post(
    "/upload",
    response_model=IngestionUploadResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                    }
                }
            },
        }
    },
)
async def upload(files: list[UploadFile] = File(...)) -> IngestionUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: list[IngestionFileResult] = []
    for file in files:
        try:
            result = await ingest_upload(file)
            results.append(result)
        except IngestionError as exc:
            if len(files) == 1:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            logger.warning(
                "Failed to index '%s': %s",
                file.filename,
                exc.detail,
            )
            results.append(
                IngestionFileResult(
                    status="failed",
                    original_filename=file.filename or "",
                    detail=exc.detail,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected ingestion failure for '%s'", file.filename)
            if len(files) == 1:
                raise HTTPException(status_code=500, detail="Unexpected ingestion failure") from exc
            results.append(
                IngestionFileResult(
                    status="failed",
                    original_filename=file.filename or "",
                    detail="Unexpected ingestion failure",
                )
            )
        finally:
            await file.close()

    return IngestionUploadResponse(files=results)
