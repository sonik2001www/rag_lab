from fastapi import APIRouter

from rag_lab.api.v1.endpoints.chat import router as chat_router
from rag_lab.api.v1.endpoints.ingestion import router as ingestion_router

router = APIRouter(prefix="/v1")

router.include_router(chat_router)
router.include_router(ingestion_router)
