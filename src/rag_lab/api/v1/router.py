from fastapi import APIRouter

from src.rag_lab.api.v1.endpoints import chat

router = APIRouter(prefix="/v1")

router.include_router(chat.router)
