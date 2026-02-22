from fastapi import APIRouter, HTTPException

from rag_lab.schemas.rag_chat import RAGChatRequest, RAGChatResponse
from rag_lab.services.rag_service import RAGServiceError, answer_with_rag

router = APIRouter(prefix="/ollama/chat", tags=["chat"])


@router.post("/chat", response_model=RAGChatResponse)
async def chat(req: RAGChatRequest) -> RAGChatResponse:
    try:
        rag_result = await answer_with_rag(req.message)
    except RAGServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return rag_result
