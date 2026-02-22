from fastapi import APIRouter, HTTPException

from rag_lab.schemas.chat import ChatRequest, ChatResponse
from rag_lab.services.chat_service import ChatServiceError, generate_answer

router = APIRouter(prefix="/ollama/chat", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        answer = await generate_answer(req.message)
    except ChatServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return ChatResponse(answer=answer)
