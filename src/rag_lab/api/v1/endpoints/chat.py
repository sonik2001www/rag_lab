from fastapi import APIRouter
from pydantic import BaseModel
import httpx

router = APIRouter(prefix="/ollama/chat", tags=["chat"])


MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    r = httpx.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": req.message, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    answer = r.json()["response"]
    return ChatResponse(answer=answer)