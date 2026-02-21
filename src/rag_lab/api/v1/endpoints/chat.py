from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])

@router.get("/health")
def health():
    return {"status": "ok"}