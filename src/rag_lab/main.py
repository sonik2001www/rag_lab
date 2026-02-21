from fastapi import FastAPI

from src.rag_lab.api.routers import router as api_router

app = FastAPI(title="RAG Lab", version="0.1.0")

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}
