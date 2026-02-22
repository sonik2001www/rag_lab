from pydantic import BaseModel


class RAGSource(BaseModel):
    doc_id: str
    file_name: str
    chunk_id: str
    score: float
    snippet: str


class RAGChatRequest(BaseModel):
    message: str


class RAGChatResponse(BaseModel):
    answer: str
    used_context: bool
    sources: list[RAGSource]
