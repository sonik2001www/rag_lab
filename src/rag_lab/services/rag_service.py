from __future__ import annotations

import logging

from rag_lab.core.config import settings
from rag_lab.schemas.rag_chat import RAGChatResponse, RAGSource
from rag_lab.services.chat_service import ChatServiceError, generate_answer
from rag_lab.services.vector_store_service import RetrievedChunk, get_vector_store_service

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "Недостатньо даних у базі знань, щоб дати обґрунтовану відповідь на це питання."
)


class RAGServiceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _build_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    context_parts: list[str] = []
    for source in sources:
        context_parts.append(
            f"[{source.chunk_id}] file={source.file_name} score={source.score:.3f}\n{source.text}"
        )

    context = "\n\n".join(context_parts)
    return (
        "Ти backend-асистент. Відповідай тільки за наданим контекстом. "
        "Якщо контексту недостатньо, прямо скажи про це.\n\n"
        f"Питання користувача:\n{question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Дай коротку точну відповідь українською."
    )


async def answer_with_rag(question: str) -> RAGChatResponse:
    if not question.strip():
        raise RAGServiceError(400, "Message must not be empty")

    try:
        vector_store = get_vector_store_service()
        raw_sources = vector_store.search(
            query=question,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
    except Exception as exc:
        logger.exception("Vector retrieval failed")
        raise RAGServiceError(503, "Vector store is unavailable") from exc

    sources = [
        RAGSource(
            doc_id=item.doc_id,
            file_name=item.file_name,
            chunk_id=item.chunk_id,
            score=item.score,
            snippet=item.text[:240],
        )
        for item in raw_sources
    ]

    if not sources:
        return RAGChatResponse(answer=NO_CONTEXT_ANSWER, used_context=False, sources=[])

    prompt = _build_prompt(question=question, sources=raw_sources)
    try:
        answer = await generate_answer(prompt)
    except ChatServiceError as exc:
        raise RAGServiceError(exc.status_code, exc.detail) from exc

    return RAGChatResponse(answer=answer, used_context=True, sources=sources)
