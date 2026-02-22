import asyncio

from rag_lab.services.rag_service import NO_CONTEXT_ANSWER, answer_with_rag
from rag_lab.services.vector_store_service import RetrievedChunk


class _FakeVectorStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, *, query: str, top_k: int, score_threshold: float):
        return self._chunks


async def _fake_generate_answer(prompt: str) -> str:
    assert "Контекст:" in prompt
    return "mocked answer"


def test_answer_with_rag_retrieval_path(monkeypatch):
    chunks = [
        RetrievedChunk(
            doc_id="doc-1",
            file_name="guide.md",
            chunk_id="doc-1:0",
            score=0.91,
            text="Some useful context",
        )
    ]
    monkeypatch.setattr(
        "rag_lab.services.rag_service.get_vector_store_service",
        lambda: _FakeVectorStore(chunks),
    )
    monkeypatch.setattr(
        "rag_lab.services.rag_service.generate_answer",
        _fake_generate_answer,
    )

    result = asyncio.run(answer_with_rag("How to configure it?"))

    assert result.used_context is True
    assert result.answer == "mocked answer"
    assert len(result.sources) == 1


def test_answer_with_rag_no_context(monkeypatch):
    monkeypatch.setattr(
        "rag_lab.services.rag_service.get_vector_store_service",
        lambda: _FakeVectorStore([]),
    )

    result = asyncio.run(answer_with_rag("Unknown question"))

    assert result.used_context is False
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
