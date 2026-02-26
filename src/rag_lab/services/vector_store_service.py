from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_lab.core.config import ensure_runtime_directories, settings


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    file_name: str
    chunk_id: str
    score: float
    text: str


class VectorStoreService:
    def __init__(self) -> None:
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ModuleNotFoundError as exc:
            raise RuntimeError("LangChain vector store dependencies are not installed") from exc

        ensure_runtime_directories()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store: Any = Chroma(
            collection_name="rag_lab_documents",
            persist_directory=str(settings.vector_store_dir),
            embedding_function=self._embeddings,
        )

    def delete_document(self, doc_id: str) -> None:
        self._store.delete(where={"doc_id": doc_id})

    def upsert_document_chunks(self, *, doc_id: str, file_name: str, stored_path: Path, chunks: list[str]) -> int:
        try:
            from langchain_core.documents import Document
        except ModuleNotFoundError as exc:
            raise RuntimeError("LangChain core dependency is not installed") from exc

        self.delete_document(doc_id)
        documents: list[Any] = []
        ids: list[str] = []

        for index, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}:{index}"
            metadata = {
                "doc_id": doc_id,
                "file_name": file_name,
                "stored_path": str(stored_path),
                "chunk_index": index,
                "chunk_id": chunk_id,
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
            ids.append(chunk_id)

        if not documents:
            return 0

        self._store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def search(self, *, query: str, top_k: int, score_threshold: float) -> list[RetrievedChunk]:
        pairs = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        results: list[RetrievedChunk] = []

        for document, score in pairs:
            if score < score_threshold:
                continue

            metadata = document.metadata
            results.append(
                RetrievedChunk(
                    doc_id=str(metadata.get("doc_id", "")),
                    file_name=str(metadata.get("file_name", "unknown")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    score=float(score),
                    text=document.page_content,
                )
            )

        return results


_vector_store_service: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
