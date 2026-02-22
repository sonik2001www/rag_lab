import pytest

from rag_lab.core.config import settings
from rag_lab.services.ingestion_service import chunk_text


def test_chunk_text_splits_long_text(monkeypatch):
    pytest.importorskip("langchain_text_splitters")

    monkeypatch.setattr(settings, "chunk_size", 40)
    monkeypatch.setattr(settings, "chunk_overlap", 10)

    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8
    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
