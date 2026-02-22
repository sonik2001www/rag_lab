from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_LAB_", extra="ignore")

    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    vector_store_dir: Path = Path("data/vector_store")

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.2
    max_upload_size_bytes: int = 10_000_000

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 120.0


settings = Settings()


def ensure_runtime_directories() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
