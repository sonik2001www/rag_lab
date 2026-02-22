from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_LAB_", extra="ignore")

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 120.0


settings = Settings()
