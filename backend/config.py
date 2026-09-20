from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gemini_model_small: str = "gemini-2.0-flash-lite"
    gemini_model_large: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"

    enable_cache: bool = False
    enable_rag_selection: bool = False
    enable_routing: bool = False

    cache_similarity_threshold: float = 0.95
    rag_top_k: int = 5

    database_url: str = "postgresql://tokenflow:tokenflow@localhost:5432/tokenflow"
    metrics_log_path: str = "evaluation/logs/requests.jsonl"


settings = Settings()
