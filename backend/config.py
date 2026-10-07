from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "gemini"

    gemini_api_key: str = ""
    gemini_model_small: str = "gemini-3.5-flash-lite"
    gemini_model_large: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    claude_api_key: str = ""
    claude_model_small: str = "claude-3-5-haiku-latest"
    claude_model_large: str = "claude-3-5-sonnet-latest"

    enable_cache: bool = False
    enable_rag_selection: bool = False
    enable_routing: bool = False

    cache_similarity_threshold: float = 0.95
    rag_top_k: int = 5

    database_url: str = "postgresql://tokenflow:tokenflow@localhost:5432/tokenflow"
    metrics_log_path: str = "evaluation/logs/requests.jsonl"

    @property
    def active_model_small(self) -> str:
        if self.llm_provider.lower() == "claude":
            return self.claude_model_small
        return self.gemini_model_small

    @property
    def active_model_large(self) -> str:
        if self.llm_provider.lower() == "claude":
            return self.claude_model_large
        return self.gemini_model_large


settings = Settings()
