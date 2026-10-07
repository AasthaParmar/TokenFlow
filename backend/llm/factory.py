from backend.config import settings
from backend.llm.claude import ClaudeClient
from backend.llm.gemini import GeminiClient


def create_llm_client(provider: str | None = None):
    active_provider = (provider or settings.llm_provider).strip().lower()
    if active_provider == "claude":
        return ClaudeClient()
    return GeminiClient()