import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from backend.config import settings


@dataclass
class GeminiResponse:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class GeminiClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=key)

    def generate(
        self,
        message: str,
        model: str | None = None,
        context: str | None = None,
        temperature: float = 0.7,
    ) -> GeminiResponse:
        model_name = model or settings.gemini_model_large
        prompt = message
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {message}"

        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return GeminiResponse(
            answer=response.text or "",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def judge(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=settings.gemini_model_large,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text or ""
