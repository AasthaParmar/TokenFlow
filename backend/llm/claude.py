import time
from dataclasses import dataclass

from anthropic import Anthropic

from backend.config import settings
from backend.llm.local_embeddings import LocalEmbeddingClient


@dataclass
class ClaudeResponse:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ClaudeClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or settings.claude_api_key
        if not key:
            raise ValueError("CLAUDE_API_KEY is required")
        self.client = Anthropic(api_key=key)
        self._embedding_client: LocalEmbeddingClient | None = None

    def _call_with_retry(self, fn, max_attempts: int = 10):
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code in {400, 401, 403, 404, 422}:
                    raise
                last_exc = exc
                if attempt + 1 >= max_attempts:
                    raise
                wait = min(120, 8 * (2**attempt))
                print(
                    f"Claude API retry {attempt + 1}/{max_attempts} in {wait}s "
                    f"({status_code or type(exc).__name__})",
                    flush=True,
                )
                time.sleep(wait)
        raise last_exc  # pragma: no cover

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    def generate(
        self,
        message: str,
        model: str | None = None,
        context: str | None = None,
        temperature: float = 0.7,
    ) -> ClaudeResponse:
        model_name = model or settings.active_model_large
        prompt = message
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {message}"

        start = time.perf_counter()

        def _do_generate():
            return self.client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

        response = self._call_with_retry(_do_generate)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return ClaudeResponse(
            answer=self._extract_text(response),
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def embed(self, text: str) -> list[float]:
        if self._embedding_client is None:
            self._embedding_client = LocalEmbeddingClient()
        return self._embedding_client.embed(text)

    def judge(self, prompt: str) -> str:
        result = self.generate(prompt, model=settings.active_model_large, temperature=0)
        return result.answer