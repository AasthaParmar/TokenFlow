from fastapi import FastAPI, HTTPException

from backend.llm.gemini import GeminiClient
from backend.schemas import ChatRequest, ChatResponse

app = FastAPI(title="TokenFlow", description="LLM efficiency gateway")

_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenflow"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = get_client().generate(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
