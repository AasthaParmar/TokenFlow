from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.llm.gemini import GeminiClient
from backend.logging.metrics import log_request
from backend.schemas import ChatRequest, ChatResponse
from database.models import cache_count, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        pass
    yield


app = FastAPI(
    title="TokenFlow",
    description="LLM efficiency gateway",
    lifespan=lifespan,
)

_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenflow"}


@app.get("/cache/stats")
def cache_stats():
    try:
        count = cache_count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return {"cache_entries": count}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = get_client().generate(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = ChatResponse(
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )

    log_request(
        {
            "question": request.message,
            "answer": response.answer,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cache_hit": False,
            "config_flags": {
                "enable_cache": False,
                "enable_rag_selection": False,
                "enable_routing": False,
            },
        }
    )

    return response
