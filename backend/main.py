from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.cache.semantic import SemanticCache
from backend.config import settings
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
_cache: SemanticCache | None = None


def get_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(get_client())
    return _cache


def config_flags() -> dict[str, bool]:
    return {
        "enable_cache": settings.enable_cache,
        "enable_rag_selection": False,
        "enable_routing": False,
    }


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
    flags = config_flags()

    if settings.enable_cache:
        cache_result = get_cache().lookup(request.message)
        if cache_result.hit:
            response = ChatResponse(
                answer=cache_result.answer or "",
                model="cache",
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                cache_hit=True,
                similarity_score=cache_result.similarity_score,
                matched_question=cache_result.matched_question,
            )
            log_request(
                {
                    "question": request.message,
                    "answer": response.answer,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "latency_ms": response.latency_ms,
                    "cache_hit": True,
                    "similarity_score": cache_result.similarity_score,
                    "matched_question": cache_result.matched_question,
                    "config_flags": flags,
                }
            )
            return response

    try:
        result = get_client().generate(request.message)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if settings.enable_cache:
        get_cache().store(request.message, result.answer)

    response = ChatResponse(
        answer=result.answer,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cache_hit=False,
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
            "config_flags": flags,
        }
    )

    return response
