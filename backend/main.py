from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.cache.semantic import SemanticCache
from backend.config import settings
from backend.llm.gemini import GeminiClient
from backend.logging.metrics import log_request
from backend.rag.selector import RAGSelector
from backend.router.complexity import route
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
_rag: RAGSelector | None = None


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


def get_rag() -> RAGSelector:
    global _rag
    if _rag is None:
        _rag = RAGSelector(get_client())
    return _rag


def config_flags() -> dict[str, bool]:
    return {
        "enable_cache": settings.enable_cache,
        "enable_rag_selection": settings.enable_rag_selection,
        "enable_routing": settings.enable_routing,
    }


def prepare_context(request: ChatRequest) -> tuple[str | None, int, int, float | None]:
    chunks = request.context_chunks
    if not chunks:
        return None, 0, 0, None

    if settings.enable_rag_selection:
        rag_result = get_rag().select(
            request.message,
            chunks,
            request.relevant_chunk_ids,
        )
        context = "\n\n".join(rag_result.selected_chunks) if rag_result.selected_chunks else None
        return context, rag_result.chunks_in, rag_result.chunks_kept, rag_result.recall

    context = "\n\n".join(chunks)
    return context, len(chunks), len(chunks), None


def log_chat(
    request: ChatRequest,
    response: ChatResponse,
    flags: dict[str, bool],
    recall: float | None = None,
) -> None:
    log_request(
        {
            "question": request.message,
            "answer": response.answer,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "similarity_score": response.similarity_score,
            "matched_question": response.matched_question,
            "chunks_in": response.chunks_in,
            "chunks_kept": response.chunks_kept,
            "recall": recall,
            "complexity_score": response.complexity_score,
            "config_flags": flags,
        }
    )


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
    context, chunks_in, chunks_kept, recall = prepare_context(request)

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
                chunks_in=chunks_in,
                chunks_kept=chunks_kept,
            )
            log_chat(request, response, flags, recall)
            return response

    complexity_score = None
    model = settings.gemini_model_large
    if settings.enable_routing:
        decision = route(request.message, num_chunks=chunks_kept)
        model = decision.model
        complexity_score = decision.complexity_score

    try:
        result = get_client().generate(request.message, model=model, context=context)
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
        chunks_in=chunks_in,
        chunks_kept=chunks_kept,
        complexity_score=complexity_score,
    )
    log_chat(request, response, flags, recall)
    return response
