from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings
from backend.gateway.pipeline import GatewayPipeline, PipelineConfig
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

_pipeline: GatewayPipeline | None = None


def get_pipeline() -> GatewayPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GatewayPipeline()
    return _pipeline


def baseline_config() -> PipelineConfig:
    return PipelineConfig(
        enable_cache=False,
        enable_rag_selection=False,
        enable_routing=False,
    )


def optimized_config() -> PipelineConfig:
    return PipelineConfig(
        enable_cache=settings.enable_cache,
        enable_rag_selection=settings.enable_rag_selection,
        enable_routing=settings.enable_routing,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "tokenflow"}


@app.get("/config")
def config():
    return {
        "enable_cache": settings.enable_cache,
        "enable_rag_selection": settings.enable_rag_selection,
        "enable_routing": settings.enable_routing,
        "cache_similarity_threshold": settings.cache_similarity_threshold,
        "rag_top_k": settings.rag_top_k,
        "model_small": settings.gemini_model_small,
        "model_large": settings.gemini_model_large,
    }


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
        return get_pipeline().run(request, baseline_config())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc


@app.post("/chat/optimized", response_model=ChatResponse)
def chat_optimized(request: ChatRequest):
    try:
        return get_pipeline().run(request, optimized_config())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc


@app.get("/dashboard")
def dashboard():
    index = Path("dashboard/index.html")
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index)


@app.get("/dashboard/data/{name}")
def dashboard_data(name: str):
    allowed = {"latest.json", "judge_latest.json", "cache_audit.json"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown data file")
    path = Path("evaluation/results") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    return FileResponse(path, media_type="application/json")
