from dataclasses import dataclass

from backend.cache.semantic import SemanticCache
from backend.config import settings
from backend.llm.gemini import GeminiClient
from backend.logging.metrics import log_request
from backend.rag.selector import RAGSelector
from backend.router.complexity import route
from backend.schemas import ChatRequest, ChatResponse


@dataclass
class PipelineConfig:
    enable_cache: bool = False
    enable_rag_selection: bool = False
    enable_routing: bool = False


def get_config_flags(cfg: PipelineConfig) -> dict[str, bool]:
    return {
        "enable_cache": cfg.enable_cache,
        "enable_rag_selection": cfg.enable_rag_selection,
        "enable_routing": cfg.enable_routing,
    }


class GatewayPipeline:
    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient()
        self.cache = SemanticCache(self.client)
        self.rag = RAGSelector(self.client)

    def run(self, request: ChatRequest, cfg: PipelineConfig) -> ChatResponse:
        flags = get_config_flags(cfg)
        chunks_in = len(request.context_chunks)
        chunks_kept = chunks_in
        selected_chunks = request.context_chunks
        recall = None
        complexity_score = None
        model = settings.gemini_model_large

        if not cfg.enable_rag_selection and request.context_chunks:
            selected_chunks = request.context_chunks
            chunks_kept = len(selected_chunks)

        # Step 1: Semantic cache
        if cfg.enable_cache:
            cache_result = self.cache.lookup(request.message)
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
                self._log(request, response, recall, flags)
                return response

        # Step 2: RAG context selection
        if cfg.enable_rag_selection and request.context_chunks:
            rag_result = self.rag.select(
                request.message,
                request.context_chunks,
                request.relevant_chunk_ids,
            )
            selected_chunks = rag_result.selected_chunks
            chunks_kept = rag_result.chunks_kept
            recall = rag_result.recall

        context = "\n\n".join(selected_chunks) if selected_chunks else None

        # Step 3: Model routing
        if cfg.enable_routing:
            decision = route(request.message, num_chunks=chunks_kept)
            model = decision.model
            complexity_score = decision.complexity_score

        # Step 4: Call Gemini
        gemini_response = self.client.generate(
            message=request.message,
            model=model,
            context=context,
        )

        # Step 5: Store in cache
        if cfg.enable_cache:
            self.cache.store(request.message, gemini_response.answer)

        response = ChatResponse(
            answer=gemini_response.answer,
            model=gemini_response.model,
            input_tokens=gemini_response.input_tokens,
            output_tokens=gemini_response.output_tokens,
            latency_ms=gemini_response.latency_ms,
            cache_hit=False,
            complexity_score=complexity_score,
            chunks_in=chunks_in,
            chunks_kept=chunks_kept,
        )
        self._log(request, response, recall, flags)
        return response

    def _log(
        self,
        request: ChatRequest,
        response: ChatResponse,
        recall: float | None,
        flags: dict[str, bool],
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
                "complexity_score": response.complexity_score,
                "chunks_in": response.chunks_in,
                "chunks_kept": response.chunks_kept,
                "recall": recall,
                "config_flags": flags,
            }
        )
