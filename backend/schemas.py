from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    context_chunks: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[int] | None = None


class ChatResponse(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool = False
    similarity_score: float | None = None
    matched_question: str | None = None
    chunks_in: int = 0
    chunks_kept: int = 0
    complexity_score: float | None = None
