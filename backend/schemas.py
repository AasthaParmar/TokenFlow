from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool = False
    similarity_score: float | None = None
    matched_question: str | None = None
