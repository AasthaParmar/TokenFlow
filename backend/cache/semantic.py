from dataclasses import dataclass

from backend.config import settings
from backend.llm.gemini import GeminiClient
from database.models import find_similar_cache, insert_cache_entry


@dataclass
class CacheResult:
    hit: bool
    answer: str | None = None
    matched_question: str | None = None
    similarity_score: float | None = None


class SemanticCache:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.threshold = settings.cache_similarity_threshold

    def lookup(self, question: str, threshold: float | None = None) -> CacheResult:
        embedding = self.client.embed(question)
        cutoff = threshold if threshold is not None else self.threshold
        match = find_similar_cache(embedding, cutoff)
        if match:
            return CacheResult(
                hit=True,
                answer=match["answer"],
                matched_question=match["question"],
                similarity_score=float(match["similarity"]),
            )
        return CacheResult(hit=False)

    def store(self, question: str, answer: str) -> None:
        embedding = self.client.embed(question)
        insert_cache_entry(question, answer, embedding)
