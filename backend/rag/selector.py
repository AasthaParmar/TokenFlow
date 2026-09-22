from dataclasses import dataclass

from backend.config import settings
from backend.llm.gemini import GeminiClient


@dataclass
class RAGSelectionResult:
    selected_chunks: list[str]
    chunks_in: int
    chunks_kept: int
    selected_indices: list[int]
    recall: float | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGSelector:
    def __init__(self, client: GeminiClient):
        self.client = client
        self.top_k = settings.rag_top_k

    def select(
        self,
        question: str,
        chunks: list[str],
        relevant_chunk_ids: list[int] | None = None,
    ) -> RAGSelectionResult:
        if not chunks:
            return RAGSelectionResult([], 0, 0, [])

        question_emb = self.client.embed(question)
        scored = []
        for idx, chunk in enumerate(chunks):
            chunk_emb = self.client.embed(chunk)
            score = _cosine_similarity(question_emb, chunk_emb)
            scored.append((score, idx, chunk))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[: self.top_k]
        selected_indices = [idx for _, idx, _ in top]
        selected_chunks = [chunk for _, _, chunk in top]

        recall = None
        if relevant_chunk_ids is not None:
            kept = set(selected_indices)
            gold = set(relevant_chunk_ids)
            recall = len(kept & gold) / len(gold) if gold else None

        return RAGSelectionResult(
            selected_chunks=selected_chunks,
            chunks_in=len(chunks),
            chunks_kept=len(selected_chunks),
            selected_indices=selected_indices,
            recall=recall,
        )
