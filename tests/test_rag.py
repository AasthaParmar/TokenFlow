from backend.rag.selector import RAGSelector, _cosine_similarity


class FakeClient:
    def embed(self, text: str) -> list[float]:
        if "mutex" in text.lower():
            return [1.0, 0.0, 0.0]
        if "semaphore" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.0]


def test_cosine_similarity_identical():
    assert _cosine_similarity([1, 0], [1, 0]) == 1.0


def test_rag_selector_keeps_top_k():
    selector = RAGSelector(FakeClient())
    selector.top_k = 2
    chunks = ["about mutex", "about semaphore", "about queues"]
    result = selector.select("What is a mutex?", chunks)
    assert result.chunks_kept == 2
    assert result.chunks_in == 3


def test_rag_recall():
    selector = RAGSelector(FakeClient())
    selector.top_k = 1
    chunks = ["about mutex", "about semaphore", "about queues"]
    result = selector.select("What is a mutex?", chunks, relevant_chunk_ids=[0, 2])
    assert result.recall == 0.5
