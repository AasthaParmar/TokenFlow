from backend.cache.semantic import CacheResult


def test_cache_result_miss():
    result = CacheResult(hit=False)
    assert result.hit is False
    assert result.answer is None


def test_cache_result_hit():
    result = CacheResult(
        hit=True,
        answer="A mutex protects shared resources.",
        matched_question="What is a mutex?",
        similarity_score=0.97,
    )
    assert result.hit is True
    assert result.similarity_score >= 0.95
