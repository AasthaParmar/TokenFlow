from backend.router.complexity import route, score_complexity


def test_simple_question_routes_to_small_model():
    decision = route("What is 2 + 2?", num_chunks=0)
    assert "lite" in decision.model


def test_complex_question_routes_to_large_model():
    decision = route(
        "Explain why this distributed transaction occasionally deadlocks and propose a fix.",
        num_chunks=5,
    )
    assert "lite" not in decision.model or decision.complexity_score >= 2.5


def test_complexity_score_increases_with_chunks():
    low = score_complexity("What is a list?", num_chunks=0)
    high = score_complexity("What is a list?", num_chunks=10)
    assert high > low
