from dataclasses import dataclass

from backend.config import settings

COMPLEX_KEYWORDS = [
    "explain why",
    "debug",
    "design",
    "distributed",
    "deadlock",
    "optimize",
    "compare and contrast",
    "trade-off",
    "architecture",
    "prove",
    "analyze",
    "implement",
]


@dataclass
class RoutingDecision:
    model: str
    complexity_score: float


def score_complexity(message: str, num_chunks: int = 0) -> float:
    text = message.lower()
    score = 0.0

    score += min(len(message) / 200, 2.0)
    score += min(num_chunks * 0.3, 2.0)

    for keyword in COMPLEX_KEYWORDS:
        if keyword in text:
            score += 1.0

    if "?" in message:
        score += 0.2
    if message.count("?") > 1:
        score += 0.5

    return score


def route(message: str, num_chunks: int = 0, threshold: float = 2.5) -> RoutingDecision:
    complexity = score_complexity(message, num_chunks)
    if complexity >= threshold:
        model = settings.gemini_model_large
    else:
        model = settings.gemini_model_small
    return RoutingDecision(model=model, complexity_score=complexity)
